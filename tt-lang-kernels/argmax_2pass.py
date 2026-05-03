"""Two-pass streaming argmax over N=4064 fp32 (per-chip vocab slice).

Per-chip vocab on Galaxy: vocab_padded=130048, num_chips=32 -> 4064 = 127 tiles.
Logits dtype fp32, shape (1, 1, 1, 4064) tile-padded to (1, 1, 32, 4064).

Pass 1: streaming reduce_max -> global max value.
Pass 2: streaming reduce_max over (i + sign(v - max_global) * BIG)
        -> only the global-max element keeps its index unpenalized,
           everything else falls below -BIG, so a plain max across blocks
           finds the global argmax.

Key property: all cross-iter state uses ttl.math.max only (no BIG-penalty
mask between iters), avoiding the multi-state-DFB bug we hit with a single-
pass select-and-mask formulation.

Output: max value and argmax index in [0,0] of two 1-tile output tensors.

Tuning: K is the per-iter block width (in tiles). Sweep on sterling-all:
    K=2,  NUM_ITERS=64 -> 215.4 us
    K=8,  NUM_ITERS=16 -> 141.6 us
    K=16, NUM_ITERS=8  -> 140.2 us
    K=32, NUM_ITERS=4  -> 124.6 us   <-- sweet spot
    K=64, NUM_ITERS=2  -> 132.1 us   (block_count=1, larger block hurts)
At K=32 the kernel is dispatch-bound; further wins need multi-core.

Comparison at the same per-chip shape on sterling-all (Galaxy):
    A) ttnn.argmax + ttnn.max (TILE_LAYOUT)            2161.7 us/call
    B) to_layout(ROW_MAJOR) + ttnn.argmax(use_multicore=True)
                              + ttnn.max                287.7 us/call
    C) this kernel (single-core tt-lang two-pass)       124.6 us/call

Notes on the ttnn variants:
  - Variant A is the naive path; ttnn.argmax in TILE_LAYOUT runs single-core
    and is the slow culprit in inference.py.
  - Variant B is ~7.5x faster than A but requires a layout flip:
    ttnn.argmax(use_multicore=True) only accepts ROW_MAJOR_LAYOUT, so you
    have to to_layout(RM) first. ttnn.max can stay in TILE.
  - Variant C beats B by ~2.3x and needs no layout flip, but is single-core
    and dispatch-bound; multi-core via per-core local + final reduce is the
    next lever.
"""
import time

import torch
import ttl
import ttnn

TILE = 32
N = 4064
K = 32
N_TILES = N // TILE  # 127
N_PAD = -(-N_TILES // K) * K  # 128
NUM_ITERS = N_PAD // K  # 4


def make_kernel():
    @ttl.operation(grid=(1, 1))
    def fused_argmax(values, indices, scaler, out_value, out_index):
        # Pass 1 buffers
        v1_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, K), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        tile_max_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)
        run_max_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)

        # Bridge: hold final global max, broadcast to (1, K) for pass 2
        max_global_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)
        max_bc_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, K), block_count=2)

        # Pass 2 buffers
        v2_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, K), block_count=2)
        i_dfb = ttl.make_dataflow_buffer_like(indices, shape=(1, K), block_count=2)
        i_filt_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, K), block_count=2)
        tile_argmax_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)
        run_argmax_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)

        ov_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)
        oi_dfb = ttl.make_dataflow_buffer_like(out_index, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            s = s_dfb.wait()

            # ---- Pass 1: streaming reduce_max(v) ----
            with run_max_dfb.reserve() as init_rm:
                init_rm.store(ttl.math.fill(init_rm, -1.0e9))

            for _ in range(NUM_ITERS):
                v = v1_dfb.wait()
                rm = run_max_dfb.wait()

                tm = tile_max_dfb.reserve()
                tm.store(ttl.math.reduce_max(v, s, dims=[0, 1]))
                tm_blk = tile_max_dfb.wait()

                new_rm = run_max_dfb.reserve()
                new_rm.store(ttl.math.max(rm, tm_blk))

            # final running-max -> max_global_dfb (so pass 2 can broadcast it)
            final_rm = run_max_dfb.wait()
            mg = max_global_dfb.reserve()
            mg.store(ttl.math.max(final_rm, final_rm))  # identity copy
            mg_blk = max_global_dfb.wait()

            mgbc = max_bc_dfb.reserve()
            mgbc.store(ttl.math.broadcast(mg_blk, mgbc, dims=[0, 1]))
            mgbc_blk = max_bc_dfb.wait()

            # also write final max value to output
            ov = ov_dfb.reserve()
            ov.store(ttl.math.max(mg_blk, mg_blk))  # identity copy

            # ---- Pass 2: streaming reduce_max(i_filt) ----
            with run_argmax_dfb.reserve() as init_ra:
                init_ra.store(ttl.math.fill(init_ra, -1.0e9))

            for _ in range(NUM_ITERS):
                v = v2_dfb.wait()
                i = i_dfb.wait()
                ra = run_argmax_dfb.wait()

                with i_filt_dfb.reserve() as ifilt:
                    ifilt.store(
                        ttl.add(
                            i,
                            ttl.mul(
                                ttl.math.sign(ttl.sub(v, mgbc_blk)),
                                ttl.math.fill(ifilt, 1000000.0),
                            ),
                        )
                    )
                ifilt_blk = i_filt_dfb.wait()

                ta = tile_argmax_dfb.reserve()
                ta.store(ttl.math.reduce_max(ifilt_blk, s, dims=[0, 1]))
                ta_blk = tile_argmax_dfb.wait()

                new_ra = run_argmax_dfb.reserve()
                new_ra.store(ttl.math.max(ra, ta_blk))

            final_ra = run_argmax_dfb.wait()
            oi = oi_dfb.reserve()
            oi.store(ttl.math.max(final_ra, final_ra))  # identity copy

        @ttl.datamovement()
        def dm_read():
            blk = s_dfb.reserve()
            ttl.copy(scaler[0, 0], blk).wait()
            # Pass 1: stream values
            for it in range(NUM_ITERS):
                blk = v1_dfb.reserve()
                ttl.copy(values[0:1, it*K:(it+1)*K], blk).wait()
            # Pass 2: stream values + indices again
            for it in range(NUM_ITERS):
                blk = v2_dfb.reserve()
                ttl.copy(values[0:1, it*K:(it+1)*K], blk).wait()
                blk = i_dfb.reserve()
                ttl.copy(indices[0:1, it*K:(it+1)*K], blk).wait()

        @ttl.datamovement()
        def dm_write():
            blk = ov_dfb.wait()
            ttl.copy(blk, out_value[0, 0]).wait()
            blk = oi_dfb.wait()
            ttl.copy(blk, out_index[0, 0]).wait()

    return fused_argmax


def main():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        v_real = torch.randn(N, dtype=torch.float32)
        v_torch = torch.full((TILE, N_PAD * TILE), -1.0e9, dtype=torch.float32)
        v_torch[0, :N] = v_real
        i_torch = torch.full((TILE, N_PAD * TILE), -1.0e9, dtype=torch.float32)
        i_torch[0, :N] = torch.arange(N, dtype=torch.float32)
        s_torch = torch.ones((TILE, TILE), dtype=torch.float32)
        ov_torch = torch.zeros((TILE, TILE), dtype=torch.float32)
        oi_torch = torch.zeros((TILE, TILE), dtype=torch.float32)

        def to_dev(t):
            return ttnn.from_torch(
                t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        v_tt = to_dev(v_torch); i_tt = to_dev(i_torch); s_tt = to_dev(s_torch)
        ov_tt = to_dev(ov_torch); oi_tt = to_dev(oi_torch)

        kernel = make_kernel()
        kernel(v_tt, i_tt, s_tt, ov_tt, oi_tt)
        ov_out = ttnn.to_torch(ov_tt).flatten()[0].item()
        oi_out = int(ttnn.to_torch(oi_tt).flatten()[0].item())
        ref_max = v_real.max().item()
        ref_idx = int(v_real.argmax().item())
        ok = (abs(ov_out - ref_max) < 1e-2) and (oi_out == ref_idx)
        print(f"\n[shape] N={N}, K={K}, NUM_ITERS={NUM_ITERS}")
        print(f"[verify] max={ov_out:.6f} (ref {ref_max:.6f})  "
              f"argmax={oi_out} (ref {ref_idx})  "
              f"{'OK' if ok else 'BAD'}")

        if not ok:
            return

        for _ in range(20):
            kernel(v_tt, i_tt, s_tt, ov_tt, oi_tt)
        ttnn.synchronize_device(device)

        n_iter = 200
        t0 = time.perf_counter()
        for _ in range(n_iter):
            kernel(v_tt, i_tt, s_tt, ov_tt, oi_tt)
        ttnn.synchronize_device(device)
        us = (time.perf_counter() - t0) / n_iter * 1e6
        print(f"[time]  {us:.1f} us/call")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
