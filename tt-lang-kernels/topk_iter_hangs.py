"""STATUS: HANGS THE DEVICE — DO NOT USE.

This file is preserved as a starting point for future iteration. The current
design (DRAM-resident `mask` scratch + dm_write walking every tile in a
compile-time loop with `if tile_idx == best_tile: element_write(...)`) hung
on sterling-all.conf during initial bring-up and the run was never recovered.

Open issues to revisit before re-enabling:
  1. Mask should live in L1 / a CB across iterations, not in DRAM. The
     current design pays K_FIXED * n_pad_tiles DRAM round-trips per call,
     which is worse than the ttnn baseline AND doesn't match the
     "everything stays in SRAM" mega-kernel goal.
  2. dm_write reserve/wait pairs across the compute / dm_read / dm_write
     threads need to be re-counted; a hang most often means a dfb is
     double-reserved or never popped (see
     `feedback_ttl_dfb_balance.md`).
  3. The compile-time `for tile_idx in range(n_pad_tiles)` walk inside
     dm_write may itself be the hang trigger when combined with the inner
     `with update_dfb.reserve(): ... with .wait()` pattern (see
     `feedback_ttl_loop_dfb_still_buggy.md`).

Strategic note (per discussion 2026-05-04): topk is NOT on the critical
path for any mega-kernel single-zone fusion. test_lk_d_topk.py already
fuses to one ttl.operation (mask_build) with topk + epilogue as trailing
ttnn ceremony; test_lk_f.py is gated on bucket #7 (batched per-chip-local
SUMMA) which is a real primitive gap downstream of topk. So unblocking
topk doesn't move the fusion needle. Higher-leverage primitives to focus
on first: bucket #1 (paged_update_cache via element_write) and bucket #2
(cos/sin row gather via element_read), each of which dissolves a non-
trailing op in many zones.

Original design notes follow.

Iterative top-K via repeated streaming argmax with element_write masking.

Design:
  Inside one ttl.operation, run K_FIXED outer iterations of streaming
  reduce_max over `(values + mask)`. After each iteration's argmax winner
  is found via an element_read scan in the dm_write thread, mutate
  `mask` DRAM at the winning lane to -1e9 via element_write + ttl.copy
  round-trip. Iteration k+1's reduce then sees the winner removed.

  We avoid runtime tile-column indexing (which the frontend rejects with
  shape errors) by walking ALL tiles in a compile-time loop and using a
  cheap `if tile_idx == best_tile` conditional inside DM to decide which
  lane to mask. For K_FIXED=8 this costs O(K * n_pad_tiles) tile copies
  per topk call.

Synchronization:
  - sync_dfb hands off between dm_write (mask done writing iter k) and
    dm_read (about to read mask for iter k+1).

Inputs:
  values:      [TILE, n_pad_tiles*TILE]   dtype  (read-only; valid data
                                                  in row 0, cols [0, N),
                                                  -1e9 elsewhere)
  mask:        [TILE, n_pad_tiles*TILE]   dtype  (scratch; caller MUST
                                                  zero before each call)
  scaler:      [TILE, TILE]               dtype  (ones tile for reduce)
  out_indices: [TILE, TILE]               dtype  (output: row 0 holds
                                                  K_FIXED (tile, lane)
                                                  pairs in lanes
                                                  [0, 2*K_FIXED). Host
                                                  decodes via
                                                  tile*TILE + lane.)

Sizes covered by the standalone test:
  (N, K_FIXED) = (256, 8)   for Lk-F gate post topk
                 (2048, 8)  for Lk-D-topk bucketed topk
                 (4096, 1)  forward sanity (single-stage Final argmax
                            over a 4096-element slice; full vocab=129280
                            stays on ttnn.argmax until hierarchical lands)
"""

import argparse
import time

import torch
import ttl
import ttnn


TILE = 32

_NEG_BIG_BITS = {
    torch.bfloat16: 0xCE6E,
    torch.float32:  0x286B6ECE,
}


def _ttnn_dtype(torch_dtype):
    if torch_dtype == torch.bfloat16:
        return ttnn.bfloat16
    if torch_dtype == torch.float32:
        return ttnn.float32
    raise ValueError(f"unsupported dtype {torch_dtype}")


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def make_iterative_topk(N, K_FIXED, *, dtype=torch.float32):
    """Build a (1,1)-grid iterative topk kernel.

    Returns (kernel, n_pad_tiles, neg_big_i32).
    """
    if dtype not in _NEG_BIG_BITS:
        raise ValueError(f"unsupported dtype {dtype}")
    if K_FIXED < 1:
        raise ValueError(f"K_FIXED must be >= 1, got {K_FIXED}")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if K_FIXED > N:
        raise ValueError(f"K_FIXED={K_FIXED} cannot exceed N={N}")

    n_tiles = _round_up(N, TILE) // TILE
    n_pad_tiles = n_tiles
    neg_big_i32 = _NEG_BIG_BITS[dtype]

    @ttl.operation(grid=(1, 1))
    def iter_topk(values, mask, scaler, out_indices):
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        v_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, 1), block_count=2)
        m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), block_count=2)
        vm_local_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, 1), block_count=2)
        pass_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, 1), block_count=2)

        tile_max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        run_max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        max_global_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        out_dfb = ttl.make_dataflow_buffer_like(out_indices, shape=(1, 1), block_count=2)
        update_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), block_count=2)
        sync_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            s = s_dfb.wait()

            for k in range(K_FIXED):
                with run_max_dfb.reserve() as init:
                    init.store(ttl.math.fill(init, -1.0e9))

                for _ in range(n_pad_tiles):
                    v = v_dfb.wait()
                    m = m_dfb.wait()

                    with vm_local_dfb.reserve() as vm:
                        vm.store(ttl.add(v, m))
                    vm_blk = vm_local_dfb.wait()

                    with pass_dfb.reserve() as pp:
                        pp.store(vm_blk)

                    with tile_max_dfb.reserve() as tm:
                        tm.store(ttl.math.reduce_max(vm_blk, s, dims=[0, 1]))
                    tm_blk = tile_max_dfb.wait()
                    rm = run_max_dfb.wait()
                    with run_max_dfb.reserve() as new_rm:
                        new_rm.store(ttl.math.max(rm, tm_blk))

                final = run_max_dfb.wait()
                with max_global_dfb.reserve() as mg:
                    mg.store(ttl.math.max(final, final))

        @ttl.datamovement()
        def dm_read():
            with s_dfb.reserve() as blk:
                ttl.copy(scaler[0, 0], blk).wait()

            for k in range(K_FIXED):
                if k > 0:
                    with sync_dfb.wait() as _sig:
                        pass
                for tile_idx in range(n_pad_tiles):
                    with v_dfb.reserve() as blk:
                        ttl.copy(values[0, tile_idx], blk).wait()
                    with m_dfb.reserve() as blk:
                        ttl.copy(mask[0, tile_idx], blk).wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.reserve() as oblk:
                for k in range(K_FIXED):
                    with max_global_dfb.wait() as mg_blk:
                        max_val = ttl.element_read(mg_blk, 0, 0)

                    best_tile = 0
                    best_lane = 0
                    for tile_idx in range(n_pad_tiles):
                        with pass_dfb.wait() as v_blk:
                            for c in range(TILE):
                                v = ttl.element_read(v_blk, 0, c)
                                if v == max_val:
                                    best_tile = tile_idx
                                    best_lane = c

                    ttl.element_write(oblk, 0, 2 * k, best_tile)
                    ttl.element_write(oblk, 0, 2 * k + 1, best_lane)

                    # Update mask DRAM at (best_tile, best_lane).
                    # Walk every tile (compile-time), only patch the
                    # one that equals best_tile. Avoids runtime
                    # tile-col indexing which the frontend doesn't
                    # accept here.
                    for tile_idx in range(n_pad_tiles):
                        with update_dfb.reserve() as up_blk:
                            ttl.copy(
                                mask[0, tile_idx], up_blk
                            ).wait()
                            if tile_idx == best_tile:
                                ttl.element_write(
                                    up_blk, 0, best_lane, neg_big_i32
                                )
                            ttl.copy(
                                up_blk, mask[0, tile_idx]
                            ).wait()

                    if k < K_FIXED - 1:
                        with sync_dfb.reserve() as _sig:
                            pass

                ttl.copy(oblk, out_indices[0, 0]).wait()
                oblk.pop()

    return iter_topk, n_pad_tiles, neg_big_i32


# ---------------------------------------------------------------------------
# Standalone PCC test
# ---------------------------------------------------------------------------

def _build_inputs(N, K_FIXED, n_pad_tiles, dtype, device, seed):
    torch.manual_seed(seed)
    n_pad_elems = n_pad_tiles * TILE

    if dtype == torch.bfloat16:
        v_real = torch.randn(N, dtype=torch.float32).to(torch.bfloat16)
    else:
        v_real = torch.randn(N, dtype=dtype)

    v_torch = torch.full((TILE, n_pad_elems), -1.0e9, dtype=dtype)
    v_torch[0, :N] = v_real

    m_torch = torch.zeros((TILE, n_pad_elems), dtype=dtype)
    s_torch = torch.ones((TILE, TILE), dtype=dtype)
    o_torch = torch.zeros((TILE, TILE), dtype=dtype)

    ttnn_dt = _ttnn_dtype(dtype)
    def to_dev(t):
        return ttnn.from_torch(
            t.contiguous(), dtype=ttnn_dt, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    v_tt = to_dev(v_torch)
    m_tt = to_dev(m_torch)
    s_tt = to_dev(s_torch)
    o_tt = to_dev(o_torch)

    ref_vals, ref_idxs = torch.topk(v_real.float(), K_FIXED, largest=True, sorted=True)
    return v_tt, m_tt, s_tt, o_tt, ref_idxs.tolist(), ref_vals.tolist()


def _run_one(N, K_FIXED, dtype, device, kernel_cache, seed=0):
    key = (N, K_FIXED, str(dtype))
    if key not in kernel_cache:
        kern, n_pad_tiles, _ = make_iterative_topk(N, K_FIXED, dtype=dtype)
        kernel_cache[key] = (kern, n_pad_tiles)
    kern, n_pad_tiles = kernel_cache[key]

    v_tt, m_tt, s_tt, o_tt, ref_idxs, ref_vals = _build_inputs(
        N, K_FIXED, n_pad_tiles, dtype, device, seed=seed)

    kern(v_tt, m_tt, s_tt, o_tt)

    out_pairs = ttnn.to_torch(o_tt).flatten()[: 2 * K_FIXED].tolist()
    out_int = [
        int(out_pairs[2 * k]) * TILE + int(out_pairs[2 * k + 1])
        for k in range(K_FIXED)
    ]

    matches = sum(1 for a, b in zip(out_int, ref_idxs) if a == b)
    status = "PASS" if matches == K_FIXED else f"PARTIAL {matches}/{K_FIXED}"
    print(f"  N={N:5d} K={K_FIXED:2d} dtype={str(dtype):>15s}  "
          f"got={out_int}  ref={ref_idxs}  [{status}]")
    return matches == K_FIXED


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--bench", action="store_true")
    args = p.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    cache: dict = {}
    try:
        print("=== iterative_argmax_topk PCC ===")
        _run_one(256, 8, torch.float32, device, cache, seed=0)
        _run_one(2048, 8, torch.float32, device, cache, seed=1)
        _run_one(4096, 1, torch.float32, device, cache, seed=2)
        _run_one(256, 8, torch.bfloat16, device, cache, seed=3)

        if args.bench:
            print("\n=== bench (Lk-D-topk shape: N=2048 K=8 fp32) ===")
            kern, n_pad_tiles = cache[(2048, 8, str(torch.float32))]
            v_tt, m_tt, s_tt, o_tt, _, _ = _build_inputs(
                2048, 8, n_pad_tiles, torch.float32, device, seed=99)
            for _ in range(5):
                kern(v_tt, m_tt, s_tt, o_tt)
            ttnn.synchronize_device(device)
            n_iter = 50
            t0 = time.perf_counter()
            for _ in range(n_iter):
                kern(v_tt, m_tt, s_tt, o_tt)
            ttnn.synchronize_device(device)
            us = (time.perf_counter() - t0) / n_iter * 1e6
            print(f"  N=2048 K=8 fp32: {us:.1f} us/call")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
