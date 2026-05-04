"""Targeted repro for argmax_2pass scale-dependent rounding bug.

Bug per BUGS.md: at NUM_ITERS=4, the kernel returns correct non-tile-aligned
argmax. At NUM_ITERS=127 it returns the tile-start of the correct tile
(e.g. 113280 instead of 113306 = 113280 + 26).

This repro pins down WHICH variable triggers the failure by sweeping each
axis with strictly-monotonic fp32 inputs (no ties) and a sentinel winner at
a controlled position.

Hypotheses:
  H1 (loop count): the bug appears as NUM_ITERS crosses some threshold,
                    independent of where the winner sits.
  H2 (winner iter): the bug depends on which iter contains the winner.
  H3 (within-tile lane): pass-2 always rounds to lane 0 at high NUM_ITERS,
                         regardless of which lane holds the winner.
  H4 (pass-1 max corruption): the global max VALUE is itself wrong at high
                              NUM_ITERS (would be visible in got_val).

For each test, prints:
  NUM_ITERS=N K=k winner_idx=W (iter=i, tile_in_iter=t, lane=l)
    -> got_val=V (exp E)  got_idx=G (exp W)  [STATUS]

STATUS is one of:
  PASS                    - both value and index match
  TILE_ALIGNED            - value matches, index is lane-0 of the correct tile
                            (this is the documented failure mode)
  WRONG_TILE              - value matches, index points to wrong tile entirely
  VAL_BAD                 - max value itself is wrong (Pass-1 issue)
"""
import argparse
import torch
import ttl
import ttnn

TILE = 32


def make_kernel(K: int, NUM_ITERS: int):
    """Build a fresh argmax_2pass kernel with K and NUM_ITERS captured."""
    @ttl.operation(grid=(1, 1))
    def fused_argmax(values, indices, scaler, out_value, out_index):
        v1_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, K), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        tile_max_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)
        run_max_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)

        max_global_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)
        max_bc_dfb = ttl.make_dataflow_buffer_like(values, shape=(1, K), block_count=2)

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

            final_rm = run_max_dfb.wait()
            mg = max_global_dfb.reserve()
            mg.store(ttl.math.max(final_rm, final_rm))
            mg_blk = max_global_dfb.wait()

            mgbc = max_bc_dfb.reserve()
            mgbc.store(ttl.math.broadcast(mg_blk, mgbc, dims=[0, 1]))
            mgbc_blk = max_bc_dfb.wait()

            ov = ov_dfb.reserve()
            ov.store(ttl.math.max(mg_blk, mg_blk))

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
            oi.store(ttl.math.max(final_ra, final_ra))

        @ttl.datamovement()
        def dm_read():
            ttl.copy(scaler[0, 0], s_dfb.reserve()).wait()
            for it in range(NUM_ITERS):
                ttl.copy(values[0:1, it*K:(it+1)*K], v1_dfb.reserve()).wait()
            for it in range(NUM_ITERS):
                ttl.copy(values[0:1, it*K:(it+1)*K], v2_dfb.reserve()).wait()
                ttl.copy(indices[0:1, it*K:(it+1)*K], i_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            ttl.copy(ov_dfb.wait(), out_value[0, 0]).wait()
            ttl.copy(oi_dfb.wait(), out_index[0, 0]).wait()

    return fused_argmax


def build_inputs(K, NUM_ITERS, winner_idx):
    """Strictly-monotonic arange + sentinel winner at winner_idx.

    Padded shape: (TILE, K * NUM_ITERS * TILE). Only row 0 carries data.
    """
    n_elems = K * NUM_ITERS * TILE
    assert 0 <= winner_idx < n_elems

    # Strictly monotonic: 0.0, 1.0, ..., n_elems-1.0 (no ties).
    # Sentinel winner: n_elems + 100.0 (clearly distinct).
    v_row = torch.arange(n_elems, dtype=torch.float32)
    expect_runner_up = float(n_elems - 1)  # second largest (or first if winner==N-1)
    v_row[winner_idx] = float(n_elems) + 100.0
    expect_val = v_row[winner_idx].item()

    i_row = torch.arange(n_elems, dtype=torch.float32)

    v_padded = torch.full((TILE, n_elems), -1.0e9, dtype=torch.float32)
    v_padded[0, :] = v_row
    i_padded = torch.full((TILE, n_elems), -1.0e9, dtype=torch.float32)
    i_padded[0, :] = i_row
    s = torch.ones((TILE, TILE), dtype=torch.float32)
    return v_padded, i_padded, s, expect_val, expect_runner_up


def to_dev(t, device):
    return ttnn.from_torch(
        t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def run_one(device, K, NUM_ITERS, winner_idx, kernel_cache):
    v_padded, i_padded, s, expect_val, _ = build_inputs(K, NUM_ITERS, winner_idx)

    v_tt = to_dev(v_padded, device)
    i_tt = to_dev(i_padded, device)
    s_tt = to_dev(s, device)
    ov_tt = to_dev(torch.zeros((TILE, TILE), dtype=torch.float32), device)
    oi_tt = to_dev(torch.zeros((TILE, TILE), dtype=torch.float32), device)

    key = (K, NUM_ITERS)
    if key not in kernel_cache:
        kernel_cache[key] = make_kernel(K, NUM_ITERS)
    kernel = kernel_cache[key]
    kernel(v_tt, i_tt, s_tt, ov_tt, oi_tt)

    got_val = ttnn.to_torch(ov_tt).flatten()[0].item()
    got_idx = int(ttnn.to_torch(oi_tt).flatten()[0].item())

    tile = winner_idx // TILE
    lane = winner_idx % TILE
    iter_idx = tile // K
    in_iter_tile = tile % K

    val_ok = abs(got_val - expect_val) < 0.5
    idx_ok = got_idx == winner_idx

    if val_ok and idx_ok:
        status = "PASS"
    elif not val_ok:
        status = f"VAL_BAD (delta={got_val - expect_val:+.3f})"
    else:
        got_tile = got_idx // TILE
        got_lane = got_idx % TILE
        if got_tile == tile and got_lane == 0:
            status = "TILE_ALIGNED"
        elif got_tile == tile:
            status = f"WITHIN_TILE_WRONG_LANE (got_lane={got_lane})"
        else:
            status = f"WRONG_TILE (got_tile={got_tile} got_lane={got_lane})"

    print(f"  NUM_ITERS={NUM_ITERS:3d} K={K:2d} winner={winner_idx:6d} "
          f"(iter={iter_idx:3d}, tile_in_iter={in_iter_tile:2d}, lane={lane:2d}) "
          f"-> got_val={got_val:11.2f} (exp {expect_val:11.2f})  "
          f"got_idx={got_idx:6d}  [{status}]")
    return status.startswith("PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="all",
                   choices=["scale", "position", "lane", "k", "all"])
    p.add_argument("--device-id", type=int, default=0)
    args = p.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    kernel_cache: dict = {}
    try:
        if args.mode in ("scale", "all"):
            print("=== H1: NUM_ITERS sweep, winner fixed at iter=0 tile=2 lane=17 ===")
            K = 32
            for ni in [4, 8, 16, 32, 64, 96, 127]:
                wi = 2 * TILE + 17  # tile 2 lane 17 of iter 0
                run_one(device, K, ni, wi, kernel_cache)

        if args.mode in ("position", "all"):
            print("\n=== H2: NUM_ITERS=127, winner at varying iter, tile_in_iter=0 lane=17 ===")
            K = 32
            for iter_idx in [0, 1, 2, 4, 10, 50, 100, 126]:
                tile = iter_idx * K  # tile 0 within the iter
                wi = tile * TILE + 17
                run_one(device, K, 127, wi, kernel_cache)

        if args.mode in ("lane", "all"):
            print("\n=== H3: NUM_ITERS=127, winner in tile=50, lane sweep ===")
            K = 32
            tile = 50  # tile 50 = iter 1 tile 18
            for lane in [0, 1, 5, 13, 17, 26, 31]:
                wi = tile * TILE + lane
                run_one(device, K, 127, wi, kernel_cache)

        if args.mode in ("k", "all"):
            print("\n=== H4: K (block width) sweep at fixed total tiles=128 ===")
            # winner at element 113306 - 113280 = lane 26 of tile 3540
            # but 3540 is too big for total_tiles=128. Use a proxy:
            # winner in mid-range tile, lane 17.
            for K, NI in [(2, 64), (4, 32), (8, 16), (16, 8), (32, 4)]:
                # winner around 70% of total range, lane 17
                total = K * NI
                tile = int(total * 0.7)
                wi = tile * TILE + 17
                run_one(device, K, NI, wi, kernel_cache)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
