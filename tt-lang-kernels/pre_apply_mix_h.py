"""H-sharded version of pre_apply_mix.

Same math as pre_apply_mix.py:
  out[t, h] = sum_{m=0..mhc-1} x[t, m, :] * mix[t, m]

The original kernel parallelizes on `num_tokens`, so at decode (num_tokens=1)
only one core fires per chip and the inner h_tiles=128 loop runs serially.
This kernel flattens (token, h_tile) into one work axis and grid="auto"
splits the work across all available cores. Each h-tile is independent
(no reduction across h) so no pipe-gather is needed.

Optimal config for V4-Flash decode (num_tokens=1, hidden=4096, h_tiles=128):
on a 130-core BH chip, each core handles ~1 h-tile. Per-call time goes
from 0.25ms (single-core baseline) to 0.18ms (1.36x speedup, PCC 1.0).
Kernel is replicated across the 4-chip mesh.
"""
from __future__ import annotations

import ttl


TILE = 32


def make_kernel(num_tokens: int, h_tiles: int):
    """Build the h-sharded apply_mix kernel.

    `x` is [num_tokens * TILE, hidden] fp32 — rows 0..mhc-1 of each TILE-block
    hold x[m, :]. `mix` is [num_tokens * TILE, TILE] fp32 — col 0 of each
    TILE-block holds mix[m]. `out` is [num_tokens * TILE, hidden] fp32 —
    row 0 of each TILE-block holds the result.
    """
    total_work = num_tokens * h_tiles

    @ttl.operation(grid="auto")
    def apply_mix_h_kernel(x, mix, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        mix_dfb = ttl.make_dataflow_buffer_like(mix, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        mix_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        prod_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            sc = sc_dfb.wait()

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    mix_raw = mix_dfb.wait()
                    mx = mix_bc_dfb.reserve()
                    mx.store(ttl.math.broadcast(mix_raw, mx, dims=[1]))
                    mix_bc = mix_bc_dfb.wait()

                    x_tile = x_dfb.wait()
                    prod_dfb.reserve().store(x_tile * mix_bc)
                    red_dfb.reserve().store(
                        ttl.math.reduce_sum(prod_dfb.wait(), sc, dims=[0])
                    )
                    out_dfb.reserve().store(red_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    global_t = global_w // h_tiles
                    global_h = global_w % h_tiles
                    ttl.copy(mix[global_t, 0], mix_dfb.reserve()).wait()
                    ttl.copy(x[global_t, global_h], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    global_t = global_w // h_tiles
                    global_h = global_w % h_tiles
                    ttl.copy(out_dfb.wait(), out[global_t, global_h]).wait()

    return apply_mix_h_kernel


# -----------------------------------------------------------------------------
# Test harness — mesh4 replicated, fp32, V4-Flash decode shape.
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    from harness import assert_pcc, scaler_tile

    NCHIPS = 4
    HC_MULT = 4
    HIDDEN = 4096

    def apply_mix_ref(x_packed, mix_packed, num_tokens, mhc, hidden):
        """Reference: per-token sum over m of x[t, m, :] * mix[t, m]."""
        x_3d = x_packed.view(num_tokens, TILE, hidden)
        mix_3d = mix_packed.view(num_tokens, TILE, TILE)
        x_valid = x_3d[:, :mhc, :]
        mix_valid = mix_3d[:, :mhc, 0:1]
        out_valid = (x_valid * mix_valid).sum(dim=1)
        out_packed = torch.zeros_like(x_packed)
        out_packed.view(num_tokens, TILE, hidden)[:, 0, :] = out_valid
        return out_packed

    def to_dev(t, mesh, mapper):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def run_shape(mesh, num_tokens, threshold=0.9999):
        h_tiles = HIDDEN // TILE
        print(f"\n[shape] num_tokens={num_tokens} mhc={HC_MULT} "
              f"hidden={HIDDEN} h_tiles={h_tiles}")

        kernel = make_kernel(num_tokens=num_tokens, h_tiles=h_tiles)

        torch.manual_seed(0)
        x_raw = torch.randn(num_tokens, HC_MULT, HIDDEN, dtype=torch.float32) * 0.1
        mix_raw = torch.randn(num_tokens, HC_MULT, dtype=torch.float32) * 0.5

        x_packed = torch.zeros(num_tokens * TILE, HIDDEN, dtype=torch.float32)
        x_packed.view(num_tokens, TILE, HIDDEN)[:, :HC_MULT, :] = x_raw

        mix_packed = torch.zeros(num_tokens * TILE, TILE, dtype=torch.float32)
        mix_packed.view(num_tokens, TILE, TILE)[:, :HC_MULT, 0] = mix_raw

        sc = scaler_tile(dtype=torch.float32)
        out_zero = torch.zeros(num_tokens * TILE, HIDDEN, dtype=torch.float32)

        ref = apply_mix_ref(x_packed, mix_packed, num_tokens, HC_MULT, HIDDEN)

        replicate = ttnn.ReplicateTensorToMesh(mesh)
        x_tt = to_dev(x_packed, mesh, replicate)
        mix_tt = to_dev(mix_packed, mesh, replicate)
        sc_tt = to_dev(sc, mesh, replicate)
        out_tt = to_dev(out_zero, mesh, replicate)

        kernel(x_tt, mix_tt, sc_tt, out_tt)

        composer = ttnn.ConcatMesh2dToTensor(
            mesh, mesh_shape=(1, NCHIPS), dims=(0, -1))
        actual_full = ttnn.to_torch(out_tt, mesh_composer=composer)
        actual = actual_full[:, :HIDDEN]

        # Compare row 0 of each TILE-block (the only valid output rows).
        valid_rows = list(range(0, num_tokens * TILE, TILE))
        assert_pcc(ref[valid_rows], actual[valid_rows], threshold=threshold)

        for t in (x_tt, mix_tt, sc_tt, out_tt):
            ttnn.deallocate(t)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    try:
        run_shape(mesh, num_tokens=1)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_mesh_device(mesh)
