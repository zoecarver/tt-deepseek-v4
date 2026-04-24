"""TT-Lang mhc_pre_apply_mix (kernel #2b in ../kernels.md).

Reference: TileKernels/tile_kernels/mhc/pre_apply_mix_kernel.py (tilelang).
Reference test: TileKernels/tests/mhc/test_pre_apply_mix.py.
CPU torch reference: `torch_refs.mhc_pre_apply_mix_ref`.

Given
  x:   [n0, n1, mhc, h]    fp32
  mix: [n0, n1, mhc, 1]    fp32
computes
  out[n0, n1, h] = sum_m x[n0, n1, m, :] * mix[n0, n1, m, 0]
which is a per-token weighted sum over the mhc axis (equivalent to per-token
matmul of mix^T @ x).

Packing (one token per 32-row tile-column block):
  - x:   [num_tokens * 32, h] fp32. For each token t, rows [t*32 : t*32+mhc]
    are the mhc vectors; rows [t*32+mhc : (t+1)*32] are zero. h must be a
    multiple of TILE.
  - mix: [num_tokens * 32, 32] fp32. For each token, col 0 of rows
    [0 : mhc] holds the mhc scalars; rest is zero.
  - out: [num_tokens * 32, h] fp32. Row 0 of each 32-row block carries the
    result; rows 1..31 are don't-care.

Per token we do once:
  mix_bc[32, 32] = broadcast(mix_col0, dims=[1])    # each row r -> mix[r] in every col
then per h-tile:
  prod   = x_tile * mix_bc                          # rows 0..mhc-1 have x*mix; rest 0
  out    = reduce_sum(prod, scaler, dims=[0])       # column sums collapse rows → (1, 32)
"""
from __future__ import annotations

import ttl

TILE = 32


def make_kernel(num_tokens: int, h_tiles: int):

    @ttl.operation(grid="auto")
    def apply_mix_kernel(x, mix, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tokens_per_core = -(-num_tokens // total_cores)

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

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    mix_raw = mix_dfb.wait()
                    mx = mix_bc_dfb.reserve()
                    mx.store(ttl.math.broadcast(mix_raw, mx, dims=[1]))
                    mix_bc = mix_bc_dfb.wait()

                    for _ in range(h_tiles):
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
            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    ttl.copy(mix[global_t, 0], mix_dfb.reserve()).wait()
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    for h in range(h_tiles):
                        ttl.copy(out_dfb.wait(), out[global_t, h]).wait()

    return apply_mix_kernel


def pack_x(x, dtype):
    """Pack [num_tokens, mhc, h] -> [num_tokens * 32, h] with zero-padded rows.

    Rows [t*32 : t*32+mhc] of output correspond to x[t, :, :]; rows
    [t*32+mhc : (t+1)*32] are zero so they contribute nothing to the per-token
    reduce_sum.
    """
    import torch
    num_tokens, mhc, h = x.shape
    assert h % TILE == 0, f"h={h} must be a multiple of {TILE}"
    assert mhc <= TILE, f"mhc={mhc} must fit in a tile row (<={TILE})"
    out = torch.zeros(num_tokens * TILE, h, dtype=dtype)
    out.view(num_tokens, TILE, h)[:, :mhc, :] = x.to(dtype)
    return out.contiguous()


def pack_mix(mix, dtype):
    """Pack [num_tokens, mhc, 1] -> [num_tokens * 32, 32].

    Col 0 of rows [t*32 : t*32+mhc] carries the mhc scalars; rest is zero.
    """
    import torch
    num_tokens, mhc, one = mix.shape
    assert one == 1, f"expected last dim 1, got {one}"
    assert mhc <= TILE, f"mhc={mhc} must fit in a tile row (<={TILE})"
    out = torch.zeros(num_tokens * TILE, TILE, dtype=dtype)
    out.view(num_tokens, TILE, TILE)[:, :mhc, 0] = mix.to(dtype).squeeze(-1)
    return out.contiguous()


def unpack_out(packed, num_tokens, h):
    """Inverse of pack_x result-wise: take row 0 of each 32-row block.

    The kernel writes a full 32x32 tile per (token, h-tile) but only row 0 is
    the valid column-sum; rows 1..31 are don't-care.
    """
    return packed.view(num_tokens, TILE, h)[:, 0, :].contiguous()


def solve(x_tt, mix_tt, scaler_tt, out_tt):
    """Run apply_mix. x/out are [num_tokens*32, h] fp32, mix is
    [num_tokens*32, 32] fp32, scaler is a [32, 32] tile of ones.
    """
    num_tokens = x_tt.shape[0] // TILE
    h_tiles = x_tt.shape[1] // TILE
    kernel = make_kernel(num_tokens=num_tokens, h_tiles=h_tiles)
    kernel(x_tt, mix_tt, scaler_tt, out_tt)


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _test_shape(device, ttnn, torch, n0, n1, mhc, h, threshold):
    from harness import assert_pcc, scaler_tile
    from torch_refs import mhc_pre_apply_mix_ref

    num_tokens = n0 * n1
    print(f"\n[shape] n0={n0} n1={n1} mhc={mhc} h={h} num_tokens={num_tokens}")

    torch.manual_seed(0)
    x_ref = torch.randn((n0, n1, mhc, h), dtype=torch.bfloat16).sigmoid()
    mix_ref = torch.randn((n0, n1, mhc, 1), dtype=torch.float32).softmax(-2)

    y_ref = mhc_pre_apply_mix_ref(x_ref, mix_ref)

    x_flat = x_ref.reshape(num_tokens, mhc, h).float()
    mix_flat = mix_ref.reshape(num_tokens, mhc, 1)

    x_packed = pack_x(x_flat, dtype=torch.float32)
    mix_packed = pack_mix(mix_flat, dtype=torch.float32)
    out_packed = torch.zeros_like(x_packed)
    sc = scaler_tile(dtype=torch.float32)

    common = dict(
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.from_torch(x_packed, **common)
    mix_tt = ttnn.from_torch(mix_packed, **common)
    out_tt = ttnn.from_torch(out_packed, **common)
    sc_tt = ttnn.from_torch(sc, **common)

    solve(x_tt, mix_tt, sc_tt, out_tt)

    out_packed = ttnn.to_torch(out_tt)
    y_tt = unpack_out(out_packed, num_tokens, h).reshape(n0, n1, h).bfloat16()

    assert_pcc(y_ref.float(), y_tt.float(), threshold=threshold)


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    THRESHOLD = 0.999

    SHAPES = [
        # (n0, n1, mhc, h)
        (1, 1024, 4, 1280),
        (1, 1024, 4, 2560),
        (1, 1024, 4, 4096),
        (2, 1024, 4, 4096),
        (1, 4096, 4, 4096),
    ]

    device = ttnn.open_device(device_id=0)
    try:
        for n0, n1, mhc, h in SHAPES:
            _test_shape(device, ttnn, torch, n0, n1, mhc, h, THRESHOLD)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)
