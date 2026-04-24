"""TT-Lang mhc_pre_split_mixes (kernel #2a in ../kernels.md).

Reference: TileKernels/tile_kernels/mhc/pre_split_mixes_kernel.py (tilelang).
Reference test: TileKernels/tests/mhc/test_pre_split_mixes.py.
CPU torch reference: `torch_refs.mhc_pre_split_mixes_ref`.

Takes input_mixes [n0, n1, mhc_mult3] fp32 plus scale[3]/base[mhc_mult3] and
produces three outputs:
  - pre_layer_mix:   sigmoid(scaled[:mhc_mult]) + eps
  - post_layer_mix:  sigmoid(scaled[mhc_mult:2*mhc_mult]) * post_mult_value
  - comb_res_mix:    scaled[2*mhc_mult:]  reshaped to (mhc_mult, mhc_mult)

where `scaled = input_mixes * scale_broadcast + base`. `scale_broadcast`
expands the three scale entries across the three sections (mhc_mult each for
the first two, mhc_mult**2 for the third).

Packing (for mhc_mult=4, mhc_mult3=24):
  - input_mixes: [num_tokens, 32] — each row is one token's 24-wide vector,
    cols 24-31 padded to 0. num_tokens must be a multiple of TILE=32.
  - scale_tile, base_tile, mask tiles: all [32, 32], each row identical
    (broadcast). Valid region within a row matches the expected column range.
  - Outputs pre_out/post_out/comb_out: same [num_tokens, 32] layout as
    input; each has zeros outside its valid column range.

Kernel is pure elementwise, so no reduce/broadcast. We use 3 mask tiles to
zero out the sections each output doesn't want, plus a pre_eps_tile that
adds `mhc_pre_eps` only in the pre section.
"""
from __future__ import annotations

import ttl

TILE = 32


def make_kernel(num_tiles: int):

    @ttl.operation(grid="auto")
    def split_mixes_kernel(
        input_mixes, scale_tile, base_tile,
        pre_mask, pre_eps_tile, post_mult_mask, comb_mask,
        pre_out, post_out, comb_out,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_tiles // total_cores)

        in_dfb = ttl.make_dataflow_buffer_like(input_mixes, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
        base_dfb = ttl.make_dataflow_buffer_like(base_tile, shape=(1, 1), block_count=1)
        prem_dfb = ttl.make_dataflow_buffer_like(pre_mask, shape=(1, 1), block_count=1)
        preeps_dfb = ttl.make_dataflow_buffer_like(pre_eps_tile, shape=(1, 1), block_count=1)
        postmm_dfb = ttl.make_dataflow_buffer_like(post_mult_mask, shape=(1, 1), block_count=1)
        combm_dfb = ttl.make_dataflow_buffer_like(comb_mask, shape=(1, 1), block_count=1)

        sig_dfb = ttl.make_dataflow_buffer_like(input_mixes, shape=(1, 1), block_count=2)

        pre_out_dfb = ttl.make_dataflow_buffer_like(pre_out, shape=(1, 1), block_count=2)
        post_out_dfb = ttl.make_dataflow_buffer_like(post_out, shape=(1, 1), block_count=2)
        comb_out_dfb = ttl.make_dataflow_buffer_like(comb_out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()
            base = base_dfb.wait()
            prem = prem_dfb.wait()
            preeps = preeps_dfb.wait()
            postmm = postmm_dfb.wait()
            combm = combm_dfb.wait()

            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    inp = in_dfb.wait()
                    # scaled = inp * sc + base, computed twice (once via sigmoid
                    # for pre/post, once plain for comb). Fused elementwise chain.
                    sig_dfb.reserve().store(ttl.math.sigmoid(inp * sc + base))
                    comb_out_dfb.reserve().store((inp * sc + base) * combm)

                    sig = sig_dfb.wait()
                    pre_out_dfb.reserve().store(sig * prem + preeps)
                    post_out_dfb.reserve().store(sig * postmm)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scale_tile[0, 0], sc_dfb.reserve()).wait()
            ttl.copy(base_tile[0, 0], base_dfb.reserve()).wait()
            ttl.copy(pre_mask[0, 0], prem_dfb.reserve()).wait()
            ttl.copy(pre_eps_tile[0, 0], preeps_dfb.reserve()).wait()
            ttl.copy(post_mult_mask[0, 0], postmm_dfb.reserve()).wait()
            ttl.copy(comb_mask[0, 0], combm_dfb.reserve()).wait()
            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    ttl.copy(input_mixes[global_i, 0], in_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    ttl.copy(pre_out_dfb.wait(), pre_out[global_i, 0]).wait()
                    ttl.copy(post_out_dfb.wait(), post_out[global_i, 0]).wait()
                    ttl.copy(comb_out_dfb.wait(), comb_out[global_i, 0]).wait()

    return split_mixes_kernel


def _scale_broadcast(mhc_scale, mhc_mult, dtype):
    """Expand mhc_scale [3] to mhc_mult3 entries per section layout."""
    import torch
    return torch.cat([
        mhc_scale[0].to(dtype).expand(mhc_mult),
        mhc_scale[1].to(dtype).expand(mhc_mult),
        mhc_scale[2].to(dtype).expand(mhc_mult * mhc_mult),
    ])


def _pad_row_to_tile(vec, valid_len, dtype):
    """Place `vec[:valid_len]` in a length-32 row, zeros in the rest."""
    import torch
    assert vec.numel() >= valid_len
    out = torch.zeros(TILE, dtype=dtype)
    out[:valid_len] = vec[:valid_len].to(dtype)
    return out


def _broadcast_row_to_tile(row, dtype):
    """Tile a [32] row to [32, 32] by repeating it across all 32 rows."""
    import torch
    return row.to(dtype).unsqueeze(0).expand(TILE, TILE).contiguous()


def make_constant_tiles(mhc_scale, mhc_base, mhc_mult, mhc_post_mult_value, mhc_pre_eps, dtype):
    """Build the seven [32, 32] constant tiles the kernel consumes.

    Returns (scale_tile, base_tile, pre_mask, pre_eps_tile, post_mult_mask, comb_mask).
    All rows of each tile are identical; a row has `mhc_mult3` valid cols with
    zeros in the rest.
    """
    import torch
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult

    scale_vec = _scale_broadcast(mhc_scale, mhc_mult, dtype)
    scale_row = _pad_row_to_tile(scale_vec, mhc_mult3, dtype)

    base_row = _pad_row_to_tile(mhc_base, mhc_mult3, dtype)

    pre_mask_row = torch.zeros(TILE, dtype=dtype)
    pre_mask_row[:mhc_mult] = 1.0

    pre_eps_row = torch.zeros(TILE, dtype=dtype)
    pre_eps_row[:mhc_mult] = mhc_pre_eps

    post_mult_mask_row = torch.zeros(TILE, dtype=dtype)
    post_mult_mask_row[mhc_mult : 2 * mhc_mult] = mhc_post_mult_value

    comb_mask_row = torch.zeros(TILE, dtype=dtype)
    comb_mask_row[2 * mhc_mult : mhc_mult3] = 1.0

    return (
        _broadcast_row_to_tile(scale_row, dtype),
        _broadcast_row_to_tile(base_row, dtype),
        _broadcast_row_to_tile(pre_mask_row, dtype),
        _broadcast_row_to_tile(pre_eps_row, dtype),
        _broadcast_row_to_tile(post_mult_mask_row, dtype),
        _broadcast_row_to_tile(comb_mask_row, dtype),
    )


def pack_tokens(input_mixes, mhc_mult3, dtype):
    """Pack [num_tokens, mhc_mult3] -> [num_tokens, 32] with zero padding.

    num_tokens must be a multiple of TILE; each tile thus holds TILE tokens as
    its rows. mhc_mult3 <= 32 required (24 for mhc_mult=4).
    """
    import torch
    num_tokens, m3 = input_mixes.shape
    assert m3 == mhc_mult3, f"expected last dim {mhc_mult3}, got {m3}"
    assert num_tokens % TILE == 0, f"num_tokens={num_tokens} must be a multiple of {TILE}"
    out = torch.zeros(num_tokens, TILE, dtype=dtype)
    out[:, :m3] = input_mixes.to(dtype)
    return out.contiguous()


def unpack_section(packed, num_tokens, start, length):
    """Inverse of pack_tokens for a specific column range."""
    return packed[:num_tokens, start : start + length].contiguous()


def solve(
    input_mixes_tt, scale_tile_tt, base_tile_tt,
    pre_mask_tt, pre_eps_tile_tt, post_mult_mask_tt, comb_mask_tt,
    pre_out_tt, post_out_tt, comb_out_tt,
):
    """Run split_mixes. input_mixes / outputs are [num_tokens, 32] tensors;
    constants are [32, 32]. Tiles a packed layout produced by `pack_tokens`.
    """
    num_tiles = input_mixes_tt.shape[0] // TILE
    kernel = make_kernel(num_tiles=num_tiles)
    kernel(
        input_mixes_tt, scale_tile_tt, base_tile_tt,
        pre_mask_tt, pre_eps_tile_tt, post_mult_mask_tt, comb_mask_tt,
        pre_out_tt, post_out_tt, comb_out_tt,
    )


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _test_shape(device, ttnn, torch, n0, n1, mhc_mult, post_mult_value, pre_eps, threshold):
    from harness import assert_pcc
    from torch_refs import mhc_pre_split_mixes_ref

    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
    num_tokens = n0 * n1
    print(f"\n[shape] n0={n0} n1={n1} mhc_mult={mhc_mult} num_tokens={num_tokens}")

    torch.manual_seed(0)
    input_mixes = torch.randn((n0, n1, mhc_mult3), dtype=torch.float32)
    mhc_scale = torch.randn((3,), dtype=torch.float32)
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float32)

    pre_ref, post_ref, comb_ref = mhc_pre_split_mixes_ref(
        input_mixes, mhc_scale, mhc_base, mhc_mult, post_mult_value, pre_eps,
    )

    input_flat = input_mixes.reshape(num_tokens, mhc_mult3)
    input_packed = pack_tokens(input_flat, mhc_mult3, dtype=torch.float32)
    pre_packed = torch.zeros_like(input_packed)
    post_packed = torch.zeros_like(input_packed)
    comb_packed = torch.zeros_like(input_packed)

    (scale_tile, base_tile, pre_mask, pre_eps_tile,
     post_mult_mask, comb_mask) = make_constant_tiles(
        mhc_scale, mhc_base, mhc_mult, post_mult_value, pre_eps, dtype=torch.float32,
    )

    common = dict(
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tt = ttnn.from_torch(input_packed, **common)
    pre_tt = ttnn.from_torch(pre_packed, **common)
    post_tt = ttnn.from_torch(post_packed, **common)
    comb_tt = ttnn.from_torch(comb_packed, **common)
    scale_tt = ttnn.from_torch(scale_tile, **common)
    base_tt = ttnn.from_torch(base_tile, **common)
    pre_mask_tt = ttnn.from_torch(pre_mask, **common)
    pre_eps_tt = ttnn.from_torch(pre_eps_tile, **common)
    post_mask_tt = ttnn.from_torch(post_mult_mask, **common)
    comb_mask_tt = ttnn.from_torch(comb_mask, **common)

    solve(
        input_tt, scale_tt, base_tt,
        pre_mask_tt, pre_eps_tt, post_mask_tt, comb_mask_tt,
        pre_tt, post_tt, comb_tt,
    )

    pre_packed = ttnn.to_torch(pre_tt)
    post_packed = ttnn.to_torch(post_tt)
    comb_packed = ttnn.to_torch(comb_tt)

    pre_tt_out = unpack_section(pre_packed, num_tokens, 0, mhc_mult).reshape(n0, n1, mhc_mult, 1)
    post_tt_out = unpack_section(post_packed, num_tokens, mhc_mult, mhc_mult).reshape(n0, n1, mhc_mult, 1)
    comb_tt_out = unpack_section(comb_packed, num_tokens, 2 * mhc_mult, mhc_mult * mhc_mult).reshape(
        n0, n1, mhc_mult, mhc_mult,
    )

    if num_tokens <= 4:
        print(f"  pre_ref[0,0]:\n{pre_ref[0, 0].squeeze(-1).numpy()}")
        print(f"  pre_tt[0,0]:\n{pre_tt_out[0, 0].squeeze(-1).numpy()}")

    print("  pre:")
    assert_pcc(pre_ref, pre_tt_out, threshold=threshold)
    print("  post:")
    assert_pcc(post_ref, post_tt_out, threshold=threshold)
    print("  comb:")
    assert_pcc(comb_ref, comb_tt_out, threshold=threshold)


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    POST_MULT = 2.0
    PRE_EPS = 1e-2
    THRESHOLD = 0.9995

    SHAPES = [
        (1, 1024, 4),
        (2, 1024, 4),
        (1, 4096, 4),
        (2, 4096, 4),
    ]

    device = ttnn.open_device(device_id=0)
    try:
        for n0, n1, mhc_mult in SHAPES:
            _test_shape(device, ttnn, torch, n0, n1, mhc_mult, POST_MULT, PRE_EPS, THRESHOLD)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)
