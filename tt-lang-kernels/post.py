"""TT-Lang mhc_post (kernel #3 in ../kernels.md).

Reference: TileKernels/tile_kernels/mhc/post_kernel.py (tilelang).
Reference test: TileKernels/tests/mhc/test_post.py.
CPU torch reference: `torch_refs.mhc_post_ref`.

Per-token computation:
    out[n, h] = x[h] * post_mix[n] + sum_m(comb[m, n] * residual[m, h])

where
    x: [num_tokens, h]          (fp32 here; bf16 in the model)
    residual: [num_tokens, mhc, h]
    post_mix: [num_tokens, mhc]
    comb: [num_tokens, mhc, mhc]  (typically sinkhorn output)
Output shape: [num_tokens, mhc, h].

For each token, the per-h-tile compute is:
    x_bc_tile * post_mix_bc_tile + (comb^T_tile @ residual_tile)

with x pre-packed so rows 0..mhc-1 each hold a copy of x[:], post_mix pre-packed
with scalars in col 0 rows 0..mhc-1 then broadcast across cols, comb stored
pre-transposed so the 32x32 matmul produces the right row layout, and residual
packed with rows 0..mhc-1 holding the mhc vectors. Tokens with valid rows 0..3
and padded zero rows 4..31 propagate as zero through the matmul.
"""
from __future__ import annotations

import ttl

TILE = 32


def make_kernel(num_tokens: int, h_tiles: int):

    @ttl.operation(grid="auto")
    def post_kernel(x, residual, comb_T, post_mix, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tokens_per_core = -(-num_tokens // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), block_count=2)
        comb_dfb = ttl.make_dataflow_buffer_like(comb_T, shape=(1, 1), block_count=2)
        post_dfb = ttl.make_dataflow_buffer_like(post_mix, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        post_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        post_term_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        matmul_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    post = post_dfb.wait()
                    comb_t = comb_dfb.wait()

                    # Broadcast post_mix's col 0 across all cols. Rows 0..mhc-1
                    # now carry the per-row scalar; rows >=mhc stay zero.
                    pbc = post_bc_dfb.reserve()
                    pbc.store(ttl.math.broadcast(post, pbc, dims=[1]))
                    post_bc = post_bc_dfb.wait()

                    for _ in range(h_tiles):
                        x_tile = x_dfb.wait()
                        res_tile = res_dfb.wait()
                        post_term_dfb.reserve().store(x_tile * post_bc)
                        matmul_dfb.reserve().store(comb_t @ res_tile)
                        out_dfb.reserve().store(
                            post_term_dfb.wait() + matmul_dfb.wait()
                        )

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    ttl.copy(post_mix[global_t, 0], post_dfb.reserve()).wait()
                    ttl.copy(comb_T[global_t, 0], comb_dfb.reserve()).wait()
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()
                        ttl.copy(residual[global_t, h], res_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    for h in range(h_tiles):
                        ttl.copy(out_dfb.wait(), out[global_t, h]).wait()

    return post_kernel


def pack_residual_or_x_bc(vec_per_token, mhc, dtype):
    """Pack [num_tokens, mhc, h] (residual) OR broadcast-replicate [num_tokens, h]
    (x) into [num_tokens * 32, h] so rows 0..mhc-1 of each token-block carry the
    mhc data rows (x is replicated into all mhc rows so downstream broadcast is
    implicit). Pad rows mhc..31 with zero.
    """
    import torch
    if vec_per_token.dim() == 3:
        num_tokens, m, h = vec_per_token.shape
        assert m == mhc
        src = vec_per_token
    elif vec_per_token.dim() == 2:
        num_tokens, h = vec_per_token.shape
        src = vec_per_token.unsqueeze(1).expand(-1, mhc, -1)
    else:
        raise ValueError(f"unexpected dim {vec_per_token.dim()}")
    assert h % TILE == 0
    out = torch.zeros(num_tokens * TILE, h, dtype=dtype)
    out.view(num_tokens, TILE, h)[:, :mhc, :] = src.to(dtype)
    return out.contiguous()


def pack_comb_T(comb, dtype):
    """[num_tokens, mhc, mhc] -> [num_tokens * 32, 32] with comb^T in the top-left
    mhc x mhc region of each tile (rest zero). Storing transposed lets a 32x32
    matmul of comb_T @ residual_tile produce the right output row layout.
    """
    import torch
    num_tokens, m, n = comb.shape
    assert m == n
    assert m <= TILE
    out = torch.zeros(num_tokens * TILE, TILE, dtype=dtype)
    out.view(num_tokens, TILE, TILE)[:, :m, :m] = comb.transpose(-1, -2).to(dtype)
    return out.contiguous()


def pack_post_mix(post_mix, dtype):
    """[num_tokens, mhc, 1] -> [num_tokens * 32, 32] with mhc scalars in col 0
    rows 0..mhc-1. Same layout as apply_mix's mix tensor.
    """
    import torch
    num_tokens, mhc, one = post_mix.shape
    assert one == 1
    out = torch.zeros(num_tokens * TILE, TILE, dtype=dtype)
    out.view(num_tokens, TILE, TILE)[:, :mhc, 0] = post_mix.to(dtype).squeeze(-1)
    return out.contiguous()


def unpack_out(packed, num_tokens, mhc, h):
    """[num_tokens * 32, h] -> [num_tokens, mhc, h] — rows 0..mhc-1 of each block."""
    return packed.view(num_tokens, TILE, h)[:, :mhc, :].contiguous()


def solve(x_tt, residual_tt, comb_T_tt, post_mix_tt, out_tt):
    num_tokens = x_tt.shape[0] // TILE
    h_tiles = x_tt.shape[1] // TILE
    kernel = make_kernel(num_tokens=num_tokens, h_tiles=h_tiles)
    kernel(x_tt, residual_tt, comb_T_tt, post_mix_tt, out_tt)


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _test_shape(device, ttnn, torch, n0, n1, mhc, h, threshold):
    from harness import assert_pcc
    from torch_refs import mhc_post_ref

    num_tokens = n0 * n1
    print(f"\n[shape] n0={n0} n1={n1} mhc={mhc} h={h} num_tokens={num_tokens}")

    torch.manual_seed(0)
    x_ref = torch.randn((n0, n1, h), dtype=torch.bfloat16)
    residual_ref = torch.randn((n0, n1, mhc, h), dtype=torch.bfloat16)
    post_mix_ref = torch.randn((n0, n1, mhc, 1), dtype=torch.float32)
    comb_ref = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32).softmax(-1)

    y_ref = mhc_post_ref(x_ref, residual_ref, post_mix_ref, comb_ref)

    x_flat = x_ref.reshape(num_tokens, h).float()
    res_flat = residual_ref.reshape(num_tokens, mhc, h).float()
    post_flat = post_mix_ref.reshape(num_tokens, mhc, 1)
    comb_flat = comb_ref.reshape(num_tokens, mhc, mhc)

    x_packed = pack_residual_or_x_bc(x_flat, mhc, dtype=torch.float32)
    res_packed = pack_residual_or_x_bc(res_flat, mhc, dtype=torch.float32)
    comb_packed = pack_comb_T(comb_flat, dtype=torch.float32)
    post_packed = pack_post_mix(post_flat, dtype=torch.float32)
    out_packed = torch.zeros_like(x_packed)

    common = dict(
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.from_torch(x_packed, **common)
    res_tt = ttnn.from_torch(res_packed, **common)
    comb_tt = ttnn.from_torch(comb_packed, **common)
    post_tt = ttnn.from_torch(post_packed, **common)
    out_tt = ttnn.from_torch(out_packed, **common)

    solve(x_tt, res_tt, comb_tt, post_tt, out_tt)

    out_packed = ttnn.to_torch(out_tt)
    y_tt = unpack_out(out_packed, num_tokens, mhc, h).reshape(n0, n1, mhc, h).bfloat16()

    assert_pcc(y_ref.float(), y_tt.float(), threshold=threshold)


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    THRESHOLD = 0.999

    SHAPES = [
        (1, 32, 4, 32),
        (1, 1024, 4, 1280),
        (1, 1024, 4, 2560),
        (1, 1024, 4, 4096),
        (2, 1024, 4, 4096),
    ]

    device = ttnn.open_device(device_id=0)
    try:
        for n0, n1, mhc, h in SHAPES:
            _test_shape(device, ttnn, torch, n0, n1, mhc, h, THRESHOLD)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)
