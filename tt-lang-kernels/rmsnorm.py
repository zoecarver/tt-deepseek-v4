"""TT-Lang fused RMSNorm (kernel #4 in ../kernels.md, simplified).

bf16 in / bf16 out. Matches the tt-metal deepseek_v3 activation convention
(activations stay bf16 throughout the forward pass; any downstream bfp8/bfp4
matmul consumes bf16 and converts at the op boundary).

Computes per token row t:
  sum_sq[t]  = sum_k x[t, k]^2
  inv_rms[t] = rsqrt(sum_sq[t] / D + eps)
  y[t, k]    = x[t, k] * gamma[k] * inv_rms[t]

Packing (host-side):
  x     -> [num_tokens, hidden] bf16, num_tokens and hidden multiples of 32.
  gamma -> [TILE, hidden] bf16 with gamma replicated across all 32 rows so
           every token-row in a row-tile sees the same gamma.
  out   -> [num_tokens, hidden] bf16.

Streaming structure: one row-tile (32 tokens) per core-iteration. Two passes
over x per row-tile -- first pass accumulates sum(x^2), second pass applies
gamma*inv_rms. dm_read re-reads x for the second pass; accepting the extra
DRAM traffic keeps the kernel simple (no intermediate staging DFB for x).

Not perf-tuned: (1,1) DFB block shape, per-tile ping-pong accumulator. Good
enough for on-device correctness; revisit for perf later.

bfp8 output follow-up: tt-lang rejects storing a bf16 compute expression
directly into a bfp8 CB ("tensor element type bf16 must match view element
type bfp_bf8"), and ttl.copy from a bf16 L1 block to a bfp8 DRAM tensor
silently produces NaN in the current toolchain. Leaving bf16-out for now;
downstream matmul can either consume bf16 directly or a separate
ttnn.typecast step can stamp it to bfp8 if a matmul requires that.
"""
from __future__ import annotations

import ttl

TILE = 32


def make_kernel(num_row_tiles: int, h_tiles: int, rms_eps: float, inv_D: float):

    @ttl.operation(
        grid="auto",
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def rmsnorm_kernel(x, gamma, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_row_tiles // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        g_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        # Per-row-tile accumulators live in the input dtype (bf16). DST
        # register accumulates in fp32 regardless via fp32_dest_acc_en so
        # reduce_sum drift stays bounded.
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        xsq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        red_step_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    # Pass 1: accumulate sum(x^2) per token (reduce dims=[1]).
                    x0 = x_dfb.wait()
                    xsq_dfb.reserve().store(x0 * x0)
                    sq_dfb.reserve().store(
                        ttl.math.reduce_sum(xsq_dfb.wait(), sc, dims=[1])
                    )
                    for _ in range(h_tiles - 1):
                        xk = x_dfb.wait()
                        xsq_dfb.reserve().store(xk * xk)
                        red_step_dfb.reserve().store(
                            ttl.math.reduce_sum(xsq_dfb.wait(), sc, dims=[1])
                        )
                        prev = sq_dfb.wait()
                        sq_dfb.reserve().store(prev + red_step_dfb.wait())

                    # Finalize: inv_rms = rsqrt(sum_sq / D + eps), broadcast
                    # across columns so the per-token scalar applies to all h.
                    sq = sq_dfb.wait()
                    inv_bc = inv_bc_dfb.reserve()
                    inv_bc.store(ttl.math.broadcast(
                        ttl.math.rsqrt(
                            sq * ttl.math.fill(sq, inv_D) + ttl.math.fill(sq, rms_eps)
                        ),
                        inv_bc, dims=[1],
                    ))
                    inv = inv_bc_dfb.wait()

                    # Pass 2: apply. Re-read x and stream gamma per h-tile.
                    for _ in range(h_tiles):
                        xk = x_dfb.wait()
                        gk = g_dfb.wait()
                        out_dfb.reserve().store(xk * gk * inv)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    # Pass 1: x only.
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()
                    # Pass 2: x again + gamma.
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()
                        ttl.copy(gamma[0, h], g_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for h in range(h_tiles):
                        ttl.copy(out_dfb.wait(), out[global_t, h]).wait()

    return rmsnorm_kernel


def pack_gamma(gamma, dtype):
    """[hidden] -> [TILE, hidden] with gamma replicated across all 32 rows.

    Every row of a [32, hidden] block gets the same gamma, so each of the
    32 tokens in a row-tile sees gamma when multiplied elementwise with its
    x row.
    """
    import torch
    (hidden,) = gamma.shape
    assert hidden % TILE == 0, f"hidden={hidden} must be multiple of {TILE}"
    return gamma.unsqueeze(0).expand(TILE, -1).to(dtype).contiguous()


def solve(x_tt, gamma_tt, sc_tt, out_tt, *, rms_eps: float, hidden: int):
    num_row_tiles = x_tt.shape[0] // TILE
    h_tiles = x_tt.shape[1] // TILE
    kernel = make_kernel(
        num_row_tiles=num_row_tiles,
        h_tiles=h_tiles,
        rms_eps=rms_eps,
        inv_D=1.0 / hidden,
    )
    kernel(x_tt, gamma_tt, sc_tt, out_tt)


def rmsnorm(x, gamma, eps: float = 1e-6):
    """Plug-and-play entry point.

    Inputs:
      x:     ttnn.Tensor [num_tokens, hidden] bfloat16, TILE_LAYOUT, DRAM.
      gamma: ttnn.Tensor [1, hidden], [hidden], or pre-replicated [TILE, hidden]
             bfloat16, TILE_LAYOUT, DRAM.
      eps:   RMSNorm epsilon.

    Returns:
      y: ttnn.Tensor [num_tokens, hidden] bfloat16, TILE_LAYOUT, DRAM.
    """
    import torch
    import ttnn

    assert x.dtype == ttnn.bfloat16
    assert gamma.dtype == ttnn.bfloat16
    device = x.device()

    num_tokens, hidden = x.shape
    assert num_tokens % TILE == 0
    assert hidden % TILE == 0

    g_shape = tuple(gamma.shape)
    if g_shape == (TILE, hidden):
        gamma_tt = gamma
    elif g_shape in ((1, hidden), (hidden,)):
        g_host = ttnn.to_torch(gamma).reshape(hidden)
        g_packed = pack_gamma(g_host, dtype=torch.bfloat16)
        gamma_tt = ttnn.from_torch(
            g_packed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        raise ValueError(f"unexpected gamma shape {g_shape}")

    from harness import scaler_tile  # noqa: F401 (script context)
    sc = scaler_tile(dtype=torch.bfloat16)
    sc_tt = ttnn.from_torch(
        sc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_host = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16)
    out_tt = ttnn.from_torch(
        out_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    solve(x, gamma_tt, sc_tt, out_tt, rms_eps=eps, hidden=hidden)
    return out_tt


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _rmsnorm_ref(x: "torch.Tensor", gamma: "torch.Tensor", eps: float) -> "torch.Tensor":
    """bf16 torch reference. Computes in fp32 to avoid accumulation drift."""
    import torch
    xf = x.float()
    inv_rms = xf.square().mean(-1, keepdim=True).add(eps).rsqrt()
    return (xf * gamma.float() * inv_rms).bfloat16()


def _test_shape(device, ttnn, torch, num_tokens, hidden, rms_eps, threshold):
    from harness import assert_pcc, scaler_tile

    print(f"\n[shape] num_tokens={num_tokens} hidden={hidden}")

    torch.manual_seed(0)
    x_ref = torch.randn((num_tokens, hidden), dtype=torch.bfloat16)
    gamma_ref = torch.randn((hidden,), dtype=torch.bfloat16)

    y_ref = _rmsnorm_ref(x_ref, gamma_ref, rms_eps)

    gamma_packed = pack_gamma(gamma_ref, dtype=torch.bfloat16)
    sc = scaler_tile(dtype=torch.bfloat16)

    x_tt = ttnn.from_torch(
        x_ref, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g_tt = ttnn.from_torch(
        gamma_packed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sc_tt = ttnn.from_torch(
        sc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_torch = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16)
    out_tt = ttnn.from_torch(
        out_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    solve(x_tt, g_tt, sc_tt, out_tt, rms_eps=rms_eps, hidden=hidden)

    y_tt = ttnn.to_torch(out_tt)  # bfp8 -> fp32 round-trip
    assert_pcc(y_ref.float(), y_tt.float(), threshold=threshold)


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    RMS_EPS = 1e-6

    # bfp8 is lossy relative to bf16 (per-row shared exponent, 8 mantissa bits).
    # PCC >0.999 is realistic; >0.9995 sometimes achievable on well-conditioned
    # inputs.
    SHAPES = [
        (32,   32,    0.999),
        (32,   128,   0.999),
        (128,  1280,  0.999),
        (1024, 2560,  0.999),
        (1024, 4096,  0.999),
        (4096, 4096,  0.999),
    ]

    device = ttnn.open_device(device_id=0)
    try:
        for num_tokens, hidden, threshold in SHAPES:
            _test_shape(device, ttnn, torch, num_tokens, hidden, RMS_EPS, threshold)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)
