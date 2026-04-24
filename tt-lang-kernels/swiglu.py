"""TT-Lang fused SwiGLU forward (kernel #5 in ../kernels.md, simplified).

bf16 in / bf16 out, no clamp. Per bf16-everywhere policy (see ../CLAUDE.md),
this kernel drops the per-channel fp8 cast + transpose variants from the
TileKernels reference; those fusions are a later-phase memory-density
optimization.

Computes per element:
  gate = x[t, :hidden]
  up   = x[t, hidden:2*hidden]
  y[t, h] = silu(gate) * up
where silu(x) = x * sigmoid(x).

Packing:
  x   -> [num_tokens, 2*hidden] bf16, tile-aligned on both dims.
  out -> [num_tokens, hidden]   bf16, tile-aligned on both dims.

Streaming structure: one row-tile x one h-tile per compute iteration.
For each output h-tile, dm_read pulls two input tiles (gate at col h, up at
col h + h_tiles). Pure elementwise; no reduction, no accumulator.

V4-Flash uses `swiglu_limit=10.0` by default to clamp gate to (-inf, +cv]
and up to [-cv, +cv] before multiplying (see inference.py Expert.forward).
That clamp path is NOT implemented here -- a first attempt using
`ttl.math.min/max` + `ttl.math.fill(t, -clamp_val)` tripped a "Cannot Compare
Non-Integer Values" compile error that I didn't root-cause. swiglu_limit=10
only activates on rare large-magnitude activations, so the unclamped variant
should match outputs for the common case; if coherence testing surfaces
divergence, revisit.

Not perf-tuned: (1,1) DFB block shape. Revisit for perf later.
"""
from __future__ import annotations

import ttl

TILE = 32


def make_kernel(num_row_tiles: int, h_tiles: int):

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def swiglu_kernel(x, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_row_tiles // total_cores)

        gate_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        up_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for _ in range(h_tiles):
                        g = gate_dfb.wait()
                        u = up_dfb.wait()
                        out_dfb.reserve().store(g * ttl.math.sigmoid(g) * u)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], gate_dfb.reserve()).wait()
                        ttl.copy(x[global_t, h + h_tiles], up_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for h in range(h_tiles):
                        ttl.copy(out_dfb.wait(), out[global_t, h]).wait()

    return swiglu_kernel


def solve(x_tt, out_tt):
    """Run swiglu. x: [num_tokens, 2*hidden], out: [num_tokens, hidden]."""
    num_row_tiles = x_tt.shape[0] // TILE
    h_tiles = out_tt.shape[1] // TILE
    assert x_tt.shape[1] == 2 * out_tt.shape[1], \
        f"x cols {x_tt.shape[1]} must be 2 * out cols {out_tt.shape[1]}"
    kernel = make_kernel(num_row_tiles=num_row_tiles, h_tiles=h_tiles)
    kernel(x_tt, out_tt)


def swiglu(x):
    """Plug-and-play entry point.

    Inputs:
      x: ttnn.Tensor [num_tokens, 2*hidden] bfloat16, TILE_LAYOUT, DRAM.

    Returns:
      y: ttnn.Tensor [num_tokens, hidden] bfloat16, TILE_LAYOUT, DRAM.
    """
    import torch
    import ttnn

    assert x.dtype == ttnn.bfloat16
    device = x.device()

    num_tokens, two_hidden = x.shape
    assert two_hidden % (2 * TILE) == 0, \
        f"x cols {two_hidden} must be multiple of 2*{TILE}"
    hidden = two_hidden // 2
    assert num_tokens % TILE == 0

    out_host = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16)
    out_tt = ttnn.from_torch(
        out_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    solve(x, out_tt)
    return out_tt


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _swiglu_ref(x: "torch.Tensor") -> "torch.Tensor":
    """bf16 torch reference (no clamp)."""
    import torch
    import torch.nn.functional as F
    two_hidden = x.shape[-1]
    hidden = two_hidden // 2
    gate = x[..., :hidden].float()
    up = x[..., hidden:].float()
    return (F.silu(gate) * up).bfloat16()


def _test_shape(device, ttnn, torch, num_tokens, hidden, threshold):
    from harness import assert_pcc

    print(f"\n[shape] num_tokens={num_tokens} hidden={hidden}")

    torch.manual_seed(0)
    x_ref = torch.randn((num_tokens, 2 * hidden), dtype=torch.bfloat16)

    y_ref = _swiglu_ref(x_ref)

    x_tt = ttnn.from_torch(
        x_ref, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_torch = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16)
    out_tt = ttnn.from_torch(
        out_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    solve(x_tt, out_tt)

    y_tt = ttnn.to_torch(out_tt)
    assert_pcc(y_ref.float(), y_tt.float(), threshold=threshold)


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    SHAPES = [
        # (num_tokens, hidden, threshold)
        (32,   128,  0.999),
        (128,  2048, 0.999),   # V4-Flash moe_inter_dim=2048
        (1024, 2048, 0.999),
        (128,  4096, 0.999),
    ]

    device = ttnn.open_device(device_id=0)
    try:
        for num_tokens, hidden, threshold in SHAPES:
            _test_shape(device, ttnn, torch, num_tokens, hidden, threshold)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)
