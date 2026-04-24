"""TT-Lang Sinkhorn normalization (kernel #1 in ../kernels.md).

Reference: TileKernels/tile_kernels/mhc/sinkhorn_kernel.py (tilelang).
Reference test: TileKernels/tests/mhc/test_sinkhorn.py.
CPU torch reference: `torch_refs.sinkhorn_normalize_ref`.

Input shape: [n0, n1, mhc, mhc], mhc=4 for V4-Flash. We flatten to
[num_slices, 4, 4] where num_slices = n0*n1, then embed each 4x4 slice in the
top-left of its own 32x32 tile. Padded cells are PAD_SENTINEL so exp(pad -
row_max) underflows to 0 during softmax; immediately after the softmax, a
mask-multiply zeros the padded region exactly and an eps-mask-add introduces
eps only inside the valid 4x4 region. All subsequent iterations preserve the
"0 outside the valid 4x4" invariant.

Implementation strategy: one slice = one 32x32 tile. All sinkhorn state lives
in a single reserved output block (`xs`), modified in-place via repeated
`xs.store(...)`. This is the same pattern layernorm_minimal uses for its
`mean_blk` accumulator — it sidesteps the reduce-inside-scf.for legalization
trap because there is no DFB reserve/wait pair inside the iteration; the
block handle is captured once outside the loop and rewritten each pass.

The reduce/recip pattern strictly follows prompt.md (tt-lang-import):
  - reduce's output must be stored immediately into its own DFB;
  - broadcast's second arg must be the reserved output block it stores into.
"""
from __future__ import annotations

import ttl

TILE = 32


def make_sinkhorn_kernel(num_slices: int, repeat: int, eps: float):
    """Build a sinkhorn kernel for a fixed (num_slices, repeat, eps). All
    three are compile-time constants so the loops unroll / scf.for lower
    cleanly and `ttl.math.fill(..., eps)` is well-formed.
    """

    @ttl.operation(grid=(1, 1), options="--no-ttl-reduce-full-fp32")
    def sinkhorn_kernel(x, mask, eps_mask, scaler, out):
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), block_count=1)
        em_dfb = ttl.make_dataflow_buffer_like(eps_mask, shape=(1, 1), block_count=1)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        # Reused scratch DFBs for reduce results and their reciprocals.
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            sc = sc_dfb.wait()
            m = m_dfb.wait()
            em = em_dfb.wait()
            for _ in range(num_slices):
                x_in = x_dfb.wait()
                # out_dfb acts as the sinkhorn state buffer: `xs` starts as
                # exp(x - row_max) and ends as the final sinkhorn output.
                with out_dfb.reserve() as xs:
                    # --- Row softmax ---
                    red_dfb.reserve().store(ttl.math.reduce_max(x_in, sc, dims=[1]))
                    xs.store(ttl.math.exp(x_in - ttl.math.broadcast(red_dfb.wait(), x_in, dims=[1])))
                    red_dfb.reserve().store(ttl.math.reduce_sum(xs, sc, dims=[1]))
                    inv_dfb.reserve().store(ttl.math.recip(red_dfb.wait()))
                    # Fold softmax multiply + mask + eps into one rewrite of xs.
                    xs.store((xs * ttl.math.broadcast(inv_dfb.wait(), x_in, dims=[1])) * m + em)

                    # --- First col-normalize (paired with the initial softmax) ---
                    red_dfb.reserve().store(ttl.math.reduce_sum(xs, sc, dims=[0]))
                    cs0 = red_dfb.wait()
                    inv_dfb.reserve().store(ttl.math.recip(cs0 + ttl.math.fill(cs0, eps)))
                    xs.store(xs * ttl.math.broadcast(inv_dfb.wait(), x_in, dims=[0]))

                    # --- `repeat - 1` alternating (row, col) normalizations ---
                    # Compiles to scf.for, but xs is captured from the enclosing
                    # with-block so L1 state persists across iterations.
                    for _ in range(repeat - 1):
                        red_dfb.reserve().store(ttl.math.reduce_sum(xs, sc, dims=[1]))
                        rs = red_dfb.wait()
                        inv_dfb.reserve().store(ttl.math.recip(rs + ttl.math.fill(rs, eps)))
                        xs.store(xs * ttl.math.broadcast(inv_dfb.wait(), x_in, dims=[1]))

                        red_dfb.reserve().store(ttl.math.reduce_sum(xs, sc, dims=[0]))
                        cs = red_dfb.wait()
                        inv_dfb.reserve().store(ttl.math.recip(cs + ttl.math.fill(cs, eps)))
                        xs.store(xs * ttl.math.broadcast(inv_dfb.wait(), x_in, dims=[0]))
                    # with exit -> pushes xs to out_dfb.

        @ttl.datamovement()
        def dm_read():
            ttl.copy(mask[0, 0], m_dfb.reserve()).wait()
            ttl.copy(eps_mask[0, 0], em_dfb.reserve()).wait()
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for tile_idx in range(num_slices):
                ttl.copy(x[tile_idx, 0], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            for tile_idx in range(num_slices):
                ttl.copy(out_dfb.wait(), out[tile_idx, 0]).wait()

    return sinkhorn_kernel


def solve(x_tt, mask_tt, eps_mask_tt, scaler_tt, out_tt, *, repeat: int, eps: float):
    """Run the sinkhorn kernel. `x_tt`/`out_tt` are [num_slices*32, 32] fp32
    ttnn tensors (one 32x32 tile per 4x4 slice); `mask_tt`/`eps_mask_tt`/
    `scaler_tt` are single 32x32 fp32 tiles.
    """
    num_slices = x_tt.shape[0] // TILE
    kernel = make_sinkhorn_kernel(num_slices=num_slices, repeat=repeat, eps=eps)
    kernel(x_tt, mask_tt, eps_mask_tt, scaler_tt, out_tt)


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _test_shape(device, ttnn, torch, n0, n1, mhc, repeat, eps, threshold):
    """One shape: build inputs, run kernel, compare against torch ref."""
    from harness import (
        PAD_SENTINEL,
        assert_pcc,
        eps_mask_tile,
        mask_tile,
        pack_4x4_slices,
        scaler_tile,
        unpack_4x4_slices,
    )
    from torch_refs import sinkhorn_normalize_ref

    if mhc != 4:
        raise NotImplementedError(f"only mhc=4 is supported; got {mhc}")

    num_slices = n0 * n1
    print(f"\n[shape] n0={n0} n1={n1} mhc={mhc} (num_slices={num_slices})")

    x_ref = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32)
    y_ref = sinkhorn_normalize_ref(x_ref, repeat=repeat, eps=eps)

    x_flat = x_ref.reshape(num_slices, mhc, mhc)
    x_packed = pack_4x4_slices(x_flat, pad_value=PAD_SENTINEL)
    out_packed = torch.zeros_like(x_packed)

    mask = mask_tile(valid=mhc)
    em = eps_mask_tile(eps=eps, valid=mhc)
    sc = scaler_tile()

    common = dict(
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.from_torch(x_packed, **common)
    out_tt = ttnn.from_torch(out_packed, **common)
    mask_tt = ttnn.from_torch(mask, **common)
    em_tt = ttnn.from_torch(em, **common)
    sc_tt = ttnn.from_torch(sc, **common)

    solve(x_tt, mask_tt, em_tt, sc_tt, out_tt, repeat=repeat, eps=eps)

    out_packed = ttnn.to_torch(out_tt)
    y_tt = unpack_4x4_slices(out_packed, num_slices).reshape(n0, n1, mhc, mhc)

    # Debug dump on the smoke case so we can diagnose without re-running.
    if num_slices == 1:
        print(f"  x_in[0]:\n{x_ref[0, 0].numpy()}")
        print(f"  y_ref[0]:\n{y_ref[0, 0].numpy()}")
        print(f"  y_tt[0]:\n{y_tt[0, 0].numpy()}")
        print(f"  raw out tile top-left 8x8:\n{out_packed[:8, :8].numpy()}")

    assert_pcc(y_ref, y_tt, threshold=threshold)


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    import os

    # Ramp up iteration count as we gain confidence; can override via env for
    # local debugging without editing the file.
    REPEAT = int(os.environ.get("SINKHORN_REPEAT", "1"))
    EPS = 1e-6
    THRESHOLD = 0.9995

    # Smoke first: single slice is the easiest to diagnose. Once the smoke
    # shape passes we'll extend to V4-Flash shapes from kernels.md.
    SHAPES = [
        (1, 1, 4),     # smoke: 1 slice
    ]
    if os.environ.get("SINKHORN_ALL", "") == "1":
        SHAPES = [
            (1, 1, 4),
            (2, 1, 4),
            (1, 1024, 4),
            (2, 1024, 4),
            (1, 4096, 4),
            (2, 4096, 4),
        ]

    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        for n0, n1, mhc in SHAPES:
            _test_shape(device, ttnn, torch, n0, n1, mhc, REPEAT, EPS, THRESHOLD)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)
