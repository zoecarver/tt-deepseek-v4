"""TT-Lang Sinkhorn normalization (kernel #1 in ../kernels.md).

Reference: TileKernels/tile_kernels/mhc/sinkhorn_kernel.py (tilelang).
Reference test: TileKernels/tests/mhc/test_sinkhorn.py.
CPU torch reference: `torch_refs.sinkhorn_normalize_ref`.

Input shape: [n0, n1, mhc, mhc], mhc=4 for V4-Flash. We flatten to
[num_slices, 4, 4] where num_slices = n0*n1, then embed each 4x4 slice in the
top-left of its own 32x32 tile. Padded cells are PAD_SENTINEL so exp(pad -
row_max) underflows to 0 during softmax. After the softmax, a mask-multiply
zeros the padded region exactly and an eps-mask-add introduces eps only
inside the valid 4x4 region; subsequent normalize passes preserve the "0
outside the valid 4x4" invariant.

Structure:
  - Each major transform (softmax, mask+eps, col-normalize, iteration pair)
    produces a fresh block into `state_dfb`. Intermediates (row-max, row-sum,
    reciprocal-broadcast) each have their own DFB.
  - `state_dfb` is consumed twice per normalize step (once for the reduction,
    once for the multiply), so we reserve a copy into `state_copy_dfb` in the
    same `with state_dfb.wait()` scope (FA-style, see test_fa_simple.py).
  - Compiler bug workaround: `options="--no-ttl-reduce-full-fp32"` (see
    README "reduce dims=[1] returns zeros on fp32 tiles").
"""
from __future__ import annotations

import ttl

TILE = 32


def make_sinkhorn_kernel(num_slices: int, repeat: int, eps: float):

    @ttl.operation(grid=(1, 1), options="--no-ttl-reduce-full-fp32")
    def sinkhorn_kernel(x, mask, eps_mask, scaler, out):
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), block_count=1)
        em_dfb = ttl.make_dataflow_buffer_like(eps_mask, shape=(1, 1), block_count=1)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        # Intermediates shaped like a scalar tile (reductions land in row 0 or
        # col 0 of a 32x32 tile).
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        # Broadcasts and state carry full-tile shape (like x).
        bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        state_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        state_copy_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        exp_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            sc = sc_dfb.wait()
            m = m_dfb.wait()
            em = em_dfb.wait()

            for _ in range(num_slices):
                # ----- Row softmax: state := softmax(x, dim=-1) -----
                x_in = x_dfb.wait()
                red_dfb.reserve().store(ttl.math.reduce_max(x_in, sc, dims=[1]))
                rmx = bc_dfb.reserve()
                rmx.store(ttl.math.broadcast(red_dfb.wait(), rmx, dims=[1]))
                exp_dfb.reserve().store(ttl.math.exp(x_in - bc_dfb.wait()))

                ex = exp_dfb.wait()
                red_dfb.reserve().store(ttl.math.reduce_sum(ex, sc, dims=[1]))
                state_copy_dfb.reserve().store(ex)
                rinv = bc_dfb.reserve()
                rinv.store(ttl.math.broadcast(ttl.math.recip(red_dfb.wait()), rinv, dims=[1]))
                state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                # ----- Mask + eps: state := state * mask + eps_mask -----
                state_copy_dfb.reserve().store(state_dfb.wait() * m + em)
                state_dfb.reserve().store(state_copy_dfb.wait())

                # ----- First col-normalize: state := state / (col_sum + eps) -----
                s = state_dfb.wait()
                red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[0]))
                state_copy_dfb.reserve().store(s)
                cinv = bc_dfb.reserve()
                csum = red_dfb.wait()
                cinv.store(ttl.math.broadcast(
                    ttl.math.recip(csum + ttl.math.fill(csum, eps)),
                    cinv, dims=[0]))
                state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                # ----- repeat - 1 alternating (row, col) normalizations -----
                for _ in range(repeat - 1):
                    # row normalize
                    s = state_dfb.wait()
                    red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[1]))
                    state_copy_dfb.reserve().store(s)
                    rinv = bc_dfb.reserve()
                    rsum = red_dfb.wait()
                    rinv.store(ttl.math.broadcast(
                        ttl.math.recip(rsum + ttl.math.fill(rsum, eps)),
                        rinv, dims=[1]))
                    state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                    # col normalize
                    s = state_dfb.wait()
                    red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[0]))
                    state_copy_dfb.reserve().store(s)
                    cinv = bc_dfb.reserve()
                    csum = red_dfb.wait()
                    cinv.store(ttl.math.broadcast(
                        ttl.math.recip(csum + ttl.math.fill(csum, eps)),
                        cinv, dims=[0]))
                    state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                # ----- Final copy state -> out -----
                out_dfb.reserve().store(state_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            ttl.copy(mask[0, 0], m_dfb.reserve()).wait()
            ttl.copy(eps_mask[0, 0], em_dfb.reserve()).wait()
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for t in range(num_slices):
                ttl.copy(x[t, 0], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            for t in range(num_slices):
                ttl.copy(out_dfb.wait(), out[t, 0]).wait()

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
    x_packed = pack_4x4_slices(x_flat, pad_value=PAD_SENTINEL, dtype=torch.float32)
    out_packed = torch.zeros_like(x_packed)

    mask = mask_tile(valid=mhc, dtype=torch.float32)
    em = eps_mask_tile(eps=eps, valid=mhc, dtype=torch.float32)
    sc = scaler_tile(dtype=torch.float32)

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

    if num_slices == 1:
        print(f"  x_in[0]:\n{x_ref[0, 0].numpy()}")
        print(f"  y_ref[0]:\n{y_ref[0, 0].numpy()}")
        print(f"  y_tt[0]:\n{y_tt[0, 0].numpy()}")

    assert_pcc(y_ref, y_tt, threshold=threshold)


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    REPEAT = 10
    EPS = 1e-6
    THRESHOLD = 0.9995

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
