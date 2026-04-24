"""TT-Lang fused matmul + row-softmax for the MoE gate.

Computes `softmax((x @ w^T), dim=-1)` in a single kernel, with no DRAM
round-trip between the matmul and the softmax. Designed for the V4-Flash
MoE gate: `[M, K=4096] @ [K, N=256] -> [M, N]` followed by row-softmax.

Shape assumptions (v1):
  - M == TILE (one tile-row). This matches decode (M=1 padded to 32) and
    the brief's B=1,S=32 stress case (M=32 already). Multi-tile-row
    support is a future extension.
  - N is tile-aligned and small enough that bn == Nt fits in L1. For
    gate: N=256 -> Nt=8.
  - K is tile-aligned and divisible by bk.

Structure:
  1. Matmul accumulator over Kb = Kt/bk iterations (DST/fp32 accumulate),
     producing an L1 `mm_cb` of shape (Mt, Nt) = (1, Nt).
  2. Row-softmax on `mm_cb`:
        rmax = reduce_max(mm_cb, dims=[1])        # (1, 1)
        exp_cb = exp(mm_cb - broadcast(rmax))     # (1, Nt)
        rsum = reduce_sum(exp_cb, dims=[1])       # (1, 1)
        out_cb = exp_cb * broadcast(1/rsum)       # (1, Nt)
     Patterns follow `softmax_stage.py`.
  3. dm_write copies out_cb to DRAM.

bf16 everywhere (matches project bf16 policy). Reductions use a bf16
scaler and `--no-ttl-reduce-full-fp32` to avoid the fp32 reduce bug
(same workaround as rmsnorm.py / sinkhorn.py).
"""
from __future__ import annotations

import ttl

TILE = 32


def make_kernel(M: int, K: int, N: int, bk: int = 8):
    if M != TILE:
        raise ValueError(f"v1 requires M == TILE ({TILE}); got M={M}")
    if K % TILE or N % TILE:
        raise ValueError(f"K/N must be tile-aligned: K={K}, N={N}")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Kt % bk:
        raise ValueError(f"K in tiles ({Kt}) must be divisible by bk ({bk})")
    Kb = Kt // bk

    @ttl.operation(
        grid=(1, 1),
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def matmul_softmax_kernel(a, w, scaler, out):
        a_cb = ttl.make_dataflow_buffer_like(a, shape=(Mt, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, Nt), block_count=2)

        mm_cb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), block_count=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)

        rmax_cb = ttl.make_dataflow_buffer_like(scaler, shape=(Mt, 1), block_count=2)
        rmax_bc_cb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), block_count=2)

        exp_cb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), block_count=2)
        exp_copy_cb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), block_count=2)

        rsum_cb = ttl.make_dataflow_buffer_like(scaler, shape=(Mt, 1), block_count=2)
        rinv_bc_cb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), block_count=2)

        out_cb = ttl.make_dataflow_buffer_like(out, shape=(Mt, Nt), block_count=2)

        @ttl.compute()
        def compute():
            # Phase 1: matmul accumulate.
            with mm_cb.reserve() as p:
                for _ in range(Kb):
                    a_blk = a_cb.wait()
                    b_blk = b_cb.wait()
                    p += a_blk @ b_blk

            sc = sc_cb.wait()

            # Phase 2: row-softmax over mm_cb.
            with mm_cb.wait() as scores:
                with rmax_cb.reserve() as rmax:
                    rmax.store(ttl.math.reduce_max(scores, sc, dims=[1]))
                with rmax_cb.wait() as rmax, rmax_bc_cb.reserve() as rmx:
                    rmx.store(ttl.math.broadcast(rmax, rmx, dims=[1]))
                with rmax_bc_cb.wait() as rmx, exp_cb.reserve() as ex:
                    ex.store(ttl.math.exp(scores - rmx))

            with exp_cb.wait() as ex:
                with rsum_cb.reserve() as rsum:
                    rsum.store(ttl.math.reduce_sum(ex, sc, dims=[1]))
                with exp_copy_cb.reserve() as ec:
                    ec.store(ex)

            with rsum_cb.wait() as rsum, rinv_bc_cb.reserve() as rinv:
                rinv.store(ttl.math.broadcast(ttl.math.recip(rsum), rinv, dims=[1]))

            with exp_copy_cb.wait() as ex, rinv_bc_cb.wait() as rinv, out_cb.reserve() as o:
                o.store(ex * rinv)

        @ttl.datamovement()
        def dm_read():
            ttl.copy(scaler[0, 0], sc_cb.reserve()).wait()
            for kb in range(Kb):
                kc = kb * bk
                ttl.copy(a[0:Mt, kc : kc + bk], a_cb.reserve()).wait()
                ttl.copy(w[kc : kc + bk, 0:Nt], b_cb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            ttl.copy(out_cb.wait(), out[0:Mt, 0:Nt]).wait()

    return matmul_softmax_kernel


def solve(x_tt, w_tt, scaler_tt, out_tt, bk: int = 8):
    """Run out = softmax(x @ w, dim=-1).

    x: [M=TILE, K] bf16, TILE_LAYOUT.
    w: [K, N] bf16, TILE_LAYOUT. (Pre-transposed from the nn.Linear order.)
    scaler: (TILE, TILE) bf16 tile of 1.0s (for reductions).
    out: [M, N] bf16, TILE_LAYOUT.
    """
    M, K = x_tt.shape[0], x_tt.shape[1]
    K2, N = w_tt.shape[0], w_tt.shape[1]
    if K != K2:
        raise ValueError(f"K mismatch: x K={K}, w K={K2}")
    kernel = make_kernel(M=M, K=K, N=N, bk=bk)
    kernel(x_tt, w_tt, scaler_tt, out_tt)


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _test_shape(device, ttnn, torch, K: int, N: int, threshold: float):
    from harness import assert_pcc, scaler_tile

    print(f"\n[shape] K={K} N={N} (Kt={K // TILE}, Nt={N // TILE})")

    torch.manual_seed(0)
    # Model-shape random inputs. Gate sees bf16 activations, bf16 weights.
    # Use small-ish scale so softmax isn't near-saturated.
    x_row = torch.randn(1, K, dtype=torch.bfloat16) * 0.5
    w_nk = (torch.randn(N, K, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)
    y_ref_raw = x_row.float() @ w_nk.float().T
    y_ref = y_ref_raw.softmax(dim=-1).to(torch.bfloat16)

    # x padded to [TILE, K] (row 0 valid; rest zero).
    x_packed = torch.zeros(TILE, K, dtype=torch.bfloat16)
    x_packed[0] = x_row

    # w stored as [K, N] = w_nk.T for nn.Linear -> matmul-order.
    w_kn = w_nk.T.contiguous()

    sc = scaler_tile(dtype=torch.bfloat16)

    common = dict(
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.from_torch(x_packed, **common)
    w_tt = ttnn.from_torch(w_kn, **common)
    sc_tt = ttnn.from_torch(sc, **common)
    out_tt = ttnn.from_torch(torch.zeros(TILE, N, dtype=torch.bfloat16), **common)

    solve(x_tt, w_tt, sc_tt, out_tt)

    out = ttnn.to_torch(out_tt)[0:1]  # row 0
    assert_pcc(y_ref, out, threshold=threshold)


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    SHAPES = [
        # V4-Flash gate
        (4096, 256),
        # Smaller sanity shape
        (1024, 128),
    ]

    device = ttnn.open_device(device_id=0)
    try:
        for K, N in SHAPES:
            _test_shape(device, ttnn, torch, K=K, N=N, threshold=0.999)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)
