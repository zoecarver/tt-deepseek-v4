"""TT-Lang SUMMA matmul for V4-Flash attention linears.

Self-contained port of `../../tt-lang/benchmarks/matmul/summa_kernel.py`,
stripped of the sweep planner. Hand-picked configs per decode shape
(M=32 = one tile row). Not tuned for the last 10% of perf; fusion
variants come later.

Decode-time attention shapes handled (K, N):

  wq_a:  4096 x 1024
  wq_b:  1024 x 32768
  wkv:   4096 x 512
  wo_a:  4096 x 8192
  wo_b:  8192 x 4096
"""
from __future__ import annotations

from typing import Tuple

import ttl

TILE = 32


# (block_cfg, part_cfg) per (K, N). M=32 forces bm=1 / Mp=1.
# Block is (bm, bn, bk) in tiles; part is (M_parts, N_parts, K_parts).
# K_parts=1 here (pure SUMMA); K-split variants can be slotted in later.
# Picked to fit L1 (a_cb 2*bm*bk + b_cb 2*bk*bn + out_cb 2*bm*bn ≤ budget)
# and divide the output grid cleanly.
_PLANS: dict[Tuple[int, int], Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    (4096, 1024):  ((1, 4, 8), (1, 8, 1)),   # wq_a
    (1024, 32768): ((1, 8, 8), (1, 8, 1)),   # wq_b
    (4096, 512):   ((1, 4, 8), (1, 4, 1)),   # wkv
    (4096, 8192):  ((1, 8, 8), (1, 8, 1)),   # wo_a
    (8192, 4096):  ((1, 8, 8), (1, 8, 1)),   # wo_b
}


def make_summa_matmul(
    M: int,
    K: int,
    N: int,
    block_cfg: Tuple[int, int, int],
    part_cfg: Tuple[int, int, int],
    *,
    fp32_dest_acc_en: bool = True,
):
    """Pure SUMMA matmul kernel (K_parts must be 1).

    Grid (Np, Mp). Each core at (col_c, row_c) owns output block
    (row_c * M_BPN + i_m, col_c * N_BPN + i_n). Compute reduces over Kb
    blocks locally; A is row-mcast, B is column-mcast.
    """
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg

    if Kp != 1:
        raise ValueError(f"summa kernel is K_parts=1 only, got Kp={Kp}")
    if M % TILE or N % TILE or K % TILE:
        raise ValueError(f"M/K/N must be tile-aligned: M={M} K={K} N={N}")

    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape in tiles: Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})"
        )

    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Mb % Mp or Nb % Np:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} must divide Mp={Mp} Np={Np}"
        )

    M_BPN = Mb // Mp
    N_BPN = Nb // Np

    COL = Np
    ROW = Mp

    @ttl.operation(grid=(COL, ROW), fp32_dest_acc_en=fp32_dest_acc_en)
    def summa_matmul(a, w, out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p)) for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)

        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp))) for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = out_cb.reserve()
                    for _ in range(Kb):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        p += a_blk @ b_blk

        @ttl.datamovement()
        def dm_read():
            _, row_c = ttl.node(dims=2)
            for local_mb in range(M_BPN):
                mb = row_c * M_BPN + local_mb
                mr = mb * bm
                for _ in range(N_BPN):
                    for kb in range(Kb):
                        kc = kb * bk
                        a_blk = a_cb.reserve()

                        def read_a(pipe):
                            ttl.copy(a[mr : mr + bm, kc : kc + bk], a_blk).wait()
                            ttl.copy(a_blk, pipe).wait()

                        mcast_a_net.if_src(read_a)
                        mcast_a_net.if_dst(lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            for local_mb in range(M_BPN):
                mb = row_c * M_BPN + local_mb
                mr = mb * bm
                for local_nb in range(N_BPN):
                    nb = col_c * N_BPN + local_nb
                    nc = nb * bn
                    for kb in range(Kb):
                        kc = kb * bk
                        b_blk = b_cb.reserve()

                        def read_b(pipe):
                            ttl.copy(w[kc : kc + bk, nc : nc + bn], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    o = out_cb.wait()
                    ttl.copy(o, out[mr : mr + bm, nc : nc + bn]).wait()

    return summa_matmul


def solve(x_tt, w_tt, y_tt):
    """Run y = x @ w.

    x_tt: ttnn bf16 TILE_LAYOUT tensor, shape [M, K] where M is tile-aligned
          (M=32 for decode, row 0 valid).
    w_tt: ttnn bf16 TILE_LAYOUT tensor, shape [K, N].
    y_tt: ttnn bf16 TILE_LAYOUT tensor, shape [M, N].
    """
    M = x_tt.shape[0]
    K, N = w_tt.shape[0], w_tt.shape[1]
    if (K, N) not in _PLANS:
        raise ValueError(f"no plan for shape (K={K}, N={N}); add to _PLANS")
    block_cfg, part_cfg = _PLANS[(K, N)]
    kernel = make_summa_matmul(M, K, N, block_cfg, part_cfg)
    kernel(x_tt, w_tt, y_tt)


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _test_shape(device, ttnn, torch, K: int, N: int, threshold: float):
    from harness import assert_pcc

    print(f"\n[shape] K={K} N={N} (Kt={K // TILE}, Nt={N // TILE})")
    plan = _PLANS[(K, N)]
    print(f"  plan: block={plan[0]} part={plan[1]}")

    torch.manual_seed(0)
    w_nk = torch.randn(N, K, dtype=torch.float32) * (1.0 / (K ** 0.5))  # [N, K] nn.Linear order
    x_1k = torch.randn(1, K, dtype=torch.float32)
    y_ref = (x_1k @ w_nk.T).to(torch.bfloat16)

    # Activation padded to [32, K] tile-row (row 0 valid, rest zero).
    x_packed = torch.zeros(TILE, K, dtype=torch.bfloat16)
    x_packed[0] = x_1k.to(torch.bfloat16)

    # Weight as [K, N] = w_nk.T.
    w_kn = w_nk.T.contiguous().to(torch.bfloat16)

    y_packed = torch.zeros(TILE, N, dtype=torch.bfloat16)

    common = dict(
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.from_torch(x_packed, **common)
    w_tt = ttnn.from_torch(w_kn, **common)
    y_tt = ttnn.from_torch(y_packed, **common)

    solve(x_tt, w_tt, y_tt)

    y_out = ttnn.to_torch(y_tt)[0:1]  # row 0 -> [1, N]
    assert_pcc(y_ref, y_out, threshold=threshold)


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    THRESHOLD = 0.999  # bf16 bar

    SHAPES = [
        (4096, 1024),    # wq_a
        (1024, 32768),   # wq_b
        (4096, 512),     # wkv
        (4096, 8192),    # wo_a
        (8192, 4096),    # wo_b
    ]

    device = ttnn.open_device(device_id=0)
    try:
        for K, N in SHAPES:
            _test_shape(device, ttnn, torch, K=K, N=N, threshold=THRESHOLD)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)
