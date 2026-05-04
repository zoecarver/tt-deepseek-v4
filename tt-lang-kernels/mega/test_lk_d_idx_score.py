"""Lk-D-idx-score PCC test: indexer weights_proj + score reduce.

Reference covers the tail of `DeviceIndexer.forward_device_score`:
- ttnn.matmul(x, weights_proj) -> w_tt
- ttnn.multiply(w_tt, scale)
- ttnn.transpose(kv_cache_tt, -2, -1) -> kv_T
- ttnn.matmul(q_idx, kv_T) -> score
- ttnn.relu, ttnn.transpose, ttnn.reshape, ttnn.multiply, ttnn.sum -> [B, 1, T_pad]

Boundaries: pre-CCL is indexer.weights_proj all_gather; post is no CCL
(score is consumed locally by Lk-D-topk).

Three inlined tt-lang kernels:
  - _make_summa_matmul_kernel: x @ wproj_scaled and q_idx @ kv_T
  - _make_summa_matmul_relu_b_kernel: SUMMA variant that fuses relu on
    the B operand and accumulates to produce the score reduce.

Math equivalence for the post-process:
  ref:    out[t] = sum_h relu(score[h, t]) * w[h]
  kernel: out [TILE, T] = w_padded [TILE, H] @ relu(score [H, T])
          (only row 0 of out is valid)

ttnn glue: none on the score path - kv_T transpose is folded into the
score matmul via transpose_b=True (per-block ttl.transpose). The
`ttnn.multiply(w, scale)` is folded by pre-scaling wproj on the host.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark


DIM = 4096
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
T_PAD = 128
B = 1
TILE = 32


def _summa_dims(M: int, K: int, N: int, block_cfg, part_cfg):
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Kp != 1:
        raise ValueError(f"K_parts must be 1, got {Kp}")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape (tiles): Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})")
    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Mb % Mp or Nb % Np:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} Mp={Mp} Np={Np}")
    return bm, bn, bk, Mp, Np, Mb // Mp, Nb // Np, Kb


def _make_summa_matmul_kernel(M: int, K: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    bm, bn, bk, Mp, Np, M_BPN, N_BPN, Kb = _summa_dims(
        M, K, N, block_cfg, part_cfg)

    @ttl.operation(grid=(Np, Mp), fp32_dest_acc_en=fp32_dest_acc_en)
    def summa_matmul(a, w, out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p))
                   for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp)))
                   for n_p in range(Np)]
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
                            ttl.copy(a[mr:mr + bm, kc:kc + bk], a_blk).wait()
                            ttl.copy(a_blk, pipe).wait()

                        mcast_a_net.if_src(read_a)
                        mcast_a_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

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
                            ttl.copy(w[kc:kc + bk, nc:nc + bn], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    o = out_cb.wait()
                    ttl.copy(o, out[mr:mr + bm, nc:nc + bn]).wait()

    return summa_matmul


def _make_summa_matmul_b_t_kernel(M: int, K: int, N: int,
                                  block_cfg, part_cfg,
                                  fp32_dest_acc_en: bool = True):
    """SUMMA matmul out = a @ b.T. b is stored as [N, K] (not [K, N]).

    Folds an external ttnn.transpose(b, -2, -1) into the matmul: each
    block of b is read transposed (read [N, K] tiles at [nc, kc]) and
    transposed inside the compute thread before the inner matmul.
    """
    bm, bn, bk, Mp, Np, M_BPN, N_BPN, Kb = _summa_dims(
        M, K, N, block_cfg, part_cfg)

    @ttl.operation(grid=(Np, Mp), fp32_dest_acc_en=fp32_dest_acc_en)
    def summa_matmul_b_t(a, w, out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p))
                   for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp)))
                   for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bn, bk), block_count=2)
        bt_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = out_cb.reserve()
                    for _ in range(Kb):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        with bt_cb.reserve() as bt:
                            bt.store(ttl.transpose(b_blk))
                        p += a_blk @ bt_cb.wait()

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
                            ttl.copy(a[mr:mr + bm, kc:kc + bk], a_blk).wait()
                            ttl.copy(a_blk, pipe).wait()

                        mcast_a_net.if_src(read_a)
                        mcast_a_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

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
                            ttl.copy(w[nc:nc + bn, kc:kc + bk], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    o = out_cb.wait()
                    ttl.copy(o, out[mr:mr + bm, nc:nc + bn]).wait()

    return summa_matmul_b_t


def _make_summa_matmul_relu_b_kernel(M: int, K: int, N: int,
                                     block_cfg, part_cfg,
                                     fp32_dest_acc_en: bool = True):
    """SUMMA variant that fuses elementwise relu on the B operand.

    Computes out = a @ relu(b). Used for the score reduce in
    Lk-D-idx-score: out [TILE, T] = w_padded [TILE, H] @ relu(score [H, T]).
    Replaces the ttnn.relu + ttnn.multiply + ttnn.sum tail.
    """
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Kp != 1:
        raise ValueError(f"K_parts must be 1, got {Kp}")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape (tiles): Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})")
    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Mb % Mp or Nb % Np:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} Mp={Mp} Np={Np}")
    M_BPN = Mb // Mp
    N_BPN = Nb // Np

    @ttl.operation(grid=(Np, Mp), fp32_dest_acc_en=fp32_dest_acc_en)
    def summa_matmul_relu_b(a, w, out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p))
                   for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp)))
                   for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        b_relu_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = out_cb.reserve()
                    for _ in range(Kb):
                        a_blk = a_cb.wait()
                        b_raw = b_cb.wait()
                        b_relu = b_relu_cb.reserve()
                        b_relu.store(ttl.math.max(b_raw, ttl.math.fill(b_raw, 0.0)))
                        p += a_blk @ b_relu_cb.wait()

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
                            ttl.copy(a[mr:mr + bm, kc:kc + bk], a_blk).wait()
                            ttl.copy(a_blk, pipe).wait()

                        mcast_a_net.if_src(read_a)
                        mcast_a_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

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
                            ttl.copy(w[kc:kc + bk, nc:nc + bn], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    o = out_cb.wait()
                    ttl.copy(o, out[mr:mr + bm, nc:nc + bn]).wait()

    return summa_matmul_relu_b


def make_lk_d_idx_score_kernel(mesh):
    """Mega kernel for Lk-D-idx-score.

    Pipeline:
      x_2d [1, 4096] -> pad to [TILE, 4096]
      SUMMA wproj_scaled: x_padded @ wproj_scaled -> w_padded [TILE, H]
      SUMMA qkv_T (B-transpose fused): q_idx [H, d] @ kv [T, d].T -> score [H, T_pad]
      SUMMA relu_b reduce: w_padded @ relu(score) -> out_padded [TILE, T_pad]
      slice/reshape -> [B, 1, T_pad]
    """
    H = INDEX_N_HEADS
    d = INDEX_HEAD_DIM
    T = T_PAD

    # SUMMA1: M=TILE, K=4096, N=H=64. Mt=1, Kt=128, Nt=2.
    # block=(1, 1, 8) part=(1, 2, 1) -> 2 cores (Nb=2, M_BPN=N_BPN=1, Kb=16).
    matmul_wproj_kernel = _make_summa_matmul_kernel(
        M=TILE, K=DIM, N=H,
        block_cfg=(1, 1, 8), part_cfg=(1, 2, 1))

    # SUMMA2: M=H=64, K=d=128, N=T=128. Mt=2, Kt=4, Nt=4.
    # block=(1, 1, 4) part=(2, 4, 1) -> 8 cores (Mb=2, Nb=4, M_BPN=N_BPN=1, Kb=1).
    # B-transpose variant: folds kv_T transpose into the matmul.
    matmul_score_kernel = _make_summa_matmul_b_t_kernel(
        M=H, K=d, N=T,
        block_cfg=(1, 1, 4), part_cfg=(2, 4, 1))

    # SUMMA3 (relu fused on B): M=TILE, K=H=64, N=T=128. Mt=1, Kt=2, Nt=4.
    # block=(1, 1, 2) part=(1, 4, 1) -> 4 cores (Nb=4, Mb=1, M_BPN=N_BPN=1, Kb=1).
    matmul_relu_reduce_kernel = _make_summa_matmul_relu_b_kernel(
        M=TILE, K=H, N=T,
        block_cfg=(1, 1, 2), part_cfg=(1, 4, 1))

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    state: dict = {}

    def lk_d_idx_score_kernel(x_tt, wproj_scaled_tt, q_idx_tt, kv_cache_tt,
                              score_out):
        if "scratch" not in state:
            state["w_padded"] = ttnn.from_torch(
                torch.zeros(TILE, H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["score_padded"] = ttnn.from_torch(
                torch.zeros(H, T, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["out_padded"] = ttnn.from_torch(
                torch.zeros(TILE, T, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["scratch"] = True

        # Pad x [1, 1, dim] -> [TILE, dim].
        x_2d = ttnn.reshape(x_tt, [B, DIM])
        x_padded = ttnn.pad(x_2d, padding=[(0, TILE - B), (0, 0)], value=0.0)

        # SUMMA: x_padded @ wproj_scaled -> w_padded [TILE, H] (row 0 valid).
        matmul_wproj_kernel(x_padded, wproj_scaled_tt, state["w_padded"])

        # SUMMA with transpose_b=True: q_idx @ kv.T -> score [H, T]. The
        # transpose folds into the matmul (per-block ttl.transpose); kv is
        # passed in its natural [T, d] layout, no ttnn.transpose dispatch.
        kv_2d = ttnn.reshape(kv_cache_tt, [T, d])
        q_idx_2d = ttnn.reshape(q_idx_tt, [H, d])
        matmul_score_kernel(q_idx_2d, kv_2d, state["score_padded"])

        # SUMMA with fused relu(B): w_padded @ relu(score) -> out [TILE, T].
        matmul_relu_reduce_kernel(state["w_padded"], state["score_padded"],
                                  state["out_padded"])

        out_row = ttnn.slice(state["out_padded"], [0, 0], [B, T])
        out_3d = ttnn.reshape(out_row, [B, 1, T])
        ttnn.copy(out_3d, score_out)

    return lk_d_idx_score_kernel


def reference(mesh, x_tt, weights_proj_w_tt, q_idx_tt, kv_cache_tt, scale):
    H = INDEX_N_HEADS
    w_tt = ttnn.matmul(x_tt, weights_proj_w_tt,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w_tt = ttnn.multiply(w_tt, scale)
    kv_T = ttnn.transpose(kv_cache_tt, -2, -1)
    score = ttnn.matmul(q_idx_tt, kv_T)
    score = ttnn.relu(score)
    score_t = ttnn.transpose(score, -2, -1)
    w_b = ttnn.reshape(w_tt, [B, 1, 1, H])
    score_t = ttnn.multiply(score_t, w_b)
    return ttnn.sum(score_t, dim=-1, keepdim=False)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        x = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        weights_proj_w = torch.randn(DIM, INDEX_N_HEADS, dtype=torch.bfloat16) * 0.02
        q_idx = torch.randn(1, 1, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_cache = torch.randn(1, 1, T_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        scale = float(INDEX_HEAD_DIM ** -0.5) * float(INDEX_N_HEADS ** -0.5)

        # Pre-scale wproj for the kernel path so the kernel has no scalar
        # multiply step.
        weights_proj_scaled = (weights_proj_w.to(torch.float32) * scale).to(torch.bfloat16)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep)
        wproj_tt = ttnn.as_tensor(weights_proj_w.contiguous(),
                                  dtype=ttnn.bfloat16, **rep)
        wproj_scaled_tt = ttnn.as_tensor(weights_proj_scaled.contiguous(),
                                         dtype=ttnn.bfloat16, **rep)
        q_idx_tt = ttnn.as_tensor(q_idx.contiguous(), dtype=ttnn.bfloat16, **rep)
        kv_cache_tt = ttnn.as_tensor(kv_cache.contiguous(),
                                     dtype=ttnn.bfloat16, **rep)

        ref_out_tt = reference(mesh, x_tt, wproj_tt, q_idx_tt, kv_cache_tt, scale)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_d_idx_score_kernel(mesh)
        score_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, T_PAD, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(x_tt, wproj_scaled_tt, q_idx_tt, kv_cache_tt, score_out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, score_out_tt)

        ok = report_pcc("Lk-D-idx-score", ref_host, kernel_host)

        benchmark("Lk-D-idx-score ref",
                  lambda: reference(mesh, x_tt, wproj_tt, q_idx_tt,
                                    kv_cache_tt, scale),
                  mesh)
        benchmark("Lk-D-idx-score ttl",
                  lambda: kernel(x_tt, wproj_scaled_tt, q_idx_tt,
                                 kv_cache_tt, score_out_tt),
                  mesh)

        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
