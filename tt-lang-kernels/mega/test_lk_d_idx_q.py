"""Lk-D-idx-q PCC test: indexer q-stack (sans indexer.wq_b all_gather).

Reference is the start of `DeviceIndexer.forward_device_score`:
- self.wq_b.forward_device(qr_tt) — modeled here as a plain matmul
- reshape to [B, 1, H, D]
- pick cos/sin via embedding(start_pos, table); reshape to [1,1,1,rd/2]
- slice q nope/rope, rotary on rope, concat
- Walsh-Hadamard rotation (matmul against H constant)

The indexer.wq_b all_gather is excluded — for the test the weight is replicated.

All tt-lang kernel definitions are inlined in this file (SUMMA matmul,
used twice: once for indexer.wq_b and once for the Walsh-Hadamard
matmul at d=128). Rotary is still a ttnn helper — see TODO below.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    _device_apply_rotary_interleaved,
    _sylvester_hadamard,
)


Q_LORA_RANK = 1024
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
ROPE_HEAD_DIM = 64
MAX_SEQ_LEN = 512
B, S = 1, 1
TILE = 32

# TODO: mega the rotary helper `_device_apply_rotary_interleaved` is still
# called from the reference path (and from the kernel wrapper). It walks
# pairs (real, imag) at sub-tile granularity (rd_half = 32 = exactly one
# tile column for d=128, and the rope half of head_dim=128 is 64 = two
# tiles). Lowering it to tt-lang means picking a layout that lets
# ttl.copy and ttl.math express either:
#   (a) interleaved-pair rotation using transpose + sub-tile masks, or
#   (b) a precomputed permutation on the K axis so that pairs land in
#       the same tile.
# Punt for now; once Lk-C / Lk-D1 rotary lowerings land we can share.


def _make_summa_matmul_kernel(M: int, K: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    """Pure-SUMMA matmul kernel (same shape as Lk-A/Lk-B).

    Output = a @ w. A is row-mcast across Np cores, B is column-mcast
    across Mp cores. Each core owns an M_BPN x N_BPN output sub-grid.
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


def make_lk_d_idx_q_kernel(mesh):
    """Mega kernel for Lk-D-idx-q.

    Two SUMMA matmul dispatches plus a ttnn rotary glue (TODO above).

      qr [1,1,1024]
        → pad to [TILE, 1024]
        → SUMMA matmul against indexer.wq_b [1024, 8192]
        → [TILE, 8192] -> reshape [1, 1, H=64, D=128]
        → slice nope/rope, ttnn rotary (TODO), concat
        → reshape [H, D] = [64, 128]
        → SUMMA matmul against H_d/sqrt(d) [128, 128]
        → reshape [1, 1, 64, 128]
    """
    H = INDEX_N_HEADS
    D = INDEX_HEAD_DIM
    rd = ROPE_HEAD_DIM
    rd_half = rd // 2
    N_WQB = H * D                          # 8192
    M_PAD = TILE                           # qr is [1,1,1024]; pad to 1 tile

    state: dict = {}

    # SUMMA #1: [TILE, Q_LORA_RANK] @ [Q_LORA_RANK, N_WQB] -> [TILE, N_WQB]
    # Mt=1, Kt=32, Nt=256. block=(1, 4, 8) -> Mb=1, Nb=64, Kb=4. Np=8 -> N_BPN=8.
    matmul_wqb_kernel = _make_summa_matmul_kernel(
        M=M_PAD, K=Q_LORA_RANK, N=N_WQB,
        block_cfg=(1, 4, 8), part_cfg=(1, 8, 1))

    # SUMMA #2: [H=64, D=128] @ [D, D] -> [H, D]
    # Mt=2, Kt=4, Nt=4. block=(1, 4, 4) -> Mb=2, Nb=1, Kb=1. Mp=2, Np=1
    # -> M_BPN=1, N_BPN=1.
    matmul_hada_kernel = _make_summa_matmul_kernel(
        M=H, K=D, N=D,
        block_cfg=(1, 4, 4), part_cfg=(2, 1, 1))

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def lk_d_idx_q_kernel(qr_tt, cos_full_tt, sin_full_tt, start_pos_tt,
                          indexer_wq_b_tt, H_tt, out):
        if "scratch" not in state:
            state["q_padded_tt"] = ttnn.from_torch(
                torch.zeros((M_PAD, N_WQB), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["q_h_d_tt"] = ttnn.from_torch(
                torch.zeros((H, D), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["q_rotated_tt"] = ttnn.from_torch(
                torch.zeros((H, D), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["scratch"] = True

        # 1. Pad qr [1,1,K] -> [TILE, K] for SUMMA M-axis.
        qr_2d = ttnn.reshape(qr_tt, [B * S, Q_LORA_RANK])
        qr_padded = ttnn.pad(
            qr_2d, padding=[(0, M_PAD - B * S), (0, 0)], value=0.0)

        # 2. SUMMA #1: indexer.wq_b matmul -> [TILE, 8192].
        matmul_wqb_kernel(qr_padded, indexer_wq_b_tt, state["q_padded_tt"])

        # 3. Slice [TILE, 8192] -> [B*S, 8192] and reshape [1, 1, H, D].
        q_row = ttnn.slice(state["q_padded_tt"], [0, 0], [B * S, N_WQB])
        q_4d = ttnn.reshape(q_row, [B, S, H, D])

        # 4. ttnn rotary on rope half (TODO above). Output is [1, 1, H, D].
        cos = ttnn.embedding(start_pos_tt, cos_full_tt, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(start_pos_tt, sin_full_tt, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, 1, 1, rd_half])
        sin = ttnn.reshape(sin, [1, 1, 1, rd_half])
        q_nope = ttnn.slice(q_4d, [0, 0, 0, 0], [B, 1, H, D - rd])
        q_rope = ttnn.slice(q_4d, [0, 0, 0, D - rd], [B, 1, H, D])
        q_rope = _device_apply_rotary_interleaved(
            ttnn, q_rope, cos, sin, inverse=False)
        q_full = ttnn.concat([q_nope, q_rope], dim=-1)

        # 5. Reshape [1, 1, H, D] -> [H, D] for SUMMA #2.
        q_2d = ttnn.reshape(q_full, [H, D])
        ttnn.copy(q_2d, state["q_h_d_tt"])

        # 6. SUMMA #2: Walsh-Hadamard matmul -> [H, D].
        matmul_hada_kernel(state["q_h_d_tt"], H_tt, state["q_rotated_tt"])

        # 7. Reshape back [1, 1, H, D] and copy to test-provided out.
        q_out_4d = ttnn.reshape(state["q_rotated_tt"], [B, S, H, D])
        ttnn.copy(q_out_4d, out)

    return lk_d_idx_q_kernel


def reference(mesh, qr_tt, cos_full_tt, sin_full_tt, start_pos_tt,
              indexer_wq_b_tt, H_tt):
    H = INDEX_N_HEADS
    D = INDEX_HEAD_DIM
    rd = ROPE_HEAD_DIM
    rd_half = rd // 2

    # indexer.wq_b matmul (all_gather excluded).
    q_tt = ttnn.matmul(qr_tt, indexer_wq_b_tt,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q_tt = ttnn.reshape(q_tt, [B, 1, H, D])
    cos = ttnn.embedding(start_pos_tt, cos_full_tt, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(start_pos_tt, sin_full_tt, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, 1, 1, rd_half])
    sin = ttnn.reshape(sin, [1, 1, 1, rd_half])
    q_nope = ttnn.slice(q_tt, [0, 0, 0, 0], [B, 1, H, D - rd])
    q_rope = ttnn.slice(q_tt, [0, 0, 0, D - rd], [B, 1, H, D])
    q_rope = _device_apply_rotary_interleaved(ttnn, q_rope, cos, sin, inverse=False)
    q_tt = ttnn.concat([q_nope, q_rope], dim=-1)
    # Walsh-Hadamard rotation (single matmul).
    q_tt = ttnn.matmul(q_tt, H_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return q_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        qr = torch.randn(1, 1, Q_LORA_RANK, dtype=torch.bfloat16) * 0.1
        cos_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        sin_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        indexer_wq_b = torch.randn(Q_LORA_RANK, INDEX_N_HEADS * INDEX_HEAD_DIM,
                                   dtype=torch.bfloat16) * 0.02
        H_mat = (_sylvester_hadamard(INDEX_HEAD_DIM) *
                 (INDEX_HEAD_DIM ** -0.5)).to(torch.bfloat16)
        start_pos = torch.tensor([[1]], dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        qr_tt = ttnn.as_tensor(qr.contiguous(), dtype=ttnn.bfloat16, **rep)
        cos_full_tt = ttnn.as_tensor(cos_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        sin_full_tt = ttnn.as_tensor(sin_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        wq_b_tt = ttnn.as_tensor(indexer_wq_b.contiguous(), dtype=ttnn.bfloat16, **rep)
        H_tt = ttnn.as_tensor(H_mat.contiguous(), dtype=ttnn.bfloat16, **rep)
        start_pos_tt = ttnn.from_torch(
            start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        ref_out_tt = reference(mesh, qr_tt, cos_full_tt, sin_full_tt,
                               start_pos_tt, wq_b_tt, H_tt)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_d_idx_q_kernel(mesh)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(qr_tt, cos_full_tt, sin_full_tt, start_pos_tt, wq_b_tt, H_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-D-idx-q", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
