"""Lk-D-idx-q PCC test: indexer q-stack (sans indexer.wq_b all_gather).

Reference is the start of `DeviceIndexer.forward_device_score`:
- self.wq_b.forward_device(qr_tt) — modeled here as a plain matmul
- reshape to [B, 1, H, D]
- pick cos/sin via embedding(start_pos, table); reshape to [1,1,1,rd/2]
- slice q nope/rope, rotary on rope, concat
- Walsh-Hadamard rotation (matmul against H constant)

The indexer.wq_b all_gather is excluded — for the test the weight is replicated.

All tt-lang kernel definitions are inlined in this file: SUMMA matmul
(used twice for indexer.wq_b + Walsh-Hadamard), and the rotary
swap-SUMMA + rotary-combine pair (same lowering as Lk-C/Lk-D1).

Rotary lowering trick (shared with Lk-C/Lk-D1): bake rotate_half
into a swap matrix P and the sin sign into sin_signed. Then
  out = x * cos_extended + (x @ P) * sin_signed
A small SUMMA matmul does x @ P; an elementwise tt-lang kernel does
the combine. Cos/sin tables are pre-replicated across TILE rows on
host so the kernel reads tile-aligned tensors.

ttnn glue (TODO: mega): ttnn.embedding(start_pos, ...) still depends
on a device uint32 index (no tt-lang gather primitive).
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark

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


def _make_ksplit_matmul_kernel(M: int, K: int, N: int,
                               block_cfg, part_cfg,
                               fp32_dest_acc_en: bool = True):
    """SUMMA matmul with K-split on the row axis. grid=(Np, Kp), Mp=1.

    K is split across Kp row cores; non-root rows ship partials to root
    (k_p=0) for summation and write-out. M is fixed at one bm-block.
    """
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Mp != 1:
        raise ValueError(f"ksplit kernel here assumes Mp=1, got {Mp}")
    if Kp < 2:
        raise ValueError(f"K_parts must be >= 2, got {Kp}")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape (tiles): Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})")
    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Nb % Np or Kb % Kp or Mb != 1:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} Kb={Kb} Np={Np} Kp={Kp}")
    N_BPN = Nb // Np
    K_BPN = Kb // Kp

    @ttl.operation(grid=(Np, Kp), fp32_dest_acc_en=fp32_dest_acc_en)
    def ksplit_matmul(a, w, out):
        a_pipes = [ttl.Pipe(src=(0, k_p), dst=(slice(0, Np), k_p))
                   for k_p in range(Kp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        reduce_pipes = [ttl.Pipe(src=(n_p, k_p), dst=(n_p, 0))
                        for n_p in range(Np) for k_p in range(1, Kp)]
        reduce_net = ttl.PipeNet(reduce_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        partial_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        recv_cb = ttl.make_dataflow_buffer_like(
            out, shape=(bm, bn), block_count=max(2, Kp - 1))
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            _, row_c = ttl.node(dims=2)
            for _ in range(N_BPN):
                p = partial_cb.reserve()
                for _ in range(K_BPN):
                    a_blk = a_cb.wait()
                    b_blk = b_cb.wait()
                    p += a_blk @ b_blk

                if row_c == 0:
                    for _ in range(Kp - 1):
                        prev = partial_cb.wait()
                        r = recv_cb.wait()
                        new = partial_cb.reserve()
                        new.store(prev + r)
                    final = partial_cb.wait()
                    o = out_cb.reserve()
                    o.store(final)

        @ttl.datamovement()
        def dm_read():
            _, row_c = ttl.node(dims=2)
            for _ in range(N_BPN):
                for kb_local in range(K_BPN):
                    kc = (row_c * K_BPN + kb_local) * bk
                    a_blk = a_cb.reserve()

                    def read_a(pipe):
                        ttl.copy(a[0:bm, kc:kc + bk], a_blk).wait()
                        ttl.copy(a_blk, pipe).wait()

                    mcast_a_net.if_src(read_a)
                    mcast_a_net.if_dst(
                        lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

                if row_c == 0:
                    def recv(pipe):
                        r = recv_cb.reserve()
                        ttl.copy(pipe, r).wait()

                    reduce_net.if_dst(recv)
                else:
                    p = partial_cb.wait()

                    def send(pipe):
                        ttl.copy(p, pipe).wait()

                    reduce_net.if_src(send)

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            for local_nb in range(N_BPN):
                nb = col_c * N_BPN + local_nb
                nc = nb * bn
                for kb_local in range(K_BPN):
                    kc = (row_c * K_BPN + kb_local) * bk
                    b_blk = b_cb.reserve()
                    ttl.copy(w[kc:kc + bk, nc:nc + bn], b_blk).wait()
                if row_c == 0:
                    o = out_cb.wait()
                    ttl.copy(o, out[0:bm, nc:nc + bn]).wait()

    return ksplit_matmul


def _make_fused_rot_hada_kernel(M: int, D: int, rd: int,
                                fp32_dest_acc_en: bool = True):
    """Fused rotary + Walsh-Hadamard matmul on the [H, D] post-wq_b layout.

    Computes per output row h, output col n:
      q_rot[h, c] = q_full[h, D-rd+c] * cos[c] + (q_rope @ P_diag)[h, c] * sin[c]
      out[h, n] = sum_{k in [0, D-rd)} q_full[h, k] * H[k, n]
                + sum_{c in [0, rd)} q_rot[h, c]   * H[D-rd+c, n]

    P is block-diagonal: P[c1, c2] = 1 iff c1==c2 XOR 1 within a 2-wide pair.
    Per K-tile, P[k_tile, k_tile] is a 32x32 swap matrix; off-diagonal blocks
    are 0. So the swap is local within each K-tile.

    Grid: (1, Mp). Each core handles one M-tile of output.
    K-axis split into nope_t nope tiles + rd_t rope tiles.
    """
    Mt = M // TILE
    Dt = D // TILE
    rd_t = rd // TILE
    nope_t = Dt - rd_t

    Mp = Mt
    Np = 1
    bm = 1
    bn = Dt
    bk = 2

    if nope_t % bk or rd_t % bk:
        raise ValueError(
            f"bk={bk} must divide nope_t={nope_t} and rd_t={rd_t}")
    M_BPN = 1
    N_BPN = 1
    Kb_nope = nope_t // bk
    Kb_rope = rd_t // bk

    @ttl.operation(grid=(Np, Mp), fp32_dest_acc_en=fp32_dest_acc_en)
    def fused_rot_hada(q_full, P, cos, sin, H_mat, out):
        q_nope_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(bm, bk), block_count=2)
        q_rope_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(bm, bk), block_count=2)
        P_cb = ttl.make_dataflow_buffer_like(
            P, shape=(bk, bk), block_count=2)
        cos_cb = ttl.make_dataflow_buffer_like(
            cos, shape=(1, bk), block_count=2)
        sin_cb = ttl.make_dataflow_buffer_like(
            sin, shape=(1, bk), block_count=2)
        swap_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(bm, bk), block_count=2)
        rope_rot_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(bm, bk), block_count=max(2, Kb_rope))
        H_cb = ttl.make_dataflow_buffer_like(
            H_mat, shape=(bk, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(
            out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            # Phase 1: rotate rope half, store all rd_t tiles in rope_rot_cb.
            for _ in range(Kb_rope):
                q_rope_blk = q_rope_cb.wait()
                P_blk = P_cb.wait()
                c_blk = cos_cb.wait()
                si_blk = sin_cb.wait()
                swap_cb.reserve().store(q_rope_blk @ P_blk)
                swap_blk = swap_cb.wait()
                rope_rot_cb.reserve().store(
                    q_rope_blk * c_blk + swap_blk * si_blk)

            # Phase 2: Walsh-Hadamard matmul over full K=D.
            p = out_cb.reserve()
            for _ in range(Kb_nope):
                q_blk = q_nope_cb.wait()
                H_blk = H_cb.wait()
                p += q_blk @ H_blk
            for _ in range(Kb_rope):
                q_blk = rope_rot_cb.wait()
                H_blk = H_cb.wait()
                p += q_blk @ H_blk

        @ttl.datamovement()
        def dm_read():
            _, row_c = ttl.node(dims=2)
            mr = row_c * M_BPN * bm
            # Phase 1 inputs: rope blocks + per-block P-diag + cos + sin.
            for k in range(Kb_rope):
                rope_kc = nope_t + k * bk
                diag_kc = k * bk
                ttl.copy(q_full[mr:mr + bm, rope_kc:rope_kc + bk],
                         q_rope_cb.reserve()).wait()
                ttl.copy(P[diag_kc:diag_kc + bk, diag_kc:diag_kc + bk],
                         P_cb.reserve()).wait()
                ttl.copy(cos[0:1, diag_kc:diag_kc + bk],
                         cos_cb.reserve()).wait()
                ttl.copy(sin[0:1, diag_kc:diag_kc + bk],
                         sin_cb.reserve()).wait()
            # Phase 2 nope inputs: q_nope blocks in K-order.
            for k in range(Kb_nope):
                kc = k * bk
                ttl.copy(q_full[mr:mr + bm, kc:kc + bk],
                         q_nope_cb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            mr = row_c * M_BPN * bm
            nc = col_c * N_BPN * bn
            # H tiles in K-order: nope range first, then rope range.
            for k in range(Kb_nope):
                kc = k * bk
                ttl.copy(H_mat[kc:kc + bk, nc:nc + bn],
                         H_cb.reserve()).wait()
            for k in range(Kb_rope):
                kc = nope_t + k * bk
                ttl.copy(H_mat[kc:kc + bk, nc:nc + bn],
                         H_cb.reserve()).wait()
            o = out_cb.wait()
            ttl.copy(o, out[mr:mr + bm, nc:nc + bn]).wait()

    return fused_rot_hada


def _build_rotary_tables(cos_full_cpu: torch.Tensor, sin_full_cpu: torch.Tensor,
                         inverse: bool):
    """Pre-replicated cos/sin tables; see Lk-C for the shape rationale."""
    max_seq_len, rd_half = cos_full_cpu.shape
    rd = 2 * rd_half
    if rd % TILE != 0:
        raise ValueError(f"rd={rd} not multiple of TILE={TILE}")
    cos_extended = cos_full_cpu.repeat_interleave(2, dim=-1)
    sign = torch.ones(rd, dtype=cos_full_cpu.dtype)
    if inverse:
        sign[1::2] = -1
    else:
        sign[0::2] = -1
    sin_signed = sin_full_cpu.repeat_interleave(2, dim=-1) * sign
    cos_extended_packed = cos_extended.unsqueeze(1).expand(
        max_seq_len, TILE, rd).reshape(max_seq_len, TILE * rd).contiguous()
    sin_signed_packed = sin_signed.unsqueeze(1).expand(
        max_seq_len, TILE, rd).reshape(max_seq_len, TILE * rd).contiguous()
    return cos_extended_packed, sin_signed_packed


def _build_swap_matrix(rd: int) -> torch.Tensor:
    """Block-diagonal swap matrix P [rd, rd] with 2x2 [[0,1],[1,0]] blocks."""
    if rd % 2:
        raise ValueError(f"rd={rd} must be even")
    P = torch.zeros(rd, rd, dtype=torch.bfloat16)
    for k in range(rd // 2):
        P[2 * k, 2 * k + 1] = 1.0
        P[2 * k + 1, 2 * k] = 1.0
    return P


def make_lk_d_idx_q_kernel(mesh, cos_full_cpu, sin_full_cpu):
    """Mega kernel for Lk-D-idx-q.

    Two tt-lang dispatches: indexer.wq_b ksplit SUMMA, then a single
    fused rotary + Walsh-Hadamard kernel.

      qr [1,1,1024]
        → pad to [TILE, 1024]
        → SUMMA matmul against indexer.wq_b [1024, 8192]
        → [TILE, 8192] -> reshape [H=64, D=128]
        → fused (rotary on rope half + matmul against Hadamard [D, D])
        → reshape [1, 1, 64, 128]

    The wq_b ksplit (64 cores) and rot+hada (Mp=2 cores) cannot share a
    grid — wq_b's output is sharded across 8 root cores by N-cols while
    rot+hada partitions by H-rows. # TODO: mega fusion blocked: wq_b
    output redistribution to rot+hada grid requires DRAM staging or a
    fan-in PipeNet that buffers >100KB per phase-2 core.
    """
    H = INDEX_N_HEADS
    D = INDEX_HEAD_DIM
    rd = ROPE_HEAD_DIM
    N_WQB = H * D                          # 8192
    M_PAD = TILE                           # qr is [1,1,1024]; pad to 1 tile

    state: dict = {}

    # KSPLIT #1: [TILE, Q_LORA_RANK=1024] @ [..., N_WQB=8192] -> [TILE, N_WQB]
    # Mt=1, Kt=32, Nt=256. block=(1, 4, 4) part=(1, 8, 8) -> 64 cores
    # (Nb=64, N_BPN=8, Kb=8, K_BPN=1).
    matmul_wqb_kernel = _make_ksplit_matmul_kernel(
        M=M_PAD, K=Q_LORA_RANK, N=N_WQB,
        block_cfg=(1, 4, 4), part_cfg=(1, 8, 8))

    # Fused rotary + Hadamard: M=H=64 split over Mp=2 (2 cores, each
    # handling one M-tile of 32 rows). bk=1 so we walk K-tiles individually
    # and branch nope vs rope per-tile.
    rot_hada_kernel = _make_fused_rot_hada_kernel(M=H, D=D, rd=rd)

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    cos_ext_packed, sin_signed_packed = _build_rotary_tables(
        cos_full_cpu, sin_full_cpu, inverse=False)
    cos_ext_tt = ttnn.as_tensor(cos_ext_packed, dtype=ttnn.bfloat16, **rep)
    sin_signed_tt = ttnn.as_tensor(sin_signed_packed, dtype=ttnn.bfloat16, **rep)
    P_cpu = _build_swap_matrix(rd)
    P_tt = ttnn.as_tensor(P_cpu.contiguous(), dtype=ttnn.bfloat16, **rep)

    def lk_d_idx_q_kernel(qr_tt, cos_full_tt, sin_full_tt, start_pos_tt,
                          indexer_wq_b_tt, H_tt, out):
        if "scratch" not in state:
            state["q_padded_tt"] = ttnn.from_torch(
                torch.zeros((M_PAD, N_WQB), dtype=torch.bfloat16),
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

        # 3. Slice [TILE, 8192] -> [B*S, 8192] and reshape [H, D].
        q_row = ttnn.slice(state["q_padded_tt"], [0, 0], [B * S, N_WQB])
        q_2d = ttnn.reshape(q_row, [H, D])

        # 4. Fused rotary on rope half + Walsh-Hadamard matmul.
        # TODO: mega fusion blocked: ttnn used for embedding(start_pos, ...)
        # to look up cos/sin (depends on a device uint32 index). Element_read
        # primitive exists; gather wiring is task C8.
        cos_b_2d = ttnn.embedding(start_pos_tt, cos_ext_tt, layout=ttnn.TILE_LAYOUT)
        sin_b_2d = ttnn.embedding(start_pos_tt, sin_signed_tt, layout=ttnn.TILE_LAYOUT)
        cos_b = ttnn.reshape(cos_b_2d, [TILE, rd])
        sin_b = ttnn.reshape(sin_b_2d, [TILE, rd])

        rot_hada_kernel(q_2d, P_tt, cos_b, sin_b, H_tt, state["q_rotated_tt"])

        # 5. Reshape back [1, 1, H, D] and copy to test-provided out.
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

        kernel = make_lk_d_idx_q_kernel(mesh, cos_full, sin_full)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(qr_tt, cos_full_tt, sin_full_tt, start_pos_tt, wq_b_tt, H_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-D-idx-q", ref_host, kernel_host)

        benchmark("Lk-D-idx-q ref",
                  lambda: reference(mesh, qr_tt, cos_full_tt, sin_full_tt,
                                    start_pos_tt, wq_b_tt, H_tt),
                  mesh)
        benchmark("Lk-D-idx-q ttl",
                  lambda: kernel(qr_tt, cos_full_tt, sin_full_tt,
                                 start_pos_tt, wq_b_tt, H_tt, out_tt),
                  mesh)

        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
