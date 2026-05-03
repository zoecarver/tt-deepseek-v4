"""Lk-D-idx-q PCC test: indexer q-stack (sans indexer.wq_b all_gather).

Reference is the start of `DeviceIndexer.forward_device_score`:
- self.wq_b.forward_device(qr_tt) — modeled here as a plain matmul
- reshape to [B, 1, H, D]
- pick cos/sin via embedding(start_pos, table); reshape to [1,1,1,rd/2]
- slice q nope/rope, rotary on rope, concat
- Walsh-Hadamard rotation (matmul against H constant)

The indexer.wq_b all_gather is excluded — for the test the weight is replicated.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    _device_apply_rotary_interleaved, _device_rotate_activation,
    _sylvester_hadamard,
)


Q_LORA_RANK = 1024
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
ROPE_HEAD_DIM = 64
MAX_SEQ_LEN = 512
B, S = 1, 1


def make_lk_d_idx_q_kernel():
    """Placeholder mega kernel for Lk-D-idx-q.

    Inputs:
      qr:          [1, 1, q_lora_rank] bf16  — post-wq_a-all_gather output
      cos_full:    [max_seq_len, rope_head_dim/2] bf16
      sin_full:    [max_seq_len, rope_head_dim/2] bf16
      start_pos:   [1, 1] uint32
      indexer_wq_b:[q_lora_rank, index_n_heads*index_head_dim] bf16
      H:           [index_head_dim, index_head_dim] bf16  — Hadamard / sqrt(d)
    Output:
      q_idx:       [1, 1, index_n_heads, index_head_dim] bf16
    """
    @ttl.operation(grid="auto")
    def lk_d_idx_q_kernel(qr, cos_full, sin_full, start_pos,
                          indexer_wq_b, H, q_idx):
        @ttl.compute()
        def compute():
            pass

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
    q_tt = _device_rotate_activation(ttnn, q_tt, H_tt)
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

        kernel = make_lk_d_idx_q_kernel()
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
