"""Lk-C PCC test: q_rsqrt_norm + q rotary + wkv (sans wkv all_gather).

Reference covers the half of `DeviceAttention.forward_device` between
the wq_b all_gather and the wkv all_gather:
- reshape q_full to [B,S,H,D]
- per-head rsqrt-norm
- pick cos/sin via embedding(start_pos, table); reshape to [1,S,1,rd/2]
- slice q nope/rope, rotary on rope half, concat
- ttnn.matmul(x, wkv) — partial pre-all_gather

The wq_a/wq_b path that produces q_full / qr is upstream (Lk-A/B).
For this test we feed in random q_full and x directly.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    _device_apply_rotary_interleaved, _device_q_rsqrt_norm,
)


DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B, S = 1, 1


def make_lk_c_kernel():
    """Placeholder mega kernel for Lk-C.

    Inputs:
      q_full:       [1, 1, n_heads*head_dim] bf16  — post-wq_b-all_gather
      x:            [1, 1, dim] bf16               — residual (replicated)
      cos_full:     [max_seq_len, rope_head_dim/2] bf16
      sin_full:     [max_seq_len, rope_head_dim/2] bf16
      start_pos:    [1, 1] uint32
      wkv_w:        [dim, head_dim] bf16
    Outputs:
      q_out:        [1, 1, n_heads, head_dim] bf16  — rotated/normed q
      wkv_out:      [1, 1, head_dim] bf16            — pre-wkv-all_gather
    """
    @ttl.operation(grid="auto")
    def lk_c_kernel(q_full, x, cos_full, sin_full, start_pos, wkv_w,
                    q_out, wkv_out):
        @ttl.compute()
        def compute():
            pass

    return lk_c_kernel


def reference(mesh, q_full_tt, x_tt, cos_full_tt, sin_full_tt,
              start_pos_tt, wkv_w_tt):
    # q-stack tail: reshape, q_rsqrt_norm, pick cos/sin, slice nope/rope,
    # rotary on rope, concat. (Mirror of DeviceAttention.forward_device,
    # the attn.q phase after wq_b matmul.)
    q_tt = ttnn.reshape(q_full_tt, [B, S, N_HEADS, HEAD_DIM])
    q_tt = _device_q_rsqrt_norm(ttnn, q_tt, NORM_EPS)
    rd_half = ROPE_HEAD_DIM // 2
    cos = ttnn.embedding(start_pos_tt, cos_full_tt, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(start_pos_tt, sin_full_tt, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, S, 1, rd_half])
    sin = ttnn.reshape(sin, [1, S, 1, rd_half])
    q_nope = ttnn.slice(q_tt, [0, 0, 0, 0], [B, S, N_HEADS, HEAD_DIM - ROPE_HEAD_DIM])
    q_rope = ttnn.slice(q_tt, [0, 0, 0, HEAD_DIM - ROPE_HEAD_DIM], [B, S, N_HEADS, HEAD_DIM])
    q_rope = _device_apply_rotary_interleaved(ttnn, q_rope, cos, sin, inverse=False)
    q_tt = ttnn.concat([q_nope, q_rope], dim=-1)

    # wkv matmul (no all_gather; weight replicated).
    wkv_out_tt = ttnn.matmul(x_tt, wkv_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return q_tt, wkv_out_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        q_full = torch.randn(1, 1, N_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.1
        x = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        cos_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        sin_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        wkv_w = torch.randn(DIM, HEAD_DIM, dtype=torch.bfloat16) * 0.02
        start_pos = torch.tensor([[1]], dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        q_full_tt = ttnn.as_tensor(q_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep)
        cos_full_tt = ttnn.as_tensor(cos_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        sin_full_tt = ttnn.as_tensor(sin_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        wkv_w_tt = ttnn.as_tensor(wkv_w.contiguous(), dtype=ttnn.bfloat16, **rep)
        start_pos_tt = ttnn.from_torch(
            start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        ref_q_tt, ref_wkv_tt = reference(
            mesh, q_full_tt, x_tt, cos_full_tt, sin_full_tt,
            start_pos_tt, wkv_w_tt)
        ref_q_host = download_chip0(mesh, mesh_shape, ref_q_tt)
        ref_wkv_host = download_chip0(mesh, mesh_shape, ref_wkv_tt)

        kernel = make_lk_c_kernel()
        q_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, N_HEADS, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        wkv_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(q_full_tt, x_tt, cos_full_tt, sin_full_tt,
               start_pos_tt, wkv_w_tt, q_out_tt, wkv_out_tt)
        kernel_q_host = download_chip0(mesh, mesh_shape, q_out_tt)
        kernel_wkv_host = download_chip0(mesh, mesh_shape, wkv_out_tt)

        ok_q = report_pcc("Lk-C/q", ref_q_host, kernel_q_host)
        ok_kv = report_pcc("Lk-C/wkv", ref_wkv_host, kernel_wkv_host)
        sys.exit(0 if (ok_q and ok_kv) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
