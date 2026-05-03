"""Lk-D-idx-score PCC test: indexer weights_proj + score reduce.

Reference covers the tail of `DeviceIndexer.forward_device_score`:
- ttnn.matmul(x, weights_proj) → w_tt
- ttnn.multiply(w_tt, scale)
- ttnn.transpose(kv_cache_tt, -2, -1) → kv_T
- ttnn.matmul(q_idx, kv_T) → score
- ttnn.relu, ttnn.transpose, ttnn.reshape, ttnn.multiply, ttnn.sum

Boundaries: pre-CCL is indexer.weights_proj all_gather; post is no CCL
(score is consumed locally by Lk-D-topk).
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0


DIM = 4096
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
T_PAD = 128
B = 1


def make_lk_d_idx_score_kernel():
    """Placeholder mega kernel for Lk-D-idx-score.

    Inputs:
      x:              [1, 1, dim] bf16
      weights_proj_w: [dim, index_n_heads] bf16
      q_idx:          [1, 1, index_n_heads, index_head_dim] bf16
      kv_cache:       [1, 1, T_pad, index_head_dim] bf16
      scale:          fp32 scalar (or [1] tile)
    Output:
      score:          [1, 1, T_pad] bf16
    """
    @ttl.operation(grid="auto")
    def lk_d_idx_score_kernel(x, weights_proj_w, q_idx, kv_cache, scale_tile, score_out):
        @ttl.compute()
        def compute():
            pass

    return lk_d_idx_score_kernel


def reference(mesh, x_tt, weights_proj_w_tt, q_idx_tt, kv_cache_tt, scale):
    H = INDEX_N_HEADS
    # weights_proj matmul (no all_gather).
    w_tt = ttnn.matmul(x_tt, weights_proj_w_tt,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w_tt = ttnn.multiply(w_tt, scale)
    kv_T = ttnn.transpose(kv_cache_tt, -2, -1)              # [B, 1, D, T_pad]
    score = ttnn.matmul(q_idx_tt, kv_T)                       # [B, 1, H, T_pad]
    score = ttnn.relu(score)
    score_t = ttnn.transpose(score, -2, -1)                   # [B, 1, T_pad, H]
    w_b = ttnn.reshape(w_tt, [B, 1, 1, H])
    score_t = ttnn.multiply(score_t, w_b)
    return ttnn.sum(score_t, dim=-1, keepdim=False)           # [B, 1, T_pad]


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

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep)
        wproj_tt = ttnn.as_tensor(weights_proj_w.contiguous(),
                                  dtype=ttnn.bfloat16, **rep)
        q_idx_tt = ttnn.as_tensor(q_idx.contiguous(), dtype=ttnn.bfloat16, **rep)
        kv_cache_tt = ttnn.as_tensor(kv_cache.contiguous(),
                                     dtype=ttnn.bfloat16, **rep)
        scale_tile = ttnn.as_tensor(
            torch.full((32, 32), scale, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)

        ref_out_tt = reference(mesh, x_tt, wproj_tt, q_idx_tt, kv_cache_tt, scale)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_d_idx_score_kernel()
        score_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, T_PAD, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(x_tt, wproj_tt, q_idx_tt, kv_cache_tt, scale_tile, score_out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, score_out_tt)

        ok = report_pcc("Lk-D-idx-score", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
