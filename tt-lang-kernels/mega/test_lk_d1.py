"""Lk-D1 PCC test: kv_norm + kv rotary + act_quant_block on nope half.

Reference is the attn.kv stage in DeviceAttention.forward_device:
- reshape kv → [B*S, head_dim]
- DeviceRMSNorm.forward_device(kv_norm)
- reshape → [B, S, head_dim]
- pick cos/sin via embedding(start_pos, table); reshape to [1, S, rd/2]
- slice nope/rope, rotary on rope, concat
- act_quant_block on nope half via the existing TTL kernel
- concat back

Boundaries: pre-CCL is wkv all_gather; post-CCL is whichever indexer/
compressor matmul comes next (or Lk-Dsparse's gather if no compressor).
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    DeviceRMSNorm, _device_apply_rotary_interleaved,
    _get_ttl_act_quant_block_kernel, _RMS_TILE,
)


HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_HEAD_DIM   # 448
ACT_QUANT_BLOCK = 64
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B, S = 1, 1


def make_lk_d1_kernel():
    """Placeholder mega kernel for Lk-D1.

    Inputs:
      kv:           [1, 1, head_dim] bf16  — post-wkv-all_gather
      kv_norm_gamma:[head_dim] bf16
      cos_full:     [max_seq_len, rope_head_dim/2] bf16
      sin_full:     [max_seq_len, rope_head_dim/2] bf16
      start_pos:    [1, 1] uint32
    Output:
      kv_out:       [1, 1, head_dim] bf16  — normed + rotated + nope-quantized
    """
    @ttl.operation(grid="auto")
    def lk_d1_kernel(kv, kv_norm_gamma, cos_full, sin_full, start_pos, kv_out):
        @ttl.compute()
        def compute():
            pass

    return lk_d1_kernel


def reference(mesh, kv_tt, gamma_cpu, cos_full_tt, sin_full_tt, start_pos_tt):
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=gamma_cpu, eps=NORM_EPS)

    # Mirror DeviceAttention.forward_device attn.kv:
    kv_2d = ttnn.reshape(kv_tt, [B * S, HEAD_DIM])
    Mpad = -(-(B * S) // _RMS_TILE) * _RMS_TILE
    if (B * S) < Mpad:
        kv_2d = ttnn.pad(kv_2d, padding=[(0, Mpad - (B * S)), (0, 0)], value=0.0)
    kv_padded = rmsn.forward_device(kv_2d, B * S)
    kv_2d = ttnn.slice(kv_padded, [0, 0], [B * S, HEAD_DIM])
    kv_tt = ttnn.reshape(kv_2d, [B, S, HEAD_DIM])

    rd_half = ROPE_HEAD_DIM // 2
    cos = ttnn.embedding(start_pos_tt, cos_full_tt, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(start_pos_tt, sin_full_tt, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, S, rd_half])
    sin = ttnn.reshape(sin, [1, S, rd_half])
    kv_nope = ttnn.slice(kv_tt, [0, 0, 0], [B, S, NOPE_DIM])
    kv_rope = ttnn.slice(kv_tt, [0, 0, NOPE_DIM], [B, S, HEAD_DIM])
    kv_rope = _device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)
    kv_tt = ttnn.concat([kv_nope, kv_rope], dim=-1)

    # act_quant_block on nope half via the existing TTL kernel.
    act_kernel = _get_ttl_act_quant_block_kernel(_RMS_TILE, NOPE_DIM, ACT_QUANT_BLOCK)
    nope_2d = ttnn.reshape(
        ttnn.slice(kv_tt, [0, 0, 0], [B, S, NOPE_DIM]), [B * S, NOPE_DIM])
    act_quant_out = ttnn.zeros(
        shape=(_RMS_TILE, NOPE_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    act_quant_sc = ttnn.as_tensor(
        torch.ones(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16),
        device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    act_kernel(nope_2d, act_quant_sc, act_quant_out)
    kv_nope_q = ttnn.reshape(
        ttnn.slice(act_quant_out, [0, 0], [B * S, NOPE_DIM]),
        [B, S, NOPE_DIM])
    kv_rope_only = ttnn.slice(kv_tt, [0, 0, NOPE_DIM], [B, S, HEAD_DIM])
    return ttnn.concat([kv_nope_q, kv_rope_only], dim=-1)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        kv = torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        gamma = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
        cos_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        sin_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        start_pos = torch.tensor([[1]], dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        kv_tt = ttnn.as_tensor(kv.contiguous(), dtype=ttnn.bfloat16, **rep)
        cos_full_tt = ttnn.as_tensor(cos_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        sin_full_tt = ttnn.as_tensor(sin_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        start_pos_tt = ttnn.from_torch(
            start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        ref_out_tt = reference(mesh, kv_tt, gamma, cos_full_tt, sin_full_tt, start_pos_tt)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_d1_kernel()
        gamma_tt = ttnn.as_tensor(
            gamma.unsqueeze(0).expand(_RMS_TILE, -1).contiguous(),
            dtype=ttnn.bfloat16, **rep)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(kv_tt, gamma_tt, cos_full_tt, sin_full_tt, start_pos_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-D1", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
