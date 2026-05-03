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
    _get_ttl_act_quant_block_kernel, _get_ttl_rmsnorm_kernel, _RMS_TILE,
)


HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_HEAD_DIM   # 448
ACT_QUANT_BLOCK = 64
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B, S = 1, 1
TILE = 32


def make_lk_d1_kernel():
    """Mega kernel for Lk-D1 = kv_norm → rotary(rope) → act_quant_block(nope).

    Composes two tt-lang dispatches (rmsnorm + act_quant_block) plus a
    ttnn-based rotary helper, mirroring DeviceAttention.forward_device's
    attn.kv stage exactly. Wrapper handles all the [1,1,D] ↔ [TILE,D]
    pad/slice glue around the kernels.
    """
    rms_kernel = _get_ttl_rmsnorm_kernel(
        num_row_tiles=1, h_tiles=HEAD_DIM // TILE,
        rms_eps=NORM_EPS, inv_D=1.0 / HEAD_DIM)
    act_kernel = _get_ttl_act_quant_block_kernel(
        M_pad=TILE, N=NOPE_DIM, BLOCK=ACT_QUANT_BLOCK)

    state: dict = {}

    def _alloc_replicated_zeros(mesh, shape):
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def lk_d1_kernel(kv, kv_norm_gamma, cos_full, sin_full, start_pos, kv_out):
        if "scratch" not in state:
            mesh = kv.device()
            state["scaler_tt"] = ttnn.from_torch(
                torch.ones((TILE, TILE), dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state["normed_tt"] = _alloc_replicated_zeros(mesh, (TILE, HEAD_DIM))
            state["nope_quant_tt"] = _alloc_replicated_zeros(mesh, (TILE, NOPE_DIM))
            state["scratch"] = True

        rd_half = ROPE_HEAD_DIM // 2

        # kv_norm: pad → rmsnorm → slice/reshape.
        kv_2d = ttnn.reshape(kv, [B * S, HEAD_DIM])
        kv_padded = ttnn.pad(
            kv_2d, padding=[(0, TILE - B * S), (0, 0)], value=0.0)
        rms_kernel(kv_padded, kv_norm_gamma, state["scaler_tt"], state["normed_tt"])
        kv_normed_2d = ttnn.slice(state["normed_tt"], [0, 0], [B * S, HEAD_DIM])
        kv_normed = ttnn.reshape(kv_normed_2d, [B, S, HEAD_DIM])

        # Rotary on rope-half via the existing ttnn-based helper.
        cos = ttnn.embedding(start_pos, cos_full, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(start_pos, sin_full, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, S, rd_half])
        sin = ttnn.reshape(sin, [1, S, rd_half])
        kv_nope = ttnn.slice(kv_normed, [0, 0, 0], [B, S, NOPE_DIM])
        kv_rope = ttnn.slice(kv_normed, [0, 0, NOPE_DIM], [B, S, HEAD_DIM])
        kv_rope = _device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)

        # act_quant_block on nope-half via the existing TTL kernel.
        nope_2d = ttnn.reshape(kv_nope, [B * S, NOPE_DIM])
        nope_padded = ttnn.pad(
            nope_2d, padding=[(0, TILE - B * S), (0, 0)], value=0.0)
        act_kernel(nope_padded, state["scaler_tt"], state["nope_quant_tt"])
        kv_nope_q_2d = ttnn.slice(state["nope_quant_tt"], [0, 0], [B * S, NOPE_DIM])
        kv_nope_q = ttnn.reshape(kv_nope_q_2d, [B, S, NOPE_DIM])

        # Concat nope_q + rope and copy into the test-provided out.
        merged = ttnn.concat([kv_nope_q, kv_rope], dim=-1)
        ttnn.copy(merged, kv_out)

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
