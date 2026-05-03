"""Lk-B PCC test: q_norm + wq_b (sans wq_b all_gather).

Reference is the exact ttnn / ttl chain: reshape into [B*S, q_lora_rank]
→ DeviceRMSNorm.forward_device(q_norm) → reshape to [B, S, q_lora_rank]
→ ttnn.matmul(wq_b). The all_gather after the matmul is excluded.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import DeviceRMSNorm, _RMS_TILE


Q_LORA_RANK = 1024
N_HEADS = 64
HEAD_DIM = 512
N = N_HEADS * HEAD_DIM    # 32768
NORM_EPS = 1e-6
B, S = 1, 1


def make_lk_b_kernel():
    """Placeholder mega kernel for Lk-B.

    Inputs:
      q_lora:        [1, 1, q_lora_rank] bf16  — post-wq_a-all_gather
      q_norm_gamma:  [q_lora_rank] bf16
      wq_b_w:        [q_lora_rank, n_heads*head_dim] bf16
    Output:
      out:           [1, 1, n_heads*head_dim] bf16  — pre-wq_b-all_gather
    """
    @ttl.operation(grid="auto")
    def lk_b_kernel(q_lora, q_norm_gamma, wq_b_w, out):
        @ttl.compute()
        def compute():
            pass

    return lk_b_kernel


def reference(mesh, q_lora_tt, gamma_cpu, wq_b_w_tt):
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=gamma_cpu, eps=NORM_EPS)
    # Mirror DeviceAttention.forward_device's q_norm step:
    #   q_lora_2d = ttnn.reshape(q_lora, [B*S, q_lora_rank])
    #   qr_2d = self._rmsnorm_device(self.q_norm_dev, q_lora_2d, B*S)
    #   qr_tt = ttnn.reshape(qr_2d, [B, S, q_lora_rank])
    q_lora_2d = ttnn.reshape(q_lora_tt, [B * S, Q_LORA_RANK])
    # _rmsnorm_device pads M to TILE; for B*S=1, pad to _RMS_TILE.
    Mpad = -(-(B * S) // _RMS_TILE) * _RMS_TILE
    if (B * S) < Mpad:
        q_lora_2d = ttnn.pad(
            q_lora_2d,
            padding=[(0, Mpad - (B * S)), (0, 0)], value=0.0)
    qr_padded = rmsn.forward_device(q_lora_2d, B * S)
    qr_2d = ttnn.slice(qr_padded, [0, 0], [B * S, Q_LORA_RANK])
    qr_tt = ttnn.reshape(qr_2d, [B, S, Q_LORA_RANK])
    return ttnn.matmul(qr_tt, wq_b_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        q_lora = torch.randn(1, 1, Q_LORA_RANK, dtype=torch.bfloat16) * 0.1
        gamma = torch.ones(Q_LORA_RANK, dtype=torch.bfloat16)
        wq_b_w = torch.randn(Q_LORA_RANK, N, dtype=torch.bfloat16) * 0.02

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        q_lora_tt = ttnn.as_tensor(q_lora.contiguous(), dtype=ttnn.bfloat16, **rep)
        wq_b_w_tt = ttnn.as_tensor(wq_b_w.contiguous(), dtype=ttnn.bfloat16, **rep)

        ref_out_tt = reference(mesh, q_lora_tt, gamma, wq_b_w_tt)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_b_kernel()
        gamma_tt = ttnn.as_tensor(
            gamma.unsqueeze(0).expand(_RMS_TILE, -1).contiguous(),
            dtype=ttnn.bfloat16, **rep)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(q_lora_tt, gamma_tt, wq_b_w_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-B", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
