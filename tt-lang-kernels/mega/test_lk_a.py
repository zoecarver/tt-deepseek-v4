"""Lk-A PCC test: hc_pre_attn + attn_norm + wq_a (sans wq_a all_gather).

Reference is the exact ttnn / ttl op chain `inference.py` runs between
the previous layer's `wo_b all_gather` (or the embed all_gather for
layer 0) and the wq_a all_gather. Specifically: `DeviceMHC.hc_pre_device`
+ ttnn.typecast + `DeviceRMSNorm.forward_device` + ttnn.slice + reshape +
`ttnn.matmul(wq_a)`. The all_gather after the matmul is excluded.

Placeholder mega kernel has empty compute; PCC is expected to fail
today and pass once the real kernel is implemented.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401  (sets up sys.path for `inference` import)
from _refs import (open_mesh, close_mesh, report_pcc, download_chip0,
                   DEFAULT_MESH_SHAPE)

from inference import DeviceMHC, DeviceRMSNorm, _MHC_TILE


# V4-Flash decode shape.
DIM = 4096
MHC = 4
MHC_MULT3 = MHC * 2 + MHC * MHC                # 24
D = MHC * DIM                                   # 16384
Q_LORA_RANK = 1024
NORM_EPS = 1e-6
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20
NUM_TOKENS = 1
NUM_TOKENS_PAD = _MHC_TILE                      # 32


def make_lk_a_kernel():
    """Placeholder mega kernel for Lk-A.

    Inputs (logical shapes; format is whatever the eventual kernel needs):
      a:                [num_tokens_pad, D] fp32    — residual stream
      hc_fn_packed:     [D, _MHC_TILE]    fp32      — _mhc_pack_fn output
      hc_scale:         [3]               fp32
      hc_base:          [mhc_mult3]       fp32
      attn_norm_gamma:  [hidden]          bf16
      wq_a_w:           [hidden, q_lora_rank] bf16
    Output:
      out:              [1, 1, q_lora_rank] bf16    — pre-all_gather wq_a partial
    Empty compute → PCC fails today.
    """
    @ttl.operation(grid="auto")
    def lk_a_kernel(a, hc_fn_packed, hc_scale, hc_base,
                    attn_norm_gamma, wq_a_w, out):
        @ttl.compute()
        def compute():
            pass

    return lk_a_kernel


def reference(mesh, a_tt, hc_fn_cpu, hc_scale_cpu, hc_base_cpu,
              gamma_cpu, wq_a_w_tt):
    """Run the exact ttnn / ttl op chain from inference.py."""
    mhc = DeviceMHC(
        mesh=mesh,
        hc_fn=hc_fn_cpu, hc_scale=hc_scale_cpu, hc_base=hc_base_cpu,
        hc_mult=MHC, hc_eps=HC_EPS,
        sinkhorn_iters=HC_SINKHORN_ITERS, norm_eps=NORM_EPS,
    )
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=gamma_cpu, eps=NORM_EPS)

    # block.hc_pre — DeviceMHC.hc_pre_device.
    hc_out_fp32 = mhc.hc_pre_device(NUM_TOKENS, NUM_TOKENS_PAD, a_tt=a_tt)
    # block.norm — typecast then rmsnorm via the tt-lang kernel.
    ttnn.typecast(hc_out_fp32, dtype=ttnn.bfloat16,
                  output_tensor=rmsn._x_upload_tt)
    norm_out_tt = rmsn.forward_device(rmsn._x_upload_tt, NUM_TOKENS)
    # bridge into [B, S, hidden].
    ttnn.slice(norm_out_tt, [0, 0], [NUM_TOKENS, DIM],
               output_tensor=mhc._norm_slice_tt)
    bridge_tt = ttnn.reshape(mhc._norm_slice_tt, [1, 1, DIM])
    # wq_a matmul (no all_gather; weight is replicated for the test).
    return ttnn.matmul(bridge_tt, wq_a_w_tt,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        # Random inputs at decode shape.
        a = torch.randn(NUM_TOKENS_PAD, D, dtype=torch.float32) * 0.1
        hc_fn = torch.randn(MHC_MULT3, D, dtype=torch.float32) * 0.05
        hc_scale = torch.randn(3, dtype=torch.float32)
        hc_base = torch.randn(MHC_MULT3, dtype=torch.float32) * 0.01
        gamma = torch.ones(DIM, dtype=torch.bfloat16)
        wq_a_w = torch.randn(DIM, Q_LORA_RANK, dtype=torch.bfloat16) * 0.02

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        a_tt = ttnn.as_tensor(a.contiguous(), dtype=ttnn.float32, **rep)
        wq_a_w_tt = ttnn.as_tensor(wq_a_w.contiguous(), dtype=ttnn.bfloat16, **rep)

        # Reference: exact ttnn/ttl chain from inference.py.
        ref_out_tt = reference(mesh, a_tt, hc_fn, hc_scale, hc_base,
                               gamma, wq_a_w_tt)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        # Placeholder kernel — set up matching input/output buffers and
        # dispatch. Compute is empty; output stays zero.
        kernel = make_lk_a_kernel()
        hc_fn_packed_tt = ttnn.as_tensor(
            torch.zeros(D, _MHC_TILE, dtype=torch.float32),
            dtype=ttnn.float32, **rep)
        hc_scale_tt = ttnn.as_tensor(
            hc_scale.reshape(1, 3).contiguous(), dtype=ttnn.float32, **rep)
        hc_base_tt = ttnn.as_tensor(
            hc_base.reshape(1, MHC_MULT3).contiguous(),
            dtype=ttnn.float32, **rep)
        gamma_tt = ttnn.as_tensor(
            gamma.unsqueeze(0).expand(_MHC_TILE, -1).contiguous(),
            dtype=ttnn.bfloat16, **rep)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, Q_LORA_RANK, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(a_tt, hc_fn_packed_tt, hc_scale_tt, hc_base_tt,
               gamma_tt, wq_a_w_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-A", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
