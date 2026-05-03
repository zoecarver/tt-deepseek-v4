"""L0 PCC test: embed_prep + first hc_pre + attn_norm + wq_a (layer-0 only).

Reference is the layer-0 entry sequence: starting from `embed_tt` shape
`[1, 1, dim]` bf16 (post-embedding-all_gather), build the layer-0
`a_tt` residual stream the way `Transformer._decode_blocks_body` does
it (typecast → repeat → pad), then run the same hc_pre + attn_norm +
wq_a chain as Lk-A. The wq_a all_gather is excluded.

Placeholder mega kernel has empty compute; PCC is expected to fail
today and pass once the real kernel is implemented.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import DeviceMHC, DeviceRMSNorm, _MHC_TILE


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


def make_l0_kernel():
    """Placeholder mega kernel for L0.

    Inputs:
      embed:           [1, 1, dim] bf16  — post-embed-all_gather output
      hc_fn_packed:    [D, _MHC_TILE] fp32
      hc_scale:        [3] fp32
      hc_base:         [mhc_mult3] fp32
      attn_norm_gamma: [hidden] bf16
      wq_a_w:          [hidden, q_lora_rank] bf16
    Outputs:
      a_next:          [num_tokens_pad, D] fp32 — residual stream for next layer's hc_pre
      wq_a_out:        [1, 1, q_lora_rank] bf16 — wq_a partial pre-all_gather
    """
    @ttl.operation(grid="auto")
    def l0_kernel(embed, hc_fn_packed, hc_scale, hc_base,
                  attn_norm_gamma, wq_a_w, a_next, wq_a_out):
        @ttl.compute()
        def compute():
            pass

    return l0_kernel


def reference(mesh, embed_tt, hc_fn_cpu, hc_scale_cpu, hc_base_cpu,
              gamma_cpu, wq_a_w_tt):
    """Mirror of `_decode_blocks_body` embed-prep + layer-0 hc_pre + attn_norm + wq_a."""
    # embed-prep — from Transformer._decode_blocks_body.
    e_2d = ttnn.reshape(embed_tt, [NUM_TOKENS, DIM])
    e_fp32 = ttnn.typecast(e_2d, dtype=ttnn.float32)
    e_repeated = ttnn.repeat(e_fp32, ttnn.Shape([1, MHC]))
    a_tt = ttnn.pad(
        e_repeated,
        padding=[(0, NUM_TOKENS_PAD - NUM_TOKENS), (0, 0)],
        value=0.0,
    )

    mhc = DeviceMHC(
        mesh=mesh,
        hc_fn=hc_fn_cpu, hc_scale=hc_scale_cpu, hc_base=hc_base_cpu,
        hc_mult=MHC, hc_eps=HC_EPS,
        sinkhorn_iters=HC_SINKHORN_ITERS, norm_eps=NORM_EPS,
    )
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=gamma_cpu, eps=NORM_EPS)

    hc_out_fp32 = mhc.hc_pre_device(NUM_TOKENS, NUM_TOKENS_PAD, a_tt=a_tt)
    ttnn.typecast(hc_out_fp32, dtype=ttnn.bfloat16,
                  output_tensor=rmsn._x_upload_tt)
    norm_out_tt = rmsn.forward_device(rmsn._x_upload_tt, NUM_TOKENS)
    ttnn.slice(norm_out_tt, [0, 0], [NUM_TOKENS, DIM],
               output_tensor=mhc._norm_slice_tt)
    bridge_tt = ttnn.reshape(mhc._norm_slice_tt, [1, 1, DIM])
    wq_a_out_tt = ttnn.matmul(
        bridge_tt, wq_a_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return a_tt, wq_a_out_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        embed = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        hc_fn = torch.randn(MHC_MULT3, D, dtype=torch.float32) * 0.05
        hc_scale = torch.randn(3, dtype=torch.float32)
        hc_base = torch.randn(MHC_MULT3, dtype=torch.float32) * 0.01
        gamma = torch.ones(DIM, dtype=torch.bfloat16)
        wq_a_w = torch.randn(DIM, Q_LORA_RANK, dtype=torch.bfloat16) * 0.02

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        embed_tt = ttnn.as_tensor(embed.contiguous(), dtype=ttnn.bfloat16, **rep)
        wq_a_w_tt = ttnn.as_tensor(wq_a_w.contiguous(), dtype=ttnn.bfloat16, **rep)

        ref_a_tt, ref_wq_a_tt = reference(
            mesh, embed_tt, hc_fn, hc_scale, hc_base, gamma, wq_a_w_tt)
        ref_a_host = download_chip0(mesh, mesh_shape, ref_a_tt)
        ref_wq_a_host = download_chip0(mesh, mesh_shape, ref_wq_a_tt)

        # Placeholder kernel: build matching dummy buffers, dispatch.
        kernel = make_l0_kernel()
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
        a_next_tt = ttnn.from_torch(
            torch.zeros(NUM_TOKENS_PAD, D, dtype=torch.float32),
            dtype=ttnn.float32, **rep)
        wq_a_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, Q_LORA_RANK, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(embed_tt, hc_fn_packed_tt, hc_scale_tt, hc_base_tt,
               gamma_tt, wq_a_w_tt, a_next_tt, wq_a_out_tt)

        kernel_a_host = download_chip0(mesh, mesh_shape, a_next_tt)
        kernel_wq_a_host = download_chip0(mesh, mesh_shape, wq_a_out_tt)

        ok_a = report_pcc("L0/a_next", ref_a_host, kernel_a_host)
        ok_q = report_pcc("L0/wq_a", ref_wq_a_host, kernel_wq_a_host)
        sys.exit(0 if (ok_a and ok_q) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
