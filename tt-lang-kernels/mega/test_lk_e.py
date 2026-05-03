"""Lk-E PCC test: hc_post_attn + hc_pre_ffn + ffn_norm + shared expert (sans all_reduce).

Reference covers everything from `wo_b all_gather` (the attn output is now
replicated across the mesh) up to the shared-expert all_reduce. Mirrors
the relevant slice of `_block_forward` plus `DeviceSharedExpert._compute_body`:

- block.hc_post (attn-side):
    reshape, repeat (mhc), pad, reshape, typecast(fp32),
    DeviceMHC.hc_post_device → tt-lang post kernel + ttnn glue
    _mhc_post_to_a_tt (slice/reshape/pad)
- block.hc_pre (ffn-side):
    DeviceMHC.hc_pre_device on the ffn-side instance
- block.norm (ffn):
    typecast(fp32→bf16), DeviceRMSNorm.forward_device, slice, reshape
- shared expert:
    matmul(x, w1), matmul(x, w3), clamp ×2, silu, multiply, matmul(mid, w2)
    → partial (pre-all_reduce)

Boundaries: pre-CCL is wo_b all_gather; post-CCL is shared-expert all_reduce.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    DeviceMHC, DeviceRMSNorm, _MHC_TILE, _mhc_post_to_a_tt,
)


DIM = 4096
MHC = 4
MHC_MULT3 = MHC * 2 + MHC * MHC
D = MHC * DIM
INTER_DIM = 2048
NORM_EPS = 1e-6
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20
SWIGLU_LIMIT = 10.0
NUM_TOKENS = 1
NUM_TOKENS_PAD = _MHC_TILE
B, S = 1, 1


def make_lk_e_kernel():
    """Placeholder mega kernel for Lk-E.

    Inputs:
      attn_out:       [1, 1, dim] bf16  — post-wo_b-all_gather attn output
      prev_a:         [num_tokens_pad, D] fp32  — residual (pre-attn)
      hc_attn_post_mix, hc_attn_comb_sk:  the stash from this layer's hc_pre_attn
      hc_ffn_fn_packed:[D, _MHC_TILE] fp32
      hc_ffn_scale:   [3] fp32
      hc_ffn_base:    [mhc_mult3] fp32
      ffn_norm_gamma: [hidden] bf16
      w1, w2, w3:     [inter, dim], [dim, inter], [inter, dim] bf16
    Outputs:
      shared_partial: [1, dim] bf16  — pre-all_reduce shared expert partial
      next_a:         [num_tokens_pad, D] fp32  — input to next layer's hc_pre_attn
    """
    @ttl.operation(grid="auto")
    def lk_e_kernel(
        attn_out, prev_a,
        hc_attn_post_mix, hc_attn_comb_sk,
        hc_ffn_fn_packed, hc_ffn_scale, hc_ffn_base,
        ffn_norm_gamma, w1, w2, w3,
        shared_partial, next_a,
    ):
        @ttl.compute()
        def compute():
            pass

    return lk_e_kernel


def reference(mesh, attn_out_tt, prev_a_tt,
              hc_attn_fn_cpu, hc_attn_scale_cpu, hc_attn_base_cpu,
              hc_ffn_fn_cpu, hc_ffn_scale_cpu, hc_ffn_base_cpu,
              attn_norm_gamma_cpu, ffn_norm_gamma_cpu,
              w1_tt, w2_tt, w3_tt):
    """Mirror of _block_forward's tail (attn-side hc_post + ffn-side hc_pre +
    ffn_norm) plus DeviceSharedExpert._compute_body, sans the all_reduce."""
    # We need an attn-side DeviceMHC instance with a stash from a prior
    # hc_pre_attn call so that hc_post_device has something to consume.
    mhc_attn = DeviceMHC(
        mesh=mesh, hc_fn=hc_attn_fn_cpu, hc_scale=hc_attn_scale_cpu,
        hc_base=hc_attn_base_cpu,
        hc_mult=MHC, hc_eps=HC_EPS,
        sinkhorn_iters=HC_SINKHORN_ITERS, norm_eps=NORM_EPS,
    )
    mhc_ffn = DeviceMHC(
        mesh=mesh, hc_fn=hc_ffn_fn_cpu, hc_scale=hc_ffn_scale_cpu,
        hc_base=hc_ffn_base_cpu,
        hc_mult=MHC, hc_eps=HC_EPS,
        sinkhorn_iters=HC_SINKHORN_ITERS, norm_eps=NORM_EPS,
    )
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=ffn_norm_gamma_cpu, eps=NORM_EPS)

    # Run hc_pre_attn first to populate the stash (post / comb_sk).
    mhc_attn.hc_pre_device(NUM_TOKENS, NUM_TOKENS_PAD, a_tt=prev_a_tt)

    # block.hc_post (attn-side): mirror of _block_forward.
    x_2d = ttnn.reshape(attn_out_tt, [NUM_TOKENS, 1, DIM])
    x_repeated = ttnn.repeat(x_2d, ttnn.Shape([1, MHC, 1]))
    x_padded = ttnn.pad(
        x_repeated,
        padding=[(0, 0), (0, _MHC_TILE - MHC), (0, 0)], value=0.0)
    x_post_input = ttnn.reshape(x_padded, [NUM_TOKENS * _MHC_TILE, DIM])
    ttnn.typecast(x_post_input, dtype=ttnn.float32,
                  output_tensor=mhc_attn._x_upload_tt)
    post_out_tt = mhc_attn.hc_post_device(NUM_TOKENS)

    # block.hc_pre (ffn-side).
    a_input_tt = _mhc_post_to_a_tt(
        ttnn, post_out_tt, NUM_TOKENS, NUM_TOKENS_PAD, MHC, DIM)
    ffn_hc_out_fp32 = mhc_ffn.hc_pre_device(
        NUM_TOKENS, NUM_TOKENS_PAD, a_tt=a_input_tt)

    # block.norm (ffn).
    ttnn.typecast(ffn_hc_out_fp32, dtype=ttnn.bfloat16,
                  output_tensor=rmsn._x_upload_tt)
    ffn_norm_out_tt = rmsn.forward_device(rmsn._x_upload_tt, NUM_TOKENS)
    ttnn.slice(ffn_norm_out_tt, [0, 0], [NUM_TOKENS, DIM],
               output_tensor=mhc_ffn._norm_slice_tt)
    x_ffn_tt = mhc_ffn._norm_slice_tt   # [NUM_TOKENS, hidden] bf16

    # Shared expert SwiGLU (no all_reduce).
    x_3d = ttnn.reshape(x_ffn_tt, [1, NUM_TOKENS, DIM])
    y1 = ttnn.matmul(x_3d, w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y3 = ttnn.matmul(x_3d, w3_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if SWIGLU_LIMIT > 0:
        y1 = ttnn.clamp(y1, max=SWIGLU_LIMIT)
        y3 = ttnn.clamp(y3, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    silu = ttnn.silu(y1)
    mid = ttnn.multiply(silu, y3)
    partial = ttnn.matmul(mid, w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return partial, a_input_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        attn_out = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        prev_a = torch.randn(NUM_TOKENS_PAD, D, dtype=torch.float32) * 0.1
        hc_attn_fn = torch.randn(MHC_MULT3, D, dtype=torch.float32) * 0.05
        hc_attn_scale = torch.randn(3, dtype=torch.float32)
        hc_attn_base = torch.randn(MHC_MULT3, dtype=torch.float32) * 0.01
        hc_ffn_fn = torch.randn(MHC_MULT3, D, dtype=torch.float32) * 0.05
        hc_ffn_scale = torch.randn(3, dtype=torch.float32)
        hc_ffn_base = torch.randn(MHC_MULT3, dtype=torch.float32) * 0.01
        attn_norm_gamma = torch.ones(DIM, dtype=torch.bfloat16)
        ffn_norm_gamma = torch.ones(DIM, dtype=torch.bfloat16)
        # Shared expert weights stored as [out, in] (matches inference convention).
        w1 = torch.randn(INTER_DIM, DIM, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(DIM, INTER_DIM, dtype=torch.bfloat16) * 0.02
        w3 = torch.randn(INTER_DIM, DIM, dtype=torch.bfloat16) * 0.02
        # ttnn matmul wants [in, out].
        w1_kn = w1.transpose(0, 1).contiguous()
        w3_kn = w3.transpose(0, 1).contiguous()
        w2_kn = w2.transpose(0, 1).contiguous()

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        attn_out_tt = ttnn.as_tensor(attn_out.contiguous(), dtype=ttnn.bfloat16, **rep)
        prev_a_tt = ttnn.as_tensor(prev_a.contiguous(), dtype=ttnn.float32, **rep)
        w1_tt = ttnn.as_tensor(w1_kn, dtype=ttnn.bfloat16, **rep)
        w2_tt = ttnn.as_tensor(w2_kn, dtype=ttnn.bfloat16, **rep)
        w3_tt = ttnn.as_tensor(w3_kn, dtype=ttnn.bfloat16, **rep)

        ref_partial_tt, ref_next_a_tt = reference(
            mesh, attn_out_tt, prev_a_tt,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            attn_norm_gamma, ffn_norm_gamma,
            w1_tt, w2_tt, w3_tt)
        ref_partial_host = download_chip0(mesh, mesh_shape, ref_partial_tt)
        ref_next_a_host = download_chip0(mesh, mesh_shape, ref_next_a_tt)

        kernel = make_lk_e_kernel()
        # Build matching dummy buffers for the placeholder dispatch.
        mhc_post_mix_tt = ttnn.as_tensor(
            torch.zeros(NUM_TOKENS_PAD, _MHC_TILE, dtype=torch.float32),
            dtype=ttnn.float32, **rep)
        mhc_comb_sk_tt = ttnn.as_tensor(
            torch.zeros(NUM_TOKENS * _MHC_TILE, _MHC_TILE, dtype=torch.float32),
            dtype=ttnn.float32, **rep)
        hc_ffn_fn_packed_tt = ttnn.as_tensor(
            torch.zeros(D, _MHC_TILE, dtype=torch.float32),
            dtype=ttnn.float32, **rep)
        hc_ffn_scale_tt = ttnn.as_tensor(
            hc_ffn_scale.reshape(1, 3).contiguous(), dtype=ttnn.float32, **rep)
        hc_ffn_base_tt = ttnn.as_tensor(
            hc_ffn_base.reshape(1, MHC_MULT3).contiguous(),
            dtype=ttnn.float32, **rep)
        ffn_norm_gamma_tt = ttnn.as_tensor(
            ffn_norm_gamma.unsqueeze(0).expand(_MHC_TILE, -1).contiguous(),
            dtype=ttnn.bfloat16, **rep)
        partial_out_tt = ttnn.from_torch(
            torch.zeros(1, NUM_TOKENS, DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        next_a_tt = ttnn.from_torch(
            torch.zeros(NUM_TOKENS_PAD, D, dtype=torch.float32),
            dtype=ttnn.float32, **rep)
        kernel(
            attn_out_tt, prev_a_tt,
            mhc_post_mix_tt, mhc_comb_sk_tt,
            hc_ffn_fn_packed_tt, hc_ffn_scale_tt, hc_ffn_base_tt,
            ffn_norm_gamma_tt, w1_tt, w2_tt, w3_tt,
            partial_out_tt, next_a_tt,
        )
        kernel_partial_host = download_chip0(mesh, mesh_shape, partial_out_tt)
        kernel_next_a_host = download_chip0(mesh, mesh_shape, next_a_tt)

        ok_p = report_pcc("Lk-E/shared", ref_partial_host, kernel_partial_host)
        ok_a = report_pcc("Lk-E/next_a", ref_next_a_host, kernel_next_a_host)
        sys.exit(0 if (ok_p and ok_a) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
