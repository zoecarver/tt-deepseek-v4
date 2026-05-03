"""Final mega-zone PCC test: head logits + per-chip topk(k=1).

Reference is `DeviceLMHead.forward_argmax_device` end-to-end:
- _logits_body:
    slice(a_tt, last row), matmul(x, hc_fn_t), multiply(x,x), mean, add(eps),
    rsqrt, multiply (mixes * rsqrt), multiply (* hc_scale), add (hc_base),
    sigmoid, add (hc_eps), reshape, reshape, matmul (pre, x_3d), typecast,
    reshape, ttnn.rms_norm (final norm), reshape, matmul (lm_head)
- ttnn.topk(logits, k=1) → (top_val, top_idx) per chip

Boundaries: pre-CCL is the last layer's routed-expert all_reduce (the
last `a_tt` is replicated); post is no CCL (host pulls 4 bytes).
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0


DIM = 4096
MHC = 4
D = MHC * DIM
VOCAB = 129280
NORM_EPS = 1e-6
HC_EPS = 1e-6
NUM_TOKENS = 1
NUM_TOKENS_PAD = 32


def make_final_kernel():
    """Placeholder mega kernel for the head + topk(k=1).

    Inputs:
      a:               [num_tokens_pad, D] fp32  — last layer's residual
      hc_fn_t:         [mhc*hidden, mhc] fp32   — transposed for matmul
      hc_scale:        [1, 1] fp32
      hc_base:         [1, mhc] fp32
      norm_gamma:      [hidden] bf16
      w_lmhead:        [hidden, vocab_padded] bf16  — replicated for the test
    Outputs:
      top_val:         [1, 1, 1] bf16   — per-chip top-1 value
      top_idx:         [1, 1, 1] uint16 — per-chip top-1 vocab index
    """
    @ttl.operation(grid="auto")
    def final_kernel(a, hc_fn_t, hc_scale, hc_base, norm_gamma, w_lmhead,
                     top_val, top_idx):
        @ttl.compute()
        def compute():
            pass

    return final_kernel


def reference(mesh, a_tt, hc_fn_t_tt, hc_scale_tt, hc_base_tt,
              norm_gamma_tt, w_lmhead_tt):
    """Mirror of DeviceLMHead.forward_argmax_device for a replicated w_lmhead."""
    hidden = DIM
    last = NUM_TOKENS - 1

    # _logits_body
    x_2d = ttnn.slice(a_tt, [last, 0], [last + 1, D])
    mixes = ttnn.matmul(x_2d, hc_fn_t_tt)
    sq = ttnn.multiply(x_2d, x_2d)
    sq_mean = ttnn.mean(sq, dim=-1, keepdim=True)
    sq_mean_eps = ttnn.add(sq_mean, NORM_EPS)
    rsqrt_val = ttnn.rsqrt(sq_mean_eps)
    scaled = ttnn.multiply(mixes, rsqrt_val)
    scaled = ttnn.multiply(scaled, hc_scale_tt)
    scaled = ttnn.add(scaled, hc_base_tt)
    pre = ttnn.sigmoid(scaled)
    pre = ttnn.add(pre, HC_EPS)
    x_3d = ttnn.reshape(x_2d, [1, MHC, hidden])
    pre_3d = ttnn.reshape(pre, [1, 1, MHC])
    y_3d = ttnn.matmul(pre_3d, x_3d)
    y_bf16 = ttnn.typecast(y_3d, dtype=ttnn.bfloat16)
    y_4d = ttnn.reshape(y_bf16, (1, 1, 1, hidden))
    y_normed = ttnn.rms_norm(y_4d, weight=norm_gamma_tt, epsilon=NORM_EPS)
    y_normed = ttnn.reshape(y_normed, (1, 1, hidden))
    logits_tt = ttnn.matmul(y_normed, w_lmhead_tt,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Per-chip top-1.
    vals_tt, idxs_tt = ttnn.topk(
        logits_tt, k=1, dim=-1, largest=True, sorted=True)
    return vals_tt, idxs_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        a = torch.randn(NUM_TOKENS_PAD, D, dtype=torch.float32) * 0.1
        # hc_fn_t: [mhc*hidden, mhc] (transposed for the matmul).
        hc_fn = torch.randn(MHC, MHC * DIM, dtype=torch.float32) * 0.05
        hc_fn_t = hc_fn.transpose(0, 1).contiguous()
        hc_scale = torch.randn(1, 1, dtype=torch.float32)
        hc_base = torch.randn(1, MHC, dtype=torch.float32) * 0.01
        norm_gamma = torch.ones(DIM, dtype=torch.bfloat16)
        # vocab kept replicated for the test (real path shards across 32 chips).
        w_lmhead = torch.randn(DIM, VOCAB, dtype=torch.bfloat16) * 0.005

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        a_tt = ttnn.as_tensor(a.contiguous(), dtype=ttnn.float32, **rep)
        hc_fn_t_tt = ttnn.as_tensor(hc_fn_t, dtype=ttnn.float32, **rep)
        hc_scale_tt = ttnn.as_tensor(hc_scale, dtype=ttnn.float32, **rep)
        hc_base_tt = ttnn.as_tensor(hc_base, dtype=ttnn.float32, **rep)
        norm_gamma_tt = ttnn.as_tensor(
            norm_gamma.contiguous(), dtype=ttnn.bfloat16, **rep)
        w_lmhead_tt = ttnn.as_tensor(w_lmhead.contiguous(),
                                     dtype=ttnn.bfloat16, **rep)

        ref_vals_tt, ref_idxs_tt = reference(
            mesh, a_tt, hc_fn_t_tt, hc_scale_tt, hc_base_tt,
            norm_gamma_tt, w_lmhead_tt)
        ref_vals_host = download_chip0(mesh, mesh_shape, ref_vals_tt)
        ref_idxs_host = download_chip0(mesh, mesh_shape, ref_idxs_tt)

        kernel = make_final_kernel()
        top_val_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, 1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        top_idx_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, 1, dtype=torch.int32),
            dtype=ttnn.uint16, **rep)
        kernel(a_tt, hc_fn_t_tt, hc_scale_tt, hc_base_tt,
               norm_gamma_tt, w_lmhead_tt,
               top_val_out_tt, top_idx_out_tt)
        kernel_vals_host = download_chip0(mesh, mesh_shape, top_val_out_tt)
        kernel_idxs_host = download_chip0(mesh, mesh_shape, top_idx_out_tt)

        ok_v = report_pcc("Final/top_val", ref_vals_host, kernel_vals_host)
        ok_i = report_pcc("Final/top_idx", ref_idxs_host.float(),
                          kernel_idxs_host.float())
        sys.exit(0 if (ok_v and ok_i) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
