"""CPU torch references for the V4-Flash priority kernels.

Transcribed from `../TileKernels/tile_kernels/torch/mhc.py` and friends so
this directory stands on its own. Keep these in sync with the TileKernels
reference; if the upstream reference changes, update here and note why.
"""
from __future__ import annotations

import torch


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """mHC Sinkhorn normalization.

    x: [..., m, m] (operates on the last two dims). Alternates row- and
    column-normalization for `repeat` iterations. First pass is a row-softmax
    followed by a column-normalize with eps added once to break symmetry.
    """
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_split_mixes_ref(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale+base a [n0, n1, mhc_mult3] tensor and split into three outputs.

    mhc_mult3 == 2*mhc_mult + mhc_mult**2. Section layout inside the last dim:
      [:mhc_mult]          -> sigmoid() + mhc_pre_eps                  -> pre_layer_mix
      [mhc_mult:2*mhc_mult]-> sigmoid() * mhc_post_mult_value           -> post_layer_mix
      [2*mhc_mult:]        -> identity, reshaped to (mhc_mult, mhc_mult)-> comb_res_mix
    Scale is broadcast across each section (scale[0] to first, [1] to second,
    [2] to third); base is per-entry.
    """
    a, b = input_mixes.shape[:2]
    scale_full = torch.cat([
        mhc_scale[0].expand(mhc_mult),
        mhc_scale[1].expand(mhc_mult),
        mhc_scale[2].expand(mhc_mult * mhc_mult),
    ])
    scaled = input_mixes * scale_full + mhc_base
    pre_layer_mix = scaled[..., :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_layer_mix = (scaled[..., mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value).unsqueeze(-1)
    comb_res_mix = scaled[..., 2 * mhc_mult :].reshape(a, b, mhc_mult, mhc_mult)
    return pre_layer_mix, post_layer_mix, comb_res_mix


def mhc_pre_apply_mix_ref(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Apply a mixing weight to x: (x * mix).sum(-2).

    x: [n0, n1, mhc, h] bf16, mix: [n0, n1, mhc, 1] fp32. Returns [n0, n1, h]
    bf16. Equivalent to per-token matmul of mix^T (1, mhc) by x (mhc, h).
    """
    return (x * mix).sum(-2).bfloat16()


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post. Companion to pre-big-fuse on the output side.

    x: [n0, n1, h] bf16 — attention/FFN output for the token.
    residual: [n0, n1, mhc, h] bf16 — the mhc residual streams to merge back.
    post_layer_mix: [n0, n1, mhc, 1] fp32 — per-stream scaling of x.
    comb_res_mix: [n0, n1, mhc, mhc] fp32 — sinkhorn-normalized mixing matrix.
    Output: [n0, n1, mhc, h] bf16.

    out[n, h] = x[h] * post_layer_mix[n]
              + sum_m(comb_res_mix[m, n] * residual[m, h])
    """
    term2 = torch.einsum("abmn,abmc->abnc", comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def mhc_pre_norm_fn_ref(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    """RMSNorm over flattened residual, then projected by mhc_fn^T.

    residual: [n0, n1, mhc_mult, hidden] bf16. Flattens to [num_tokens, D] where
    D = mhc_mult*hidden; RMS is per-row. mhc_fn: [mhc_mult3, D] fp32. Output:
    [n0, n1, mhc_mult3] fp32. If mhc_norm_weight is given, it's folded into
    mhc_fn before the projection.
    """
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    res_flat = residual.flatten(-2, -1).float()
    D = res_flat.shape[-1]
    sqrsum = res_flat.square().sum(-1, keepdim=True)
    inv_rms = (sqrsum / D + mhc_norm_eps).rsqrt()
    # mixes[t, m] = sum_k(residual[t, k] * fn[m, k]) * inv_rms[t]
    mixes = res_flat @ mhc_fn.T
    return mixes * inv_rms
