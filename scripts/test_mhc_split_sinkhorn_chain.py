"""PCC test: chain pre_split_mixes + sinkhorn device kernels to reproduce
`hc_split_sinkhorn` on CPU.

The two device kernels use incompatible layouts:
  pre_split_mixes: [num_tokens_pad, 32] (32 tokens per tile; cols 8:24 hold the
                   16 comb values flat).
  sinkhorn:        [num_tokens_pad*32, 32] (one tile per token; 4x4 slice at
                   top-left, PAD_SENTINEL elsewhere).

We bridge these with ttnn ROW_MAJOR ops: download the packed comb, repack to
the sinkhorn layout via torch, upload, run sinkhorn. If this works end-to-end,
we have a working decode-path integration. A custom repack kernel can replace
the CPU repack later as a perf follow-up.
"""
from __future__ import annotations

import os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
# Kernel sources are dropped next to this script under /tmp by the runner
# wrapper (see scripts/run_split_sinkhorn_chain.sh). Fall back to the in-repo
# path when running locally.
for candidate in ("/tmp", str(HERE.parent / "tt-lang-kernels")):
    if os.path.isfile(os.path.join(candidate, "pre_split_mixes.py")):
        sys.path.insert(0, candidate)
        break

import torch
import ttnn

import pre_split_mixes
import sinkhorn
from harness import (
    PAD_SENTINEL, pack_4x4_slices, unpack_4x4_slices,
    mask_tile, eps_mask_tile, scaler_tile, assert_pcc,
)


def cpu_reference(mixes, hc_scale, hc_base, mhc, post_mult, pre_eps, sinkhorn_eps, sinkhorn_iters):
    """Mirrors `hc_split_sinkhorn` in inference.py."""
    hc = mhc
    pre_mix = mixes[..., :hc]
    post_mix = mixes[..., hc:2 * hc]
    comb_mix = mixes[..., 2 * hc:].view(*mixes.shape[:-1], hc, hc)

    pre = torch.sigmoid(pre_mix * hc_scale[0] + hc_base[:hc]) + pre_eps
    post = post_mult * torch.sigmoid(post_mix * hc_scale[1] + hc_base[hc:2 * hc])
    comb = comb_mix * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)
    comb = torch.softmax(comb, dim=-1) + sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    return pre, post, comb


def run():
    torch.manual_seed(0)
    mhc = 4
    mhc_mult3 = mhc * 2 + mhc * mhc  # 24
    post_mult = 2.0
    pre_eps = 1e-2
    sinkhorn_eps = 1e-6
    sinkhorn_iters = 20

    # Decode-sized: 1 real token, padded to 32.
    num_tokens = 1
    num_tokens_pad = 32

    mixes = torch.randn(num_tokens, mhc_mult3, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(mhc_mult3, dtype=torch.float32)

    pre_ref, post_ref, comb_ref = cpu_reference(
        mixes, hc_scale, hc_base, mhc, post_mult, pre_eps, sinkhorn_eps, sinkhorn_iters,
    )

    # Pack mixes into the split_mixes input layout: [num_tokens_pad, 32].
    mixes_padded = torch.zeros(num_tokens_pad, 32, dtype=torch.float32)
    mixes_padded[:num_tokens, :mhc_mult3] = mixes

    device = ttnn.open_device(device_id=0)
    try:
        rep = dict(
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Build constants for pre_split_mixes.
        (scale_tile, base_tile, pre_mask, pre_eps_tile, post_mult_mask, comb_mask) = (
            pre_split_mixes.make_constant_tiles(
                hc_scale, hc_base, mhc, post_mult, pre_eps, dtype=torch.float32,
            )
        )

        mixes_tt = ttnn.from_torch(mixes_padded, **rep)
        pre_tt = ttnn.from_torch(torch.zeros_like(mixes_padded), **rep)
        post_tt = ttnn.from_torch(torch.zeros_like(mixes_padded), **rep)
        comb_tt = ttnn.from_torch(torch.zeros_like(mixes_padded), **rep)
        scale_tt = ttnn.from_torch(scale_tile, **rep)
        base_tt = ttnn.from_torch(base_tile, **rep)
        pre_mask_tt = ttnn.from_torch(pre_mask, **rep)
        pre_eps_tt = ttnn.from_torch(pre_eps_tile, **rep)
        post_mask_tt = ttnn.from_torch(post_mult_mask, **rep)
        comb_mask_tt = ttnn.from_torch(comb_mask, **rep)

        pre_split_mixes.solve(
            mixes_tt, scale_tt, base_tt,
            pre_mask_tt, pre_eps_tt, post_mask_tt, comb_mask_tt,
            pre_tt, post_tt, comb_tt,
        )

        # Download, validate intermediate, then re-upload in sinkhorn layout.
        pre_packed = ttnn.to_torch(pre_tt)      # [32, 32], row t cols 0:4 = pre[t]
        post_packed = ttnn.to_torch(post_tt)    # [32, 32], row t cols 4:8 = post[t]
        comb_packed = ttnn.to_torch(comb_tt)    # [32, 32], row t cols 8:24 = comb[t] flat

        pre_tt_out = pre_packed[:num_tokens, :mhc]
        post_tt_out = post_packed[:num_tokens, mhc:2 * mhc]
        comb_raw_flat = comb_packed[:num_tokens, 2 * mhc:2 * mhc + mhc * mhc]
        comb_raw = comb_raw_flat.view(num_tokens, mhc, mhc)

        print("[stage 1] pre / post / comb_raw PCC vs CPU (pre-sinkhorn)")
        pre_cpu_raw = torch.sigmoid(mixes[..., :mhc] * hc_scale[0] + hc_base[:mhc]) + pre_eps
        post_cpu_raw = post_mult * torch.sigmoid(mixes[..., mhc:2 * mhc] * hc_scale[1] + hc_base[mhc:2 * mhc])
        comb_cpu_raw = mixes[..., 2 * mhc:].view(num_tokens, mhc, mhc) * hc_scale[2] + hc_base[2 * mhc:].view(mhc, mhc)
        assert_pcc(pre_cpu_raw, pre_tt_out, threshold=0.9995)
        assert_pcc(post_cpu_raw, post_tt_out, threshold=0.9995)
        assert_pcc(comb_cpu_raw, comb_raw, threshold=0.9995)

        # Repack comb to sinkhorn layout: [num_tokens*32, 32], 4x4 at top-left
        # with PAD_SENTINEL elsewhere.
        comb_for_sinkhorn = pack_4x4_slices(comb_raw, pad_value=PAD_SENTINEL)
        comb_sinkhorn_in_tt = ttnn.from_torch(comb_for_sinkhorn, **rep)
        comb_sinkhorn_out_tt = ttnn.from_torch(torch.zeros_like(comb_for_sinkhorn), **rep)
        sk_mask_tt = ttnn.from_torch(mask_tile(valid=mhc), **rep)
        sk_em_tt = ttnn.from_torch(eps_mask_tile(eps=sinkhorn_eps, valid=mhc), **rep)
        sk_sc_tt = ttnn.from_torch(scaler_tile(), **rep)

        sinkhorn.solve(
            comb_sinkhorn_in_tt, sk_mask_tt, sk_em_tt, sk_sc_tt, comb_sinkhorn_out_tt,
            repeat=sinkhorn_iters, eps=sinkhorn_eps,
        )

        comb_sk_packed = ttnn.to_torch(comb_sinkhorn_out_tt)
        comb_tt_out = unpack_4x4_slices(comb_sk_packed, num_tokens)

        print("[stage 2] pre / post / comb normalized PCC vs CPU reference")
        assert_pcc(pre_ref, pre_tt_out, threshold=0.9995)
        assert_pcc(post_ref, post_tt_out, threshold=0.9995)
        assert_pcc(comb_ref, comb_tt_out, threshold=0.9995)
        print("\nCHAIN PCC OK -- split_mixes -> CPU repack -> sinkhorn reproduces hc_split_sinkhorn")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    run()
