"""Case s3: full Indexer.forward decode path on device.

Wraps a CPU Indexer with on-device wq_b + weights_proj (DeviceColLinear)
and an inner DeviceCompressorOverlapRotate (the s2 class). Reads back
`index_score` to CPU for the topk so we can compare against the CPU
reference; topk on device is a future optimisation (README candidate
#4: `indexer_score_reduce` already covers fusing relu * weights * sum,
which would be the right place to also emit a device topk).

CPU reference (inference.py:688 with start_pos > 0, S=1):

    freqs_cis = self.freqs_cis[start_pos:start_pos+1]
    q = self.wq_b(qr).unflatten(-1, (H, D))
    apply_rotary_emb(q[..., -rd:], freqs_cis)
    q = rotate_activation(q)
    # fp4_act_quant SKIPPED (bf16 policy)
    self.compressor(x, start_pos)             # updates kv_cache (slot start_pos // ratio)
    weights = self.weights_proj(x) * (head_dim**-0.5 * H**-0.5)
    end_pos = start_pos + 1
    score = einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
    score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    topk_idxs = score.topk(min(index_topk, end_pos // ratio), dim=-1)[1] + offset
"""
from __future__ import annotations

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import ttnn

from harness import (
    load_inference, make_test_args,
    replicated, to_torch_replicated, report, open_mesh,
)
from compressor_overlap_rotate import DeviceCompressorOverlapRotate

inf = load_inference()


class DeviceIndexer:
    """Decode-only DeviceIndexer. Owns:
      - DeviceCompressorOverlapRotate (s2) for the inner compressor
      - DeviceColLinear for wq_b and weights_proj
      - Hadamard table for q rotate
    Topk happens on CPU (small tensor) for now.
    """

    def __init__(self, mesh, indexer: "inf.Indexer",
                 dc: DeviceCompressorOverlapRotate,
                 wq_b_dev, weights_proj_dev):
        self.mesh = mesh
        self.indexer = indexer
        self.dc = dc
        self.wq_b = wq_b_dev
        self.weights_proj = weights_proj_dev

        self.head_dim = indexer.head_dim
        self.rope_head_dim = indexer.rope_head_dim
        self.n_heads = indexer.n_heads
        self.compress_ratio = indexer.compress_ratio
        self.softmax_scale = indexer.softmax_scale
        self.index_topk = indexer.index_topk

        # Hadamard for q rotate (same head_dim as compressor).
        h_mat = (inf._sylvester_hadamard(self.head_dim) *
                 (self.head_dim ** -0.5)).to(torch.bfloat16)
        self.h_tt = replicated(ttnn, mesh, h_mat)

        # cos/sin shared with compressor (same compress freqs_cis).
        fc = indexer.freqs_cis
        self.cos_full_tt = replicated(ttnn, mesh, fc.real.to(torch.bfloat16).contiguous())
        self.sin_full_tt = replicated(ttnn, mesh, fc.imag.to(torch.bfloat16).contiguous())

    def forward_device_score(self, x_tt, qr_tt, B: int, start_pos: int):
        """Return index_score as a device tensor of shape [B, 1, T_pad].
        Caller pulls to CPU, masks slots >= end_pos // ratio, and runs topk.
        """
        H = self.n_heads
        D = self.head_dim
        rd = self.rope_head_dim

        # 1. q = wq_b(qr); reshape [B, 1, H*D] -> [B, 1, H, D]
        q_tt = self.wq_b.forward_device(qr_tt)
        q_tt = ttnn.reshape(q_tt, [B, 1, H, D])

        # 2. apply rotary on q[..., -rd:].
        rd_half = rd // 2
        cos = ttnn.slice(self.cos_full_tt, [start_pos, 0], [start_pos + 1, rd_half])
        sin = ttnn.slice(self.sin_full_tt, [start_pos, 0], [start_pos + 1, rd_half])
        cos = ttnn.reshape(cos, [1, 1, 1, rd_half])
        sin = ttnn.reshape(sin, [1, 1, 1, rd_half])
        q_nope = ttnn.slice(q_tt, [0, 0, 0, 0],     [B, 1, H, D - rd])
        q_rope = ttnn.slice(q_tt, [0, 0, 0, D - rd], [B, 1, H, D])
        q_rope = inf._device_apply_rotary_interleaved(ttnn, q_rope, cos, sin, inverse=False)
        q_tt = ttnn.concat([q_nope, q_rope], dim=-1)

        # 3. q = rotate_activation(q)  (Walsh-Hadamard along last dim).
        q_tt = inf._device_rotate_activation(ttnn, q_tt, self.h_tt)
        # fp4_act_quant SKIPPED (bf16 policy).

        # 4. compressor(x, start_pos)  -- updates dc.kv_cache_tt at compress events.
        self.dc.forward_device(x_tt, B, start_pos)

        # 5. weights = weights_proj(x) * scale  -> [B, 1, H]
        scale = self.softmax_scale * (H ** -0.5)
        w_tt = self.weights_proj.forward_device(x_tt)
        w_tt = ttnn.multiply(w_tt, scale)

        # 6. einsum: score [B, 1, H, T_pad] = q [B,1,H,D] @ kv_cache_T [B,1,D,T_pad]
        kv_cache_tt = self.dc.kv_cache_tt                    # [B, 1, T_pad, D]
        kv_T_tt = ttnn.transpose(kv_cache_tt, -2, -1)         # [B, 1, D, T_pad]
        score = ttnn.matmul(q_tt, kv_T_tt)                    # [B, 1, H, T_pad]

        # 7. relu * weights -> sum over H. Transpose H to last dim so the reduce
        # stays on the inner-most axis (cleanest in ttnn; tt-lang fusion candidate
        # `indexer_score_reduce` from README).
        score = ttnn.relu(score)
        score_t = ttnn.transpose(score, -2, -1)               # [B, 1, T_pad, H]
        w_b = ttnn.reshape(w_tt, [B, 1, 1, H])                # broadcast over T
        score_t = ttnn.multiply(score_t, w_b)
        score_red = ttnn.sum(score_t, dim=-1, keepdim=False)  # [B, 1, T_pad]
        return score_red


def main() -> int:
    print("=" * 72)
    print("s3: Indexer decode  (full forward path; topk on CPU for now)")
    print("=" * 72)
    args = make_test_args(inf, max_seq_len=64)
    ratio = 4
    head_dim = args.index_head_dim   # 128

    # Match production: weights and the kv_cache buffer are bf16 under
    # default_dtype=bf16 (set at the top of inference.py's main).
    torch.set_default_dtype(torch.bfloat16)

    # Build CPU Indexer with deterministic weights.
    g = torch.Generator().manual_seed(1234)

    def _build_indexer():
        ix = inf.Indexer(args, compress_ratio=ratio).eval()
        for p in ix.parameters():
            with torch.no_grad():
                p.copy_(torch.empty_like(p).normal_(generator=g) * 0.02)
        # Wire kv_cache + freqs_cis (mirrors Attention.__init__).
        ix.kv_cache = torch.zeros(args.max_batch_size,
                                  args.max_seq_len // ratio, head_dim)
        rd = ix.rope_head_dim
        ix.freqs_cis = inf.precompute_freqs_cis(
            rd, args.max_seq_len, args.original_seq_len, args.compress_rope_theta,
            args.rope_factor, args.beta_fast, args.beta_slow,
        )
        return ix

    cpu_ix = _build_indexer()
    g = torch.Generator().manual_seed(1234)  # rebuild same weights
    dev_ix = _build_indexer()
    dev_ix.load_state_dict(cpu_ix.state_dict())
    dev_ix.kv_cache = torch.zeros_like(cpu_ix.kv_cache)
    dev_ix.freqs_cis = cpu_ix.freqs_cis
    dev_ix.compressor.kv_cache = dev_ix.kv_cache  # mirror CPU side wire-up
    dev_ix.compressor.freqs_cis = dev_ix.freqs_cis
    cpu_ix.compressor.kv_cache = cpu_ix.kv_cache
    cpu_ix.compressor.freqs_cis = cpu_ix.freqs_cis

    with open_mesh(ttnn) as mesh:
        wq_b_dev = inf.DeviceColLinear(mesh, dev_ix.wq_b.weight)
        weights_proj_dev = inf.DeviceColLinear(mesh, dev_ix.weights_proj.weight)
        wkv_dev = inf.DeviceColLinear(mesh, dev_ix.compressor.wkv.weight)
        wgate_dev = inf.DeviceColLinear(mesh, dev_ix.compressor.wgate.weight)
        norm_dev = inf.DeviceRMSNorm(mesh, dev_ix.compressor.norm.weight,
                                     dev_ix.compressor.norm.eps)
        dc = DeviceCompressorOverlapRotate(mesh, dev_ix.compressor,
                                           wkv_dev, wgate_dev, norm_dev)
        dix = DeviceIndexer(mesh, dev_ix, dc, wq_b_dev, weights_proj_dev)

        B = 1
        offset = 0
        all_ok = True
        n_steps = 12
        for step in range(n_steps):
            x = torch.randn(B, 1, args.dim, dtype=torch.bfloat16)
            qr = torch.randn(B, 1, args.q_lora_rank, dtype=torch.bfloat16)

            cpu_idxs = cpu_ix(x.clone(), qr.clone(), step, offset)
            x_tt = replicated(ttnn, mesh, x)
            qr_tt = replicated(ttnn, mesh, qr)
            dev_score = dix.forward_device_score(x_tt, qr_tt, B, step)

            end_pos = step + 1
            T_active = end_pos // ratio
            if T_active == 0:
                # No compressed tokens yet; CPU topk is over an empty range.
                continue

            # Pull device score, mask inactive slots, topk.
            sc_dev = to_torch_replicated(ttnn, mesh, dev_score)[:B]   # [B, 1, T_pad]
            sc_dev = sc_dev.view(B, dc.kv_cache_T_pad)[:, :T_active]
            k = min(dix.index_topk, T_active)
            dev_idxs = sc_dev.topk(k, dim=-1)[1] + offset

            # Compare topk index sets (orderless equality on small T_active).
            cpu_set = set(cpu_idxs.flatten().tolist())
            dev_set = set(dev_idxs.flatten().tolist())
            ok = cpu_set == dev_set
            label = f"topk@step{step:02d} (T={T_active}, k={k})"
            print(f"  [{'OK' if ok else 'FAIL'}] {label:<32}  cpu={sorted(cpu_set)}  dev={sorted(dev_set)}")
            all_ok &= ok

        # Final kv_cache cross-check (state coherence after the test run).
        cache_dev = to_torch_replicated(ttnn, mesh, dc.kv_cache_tt)[:B]
        cache_dev = cache_dev.view(B, dc.kv_cache_T_pad, head_dim)[:, :dc.kv_cache_T]
        all_ok &= report("kv_cache final", cpu_ix.kv_cache[:B], cache_dev, pcc_min=0.99)

        print(f"\n{'OK' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
