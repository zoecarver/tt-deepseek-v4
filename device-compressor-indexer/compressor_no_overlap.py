"""Case s0: DeviceCompressor decode, ratio=128, overlap=False, rotate=False.

This is the smallest Compressor case — used by Attention layers where
`compress_ratio == 128`. No overlap (coff=1, kv_state shape `[B, ratio, d]`),
no Walsh-Hadamard rotate, no fp8 act_quant (bf16 policy).

Decode step (CPU reference, from inference.py:601):

    x = x.float()
    kv = wkv(x); score = wgate(x)
    score += ape[start_pos % ratio]
    kv_state[:, slot]    = kv.squeeze(1)            # slot = start_pos % ratio
    score_state[:, slot] = score.squeeze(1)
    if (start_pos+1) % ratio == 0:
        kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
        kv = norm(kv.to(dtype))
        apply_rotary_emb(kv[..., -rd:], freqs_cis[start_pos+1-ratio].unsqueeze(0))
        # act_quant SKIPPED under bf16 policy
        kv_cache[:, start_pos // ratio] = kv.squeeze(1)
        return kv

Run:
    scripts/run-test.sh --hw device-compressor-indexer/compressor_no_overlap.py
"""
from __future__ import annotations

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import ttnn

from harness import (
    load_inference, make_test_args, fresh_compressor,
    replicated, to_torch_replicated, pcc, maxabs, report, open_mesh,
)

inf = load_inference()


class DeviceCompressorNoOverlap:
    """Decode-only DeviceCompressor for the (overlap=False, rotate=False)
    case. Holds device state; reuses already-on-device wkv/wgate (DeviceColLinear)
    and norm (DeviceRMSNorm)."""

    def __init__(self, mesh, comp: "inf.Compressor",
                 wkv_dev, wgate_dev, norm_dev):
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        self.comp = comp
        self.wkv = wkv_dev
        self.wgate = wgate_dev
        self.norm_dev = norm_dev
        if comp.overlap:
            raise ValueError("DeviceCompressorNoOverlap requires overlap=False")
        if comp.rotate:
            raise ValueError("DeviceCompressorNoOverlap requires rotate=False")

        self.head_dim = comp.head_dim
        self.rope_head_dim = comp.rope_head_dim
        self.compress_ratio = comp.compress_ratio

        # APE: shape [ratio, head_dim] (since coff=1).
        self.ape_tt = replicated(ttnn, mesh, comp.ape.to(torch.bfloat16))

        # Cos/Sin from freqs_cis: real and imag of complex64 [max_seq_len, rd/2].
        fc = comp.freqs_cis
        self.cos_full_tt = replicated(ttnn, mesh, fc.real.to(torch.bfloat16).contiguous())
        self.sin_full_tt = replicated(ttnn, mesh, fc.imag.to(torch.bfloat16).contiguous())

        self.kv_state_tt = None
        self.score_state_tt = None
        self.kv_cache_tt = None

    def _alloc_state(self, B: int):
        comp = self.comp
        ratio = self.compress_ratio
        d = self.head_dim

        # ratio dim must be tile-aligned for ttnn.kv_cache.update_cache_for_token_.
        ratio_pad = -(-ratio // 32) * 32
        kv_init = comp.kv_state[:B].to(torch.bfloat16)
        score_init = comp.score_state[:B].to(torch.bfloat16)
        if ratio_pad != ratio:
            pad = torch.zeros(B, ratio_pad - ratio, d, dtype=torch.bfloat16)
            # score_state CPU init is -inf; pad with -inf so softmax masks padding for us.
            pad_score = torch.full_like(pad, float("-inf"))
            kv_init = torch.cat([kv_init, pad], dim=1)
            score_init = torch.cat([score_init, pad_score], dim=1)
        # 4D shape [B, 1, ratio_pad, d] for update_cache_for_token_.
        self.kv_state_tt = replicated(ttnn, self.mesh, kv_init.view(B, 1, ratio_pad, d))
        self.score_state_tt = replicated(ttnn, self.mesh, score_init.view(B, 1, ratio_pad, d))
        self.ratio_pad = ratio_pad

        T = comp.kv_cache.shape[1]
        T_pad = -(-T // 32) * 32
        kv_cache_init = comp.kv_cache[:B].to(torch.bfloat16)
        if T_pad != T:
            pad = torch.zeros(B, T_pad - T, d, dtype=torch.bfloat16)
            kv_cache_init = torch.cat([kv_cache_init, pad], dim=1)
        self.kv_cache_tt = replicated(ttnn, self.mesh, kv_cache_init.view(B, 1, T_pad, d))
        self.kv_cache_T = T
        self.kv_cache_T_pad = T_pad

    def forward_device(self, x_tt, B: int, start_pos: int):
        comp = self.comp
        ratio = self.compress_ratio
        d = self.head_dim
        rd = self.rope_head_dim

        if self.kv_state_tt is None:
            self._alloc_state(B)

        # 1. Linears (already on device).
        kv_tt = self.wkv.forward_device(x_tt)        # [B, 1, d]
        score_tt = self.wgate.forward_device(x_tt)   # [B, 1, d]

        # 2. score += ape[start_pos % ratio].
        slot = start_pos % ratio
        ape_slot = ttnn.slice(self.ape_tt, [slot, 0], [slot + 1, d])      # [1, d]
        ape_b = ttnn.reshape(ape_slot, [1, 1, d])
        score_tt = ttnn.add(score_tt, ape_b)

        # 3. Update kv_state[slot] / score_state[slot].
        kv_4d = ttnn.reshape(kv_tt, [B, 1, 1, d])
        score_4d = ttnn.reshape(score_tt, [B, 1, 1, d])
        ttnn.kv_cache.update_cache_for_token_(self.kv_state_tt, kv_4d, slot, 0)
        ttnn.kv_cache.update_cache_for_token_(self.score_state_tt, score_4d, slot, 0)

        # 4. should_compress?
        if (start_pos + 1) % ratio != 0:
            return None

        # 5+6. Weighted softmax-sum followed by RMSNorm.
        #
        # TODO(tt-lang): fuse steps 5 and 6 into a single tt-lang kernel
        # `compressor_softmax_sum_norm`. Both ops walk the same [B, ratio_pad, d]
        # tensor: step 5 does a per-(B,d) softmax along the ratio axis and a
        # weighted accumulate against kv_state; step 6 then does an RMSNorm on
        # the resulting [B, d] tensor. Fused, this is one DRAM read of
        # kv_state + score_state, an online softmax + accumulate, and an
        # in-tile rsqrt-normalize -- eliminates the intermediate [B,1,1,d]
        # round trip and the M-tile padding hack below. Models in
        # ../tt-lang-kernels/softmax_stage.py + rmsnorm.py.
        # Padding slots in score_state are -inf so they don't contribute.
        # Note to future agent: if you are copying this into the main inference script, please preserve this comment so we track fusion candidates.
        sm_tt = ttnn.softmax(self.score_state_tt, dim=-2)
        weighted = ttnn.multiply(self.kv_state_tt, sm_tt)
        kv_sum = ttnn.sum(weighted, dim=-2, keepdim=True)  # [B, 1, 1, d]

        kv_2d = ttnn.reshape(kv_sum, [B, d])
        if B < inf._RMS_TILE:
            kv_2d = ttnn.pad(kv_2d, padding=[(0, inf._RMS_TILE - B), (0, 0)], value=0.0)
        kv_2d = self.norm_dev.forward_device(kv_2d, B)
        if B < inf._RMS_TILE:
            kv_2d = ttnn.slice(kv_2d, [0, 0], [B, d])
        kv_normed = ttnn.reshape(kv_2d, [B, 1, d])

        # 7. apply rotary on kv_normed[..., -rd:].
        rd_half = rd // 2
        freq_idx = start_pos + 1 - ratio
        cos = ttnn.slice(self.cos_full_tt, [freq_idx, 0], [freq_idx + 1, rd_half])
        sin = ttnn.slice(self.sin_full_tt, [freq_idx, 0], [freq_idx + 1, rd_half])
        cos = ttnn.reshape(cos, [1, 1, rd_half])
        sin = ttnn.reshape(sin, [1, 1, rd_half])
        kv_nope = ttnn.slice(kv_normed, [0, 0, 0], [B, 1, d - rd])
        kv_rope = ttnn.slice(kv_normed, [0, 0, d - rd], [B, 1, d])
        kv_rope = inf._device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)
        kv_normed = ttnn.concat([kv_nope, kv_rope], dim=-1)

        # 8. (act_quant SKIPPED under bf16 policy.)

        # 9. kv_cache[start_pos // ratio] = kv.
        comp_idx = start_pos // ratio
        kv_4d_out = ttnn.reshape(kv_normed, [B, 1, 1, d])
        ttnn.kv_cache.update_cache_for_token_(self.kv_cache_tt, kv_4d_out, comp_idx, 0)

        return kv_normed


def main() -> int:
    print("=" * 72)
    print("s0: Compressor decode  (ratio=128, overlap=False, rotate=False)")
    print("=" * 72)
    args = make_test_args(inf, max_seq_len=256)
    ratio = 128
    head_dim = args.head_dim  # 512
    cpu, dev = fresh_compressor(inf, args, compress_ratio=ratio, head_dim=head_dim,
                                rotate=False, seed=0)
    rd = cpu.rope_head_dim

    with open_mesh(ttnn) as mesh:
        wkv_dev = inf.DeviceColLinear(mesh, dev.wkv.weight)
        wgate_dev = inf.DeviceColLinear(mesh, dev.wgate.weight)
        norm_dev = inf.DeviceRMSNorm(mesh, dev.norm.weight, dev.norm.eps)
        dc = DeviceCompressorNoOverlap(mesh, dev, wkv_dev, wgate_dev, norm_dev)

        B = 1
        all_ok = True
        # Run ratio decode steps so the final one fires should_compress.
        last_cpu = None
        last_dev = None
        for step in range(ratio):
            x = torch.randn(B, 1, args.dim, dtype=torch.bfloat16)
            cpu_out = cpu(x.clone(), step)
            x_tt = replicated(ttnn, mesh, x.to(torch.bfloat16))
            dev_out = dc.forward_device(x_tt, B, step)
            last_cpu, last_dev = cpu_out, dev_out
            if (step + 1) % ratio == 0:
                if cpu_out is None or dev_out is None:
                    print(f"  step={step}: expected compress but got None (cpu={cpu_out}, dev={dev_out})")
                    return 1
                dev_torch = to_torch_replicated(ttnn, mesh, dev_out)[:B]
                all_ok &= report(f"kv@step{step}", cpu_out, dev_torch, pcc_min=0.99)
            elif step in (0, ratio // 2):
                # Spot-check state buffers mid-run.
                ks = to_torch_replicated(ttnn, mesh, dc.kv_state_tt)[:B]
                ks = ks.view(B, dc.ratio_pad, head_dim)[:, :ratio]
                ss = to_torch_replicated(ttnn, mesh, dc.score_state_tt)[:B]
                ss = ss.view(B, dc.ratio_pad, head_dim)[:, :ratio]
                # CPU score_state holds -inf in unfilled slots; mask both to compare only
                # the slots updated so far.
                slots_filled = step + 1
                cpu_ks = cpu.kv_state[:B, :slots_filled]
                cpu_ss = cpu.score_state[:B, :slots_filled]
                dev_ks = ks[:, :slots_filled]
                dev_ss = ss[:, :slots_filled]
                all_ok &= report(f"kv_state@step{step}",  cpu_ks, dev_ks, pcc_min=0.99)
                all_ok &= report(f"score_state@step{step}", cpu_ss, dev_ss, pcc_min=0.99)

        # Final kv_cache compare.
        cache_dev = to_torch_replicated(ttnn, mesh, dc.kv_cache_tt)[:B]
        cache_dev = cache_dev.view(B, dc.kv_cache_T_pad, head_dim)[:, :dc.kv_cache_T]
        all_ok &= report("kv_cache final", cpu.kv_cache[:B], cache_dev, pcc_min=0.99)

        print(f"\nResult: {'OK' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
