"""Case s1: DeviceCompressor decode, ratio=4, overlap=True, rotate=False.

Used by Attention layers with `compress_ratio == 4`. Adds two things over s0:

  1. The two-stripe view at compress time:
       cat([kv_state[:, :ratio, :d],
            kv_state[:, ratio:2*ratio, d:]], dim=1)  -> [B, 2*ratio, d]
     (same for score_state). This is the "overlap" -- the weighted sum draws
     from BOTH the front (previous block's first stripe) and back (current
     block's second stripe) halves.

  2. The state shift after each compress event:
       kv_state[:, :ratio]    = kv_state[:, ratio:]
       score_state[:, :ratio] = score_state[:, ratio:]
     The back half's most recent block becomes the next compress's "previous
     block" front half.

State layout: rather than holding a single `[B, 2*ratio, 2*d]` buffer (last
dim 1024 for head_dim=512 -- triggers L1 overflow in
`update_cache_for_token_`), we maintain four buffers each `[B, 1, ratio_pad,
d]`:

    kv_state_front      <-> kv_state[..., :d]
    kv_state_back       <-> kv_state[..., d:]
    score_state_front   <-> score_state[..., :d]
    score_state_back    <-> score_state[..., d:]

The two-stripe view is then just `cat(front[:ratio], back[ratio:2*ratio])`
along the slot axis -- no column-axis slice. Each per-slot write is two
half-width updates (last dim d, not 2d), which fits in L1.

Decode-step CPU reference (inference.py:627):

    kv = wkv(x); score = wgate(x)
    score += ape[start_pos % ratio]
    self.kv_state[:bsz, ratio + start_pos % ratio]    = kv.squeeze(1)
    self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
    if (start_pos+1) % ratio == 0:
        kv_v = cat([self.kv_state[:bsz, :ratio, :d],
                    self.kv_state[:bsz, ratio:, d:]], dim=1)
        sc_v = cat([self.score_state[:bsz, :ratio, :d],
                    self.score_state[:bsz, ratio:, d:]], dim=1)
        kv = (kv_v * sc_v.softmax(dim=1)).sum(dim=1, keepdim=True)
        self.kv_state[:bsz, :ratio]    = self.kv_state[:bsz, ratio:]
        self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
        kv = norm(kv); apply_rotary_emb(kv[..., -rd:], freqs_cis[start_pos+1-ratio])
        # act_quant SKIPPED (bf16 policy)
        self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv
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


class DeviceCompressorOverlap:
    """Decode-only DeviceCompressor for the (overlap=True, rotate=False) case."""

    def __init__(self, mesh, comp: "inf.Compressor", wkv_dev, wgate_dev, norm_dev):
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        self.comp = comp
        self.wkv = wkv_dev
        self.wgate = wgate_dev
        self.norm_dev = norm_dev
        if not comp.overlap:
            raise ValueError("DeviceCompressorOverlap requires overlap=True")
        if comp.rotate:
            raise ValueError("DeviceCompressorOverlap requires rotate=False (use rotate variant)")

        self.head_dim = comp.head_dim
        self.rope_head_dim = comp.rope_head_dim
        self.compress_ratio = comp.compress_ratio
        self.coff = 2  # overlap=True
        self.cdim = self.coff * self.head_dim   # column dim of CPU kv_state / score_state
        self.two_ratio = 2 * self.compress_ratio

        # APE: [ratio, coff*head_dim]. We slice halves to add to kv_tt halves.
        self.ape_tt = replicated(ttnn, mesh, comp.ape.to(torch.bfloat16))

        fc = comp.freqs_cis
        self.cos_full_tt = replicated(ttnn, mesh, fc.real.to(torch.bfloat16).contiguous())
        self.sin_full_tt = replicated(ttnn, mesh, fc.imag.to(torch.bfloat16).contiguous())

        self.kv_state_front_tt = None
        self.kv_state_back_tt = None
        self.score_state_front_tt = None
        self.score_state_back_tt = None
        self.kv_cache_tt = None

    def _alloc_state(self, B: int):
        comp = self.comp
        d = self.head_dim
        ratio_pad = -(-self.two_ratio // 32) * 32   # at least 32 for ratio=4
        self.ratio_pad = ratio_pad

        # Split CPU [B, 2*ratio, 2*d] into front (..., :d) and back (..., d:).
        kv_init = comp.kv_state[:B].to(torch.bfloat16)            # [B, 2*ratio, 2*d]
        score_init = comp.score_state[:B].to(torch.bfloat16)
        kv_init_front = kv_init[..., :d]                          # [B, 2*ratio, d]
        kv_init_back = kv_init[..., d:]
        score_init_front = score_init[..., :d]
        score_init_back = score_init[..., d:]

        if ratio_pad != self.two_ratio:
            zero_pad = torch.zeros(B, ratio_pad - self.two_ratio, d, dtype=torch.bfloat16)
            ninf_pad = torch.full_like(zero_pad, float("-inf"))
            kv_init_front = torch.cat([kv_init_front, zero_pad], dim=1)
            kv_init_back  = torch.cat([kv_init_back,  zero_pad], dim=1)
            score_init_front = torch.cat([score_init_front, ninf_pad], dim=1)
            score_init_back  = torch.cat([score_init_back,  ninf_pad], dim=1)

        self.kv_state_front_tt = replicated(ttnn, self.mesh, kv_init_front.view(B, 1, ratio_pad, d))
        self.kv_state_back_tt  = replicated(ttnn, self.mesh, kv_init_back.view(B, 1, ratio_pad, d))
        self.score_state_front_tt = replicated(ttnn, self.mesh, score_init_front.view(B, 1, ratio_pad, d))
        self.score_state_back_tt  = replicated(ttnn, self.mesh, score_init_back.view(B, 1, ratio_pad, d))

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
        ratio = self.compress_ratio
        d = self.head_dim
        c = self.cdim
        rd = self.rope_head_dim

        if self.kv_state_front_tt is None:
            self._alloc_state(B)

        # 1. Linears.
        kv_tt = self.wkv.forward_device(x_tt)        # [B, 1, c=2d]
        score_tt = self.wgate.forward_device(x_tt)   # [B, 1, c=2d]

        # 2. score += ape[start_pos % ratio].
        slot_in_ape = start_pos % ratio
        ape_slot = ttnn.slice(self.ape_tt, [slot_in_ape, 0], [slot_in_ape + 1, c])
        ape_b = ttnn.reshape(ape_slot, [1, 1, c])
        score_tt = ttnn.add(score_tt, ape_b)

        # 3. Split kv/score halves and write to back-half slot.
        kv_front = ttnn.slice(kv_tt, [0, 0, 0], [B, 1, d])
        kv_back  = ttnn.slice(kv_tt, [0, 0, d], [B, 1, c])
        score_front = ttnn.slice(score_tt, [0, 0, 0], [B, 1, d])
        score_back  = ttnn.slice(score_tt, [0, 0, d], [B, 1, c])

        back_slot = ratio + slot_in_ape
        kv_front_4d = ttnn.reshape(kv_front, [B, 1, 1, d])
        kv_back_4d  = ttnn.reshape(kv_back,  [B, 1, 1, d])
        score_front_4d = ttnn.reshape(score_front, [B, 1, 1, d])
        score_back_4d  = ttnn.reshape(score_back,  [B, 1, 1, d])
        ttnn.kv_cache.update_cache_for_token_(self.kv_state_front_tt, kv_front_4d, back_slot, 0)
        ttnn.kv_cache.update_cache_for_token_(self.kv_state_back_tt,  kv_back_4d,  back_slot, 0)
        ttnn.kv_cache.update_cache_for_token_(self.score_state_front_tt, score_front_4d, back_slot, 0)
        ttnn.kv_cache.update_cache_for_token_(self.score_state_back_tt,  score_back_4d,  back_slot, 0)

        if (start_pos + 1) % ratio != 0:
            return None

        # 5+6. Two-stripe view -> weighted softmax-sum -> RMSNorm.
        #
        # TODO(tt-lang): fuse the slice/concat view + softmax-sum + RMSNorm
        # into one `compressor_softmax_sum_norm` kernel that streams the two
        # stripes directly out of the four state buffers, runs the per-(B,d)
        # online softmax along the ratio axis, accumulates the weighted sum,
        # and rsqrt-normalises in place. Eliminates the two-stripe concat,
        # the [B,1,1,d] DRAM round-trip, and the M-tile padding hack below.
        # Models in ../tt-lang-kernels/softmax_stage.py + rmsnorm.py.
        # Note to future agent: if you are copying this into the main inference
        # script, please preserve this comment so we track fusion candidates.
        front_kv = ttnn.slice(self.kv_state_front_tt, [0, 0, 0, 0],     [B, 1, ratio,        d])
        back_kv  = ttnn.slice(self.kv_state_back_tt,  [0, 0, ratio, 0], [B, 1, 2 * ratio,    d])
        kv_view = ttnn.concat([front_kv, back_kv], dim=2)                                # [B,1,2*ratio,d]

        front_sc = ttnn.slice(self.score_state_front_tt, [0, 0, 0, 0],     [B, 1, ratio,     d])
        back_sc  = ttnn.slice(self.score_state_back_tt,  [0, 0, ratio, 0], [B, 1, 2 * ratio, d])
        score_view = ttnn.concat([front_sc, back_sc], dim=2)                             # [B,1,2*ratio,d]

        sm_tt = ttnn.softmax(score_view, dim=-2)
        weighted = ttnn.multiply(kv_view, sm_tt)
        kv_sum = ttnn.sum(weighted, dim=-2, keepdim=True)                                # [B, 1, 1, d]

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
        kv_nope = ttnn.slice(kv_normed, [0, 0, 0],     [B, 1, d - rd])
        kv_rope = ttnn.slice(kv_normed, [0, 0, d - rd], [B, 1, d])
        kv_rope = inf._device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)
        kv_normed = ttnn.concat([kv_nope, kv_rope], dim=-1)

        # 9. kv_cache[start_pos // ratio] = kv_normed.
        comp_idx = start_pos // ratio
        kv_4d_out = ttnn.reshape(kv_normed, [B, 1, 1, d])
        ttnn.kv_cache.update_cache_for_token_(self.kv_cache_tt, kv_4d_out, comp_idx, 0)

        # 10. State shift: kv_state[:, :ratio] = kv_state[:, ratio:].
        #     Per CPU code we shift ALL of c-dim, which here is both
        #     front and back buffers. Done as ratio-many slot copies
        #     (ttnn lacks a batched slot-shift; see README candidate #2).
        #
        # TODO(tt-lang): the four sequential update_cache_for_token_ calls per
        # state buffer below are the second tt-lang candidate from this case
        # (see README's #2 candidate, `compressor_state_shift`). One streamed
        # kernel that copies [B, ratio, d] from slots ratio..2*ratio-1 into
        # slots 0..ratio-1 in a single dispatch is much cleaner than 4*ratio
        # primitive ttnn writes per compress event.
        for buf in (self.kv_state_front_tt, self.kv_state_back_tt,
                    self.score_state_front_tt, self.score_state_back_tt):
            for i in range(ratio):
                slot_src = ttnn.slice(buf, [0, 0, ratio + i, 0],
                                           [B, 1, ratio + i + 1, d])
                ttnn.kv_cache.update_cache_for_token_(buf, slot_src, i, 0)

        return kv_normed


def _gather_cpu_state(dc: DeviceCompressorOverlap, B: int):
    """Pull the four front/back state buffers and rebuild CPU [B, 2*ratio, 2*d]
    tensors for comparison."""
    d = dc.head_dim
    rp = dc.ratio_pad
    front_k = to_torch_replicated(ttnn, dc.mesh, dc.kv_state_front_tt)[:B].view(B, rp, d)[:, :dc.two_ratio]
    back_k  = to_torch_replicated(ttnn, dc.mesh, dc.kv_state_back_tt)[:B].view(B, rp, d)[:, :dc.two_ratio]
    front_s = to_torch_replicated(ttnn, dc.mesh, dc.score_state_front_tt)[:B].view(B, rp, d)[:, :dc.two_ratio]
    back_s  = to_torch_replicated(ttnn, dc.mesh, dc.score_state_back_tt)[:B].view(B, rp, d)[:, :dc.two_ratio]
    kv_state = torch.cat([front_k, back_k], dim=-1)        # [B, 2*ratio, 2*d]
    score_state = torch.cat([front_s, back_s], dim=-1)
    return kv_state, score_state


def main() -> int:
    print("=" * 72)
    print("s1: Compressor decode  (ratio=4, overlap=True, rotate=False)")
    print("=" * 72)
    args = make_test_args(inf, max_seq_len=64)   # 64/4 = 16 compress slots
    ratio = 4
    head_dim = args.head_dim   # 512
    cpu, dev = fresh_compressor(inf, args, compress_ratio=ratio, head_dim=head_dim,
                                rotate=False, seed=0)

    with open_mesh(ttnn) as mesh:
        wkv_dev = inf.DeviceColLinear(mesh, dev.wkv.weight)
        wgate_dev = inf.DeviceColLinear(mesh, dev.wgate.weight)
        norm_dev = inf.DeviceRMSNorm(mesh, dev.norm.weight, dev.norm.eps)
        dc = DeviceCompressorOverlap(mesh, dev, wkv_dev, wgate_dev, norm_dev)

        B = 1
        all_ok = True
        n_steps = 12
        compress_steps = []
        for step in range(n_steps):
            x = torch.randn(B, 1, args.dim, dtype=torch.bfloat16)
            cpu_out = cpu(x.clone(), step)
            x_tt = replicated(ttnn, mesh, x.to(torch.bfloat16))
            dev_out = dc.forward_device(x_tt, B, step)

            if (step + 1) % ratio == 0:
                compress_steps.append(step)
                if cpu_out is None or dev_out is None:
                    print(f"  step={step}: expected compress; got cpu={cpu_out}, dev={dev_out}")
                    return 1
                dev_torch = to_torch_replicated(ttnn, mesh, dev_out)[:B]
                all_ok &= report(f"kv@step{step}", cpu_out, dev_torch, pcc_min=0.99)

                # Compare state buffers AFTER the shift fires.
                ks, ss = _gather_cpu_state(dc, B)
                # CPU score_state has -inf in the back-half (slots that haven't
                # been written this block) -- those slots aren't part of the
                # compress view, but we still want PCC to be high. Mask -inf to
                # 0 on both sides so cosine similarity is well defined.
                cpu_ss = cpu.score_state[:B].clone()
                cpu_ss[cpu_ss == float("-inf")] = 0
                dev_ss = ss.clone()
                dev_ss[dev_ss == float("-inf")] = 0
                all_ok &= report(f"kv_state@step{step}",  cpu.kv_state[:B], ks, pcc_min=0.99)
                all_ok &= report(f"score_state@step{step}", cpu_ss, dev_ss, pcc_min=0.99)

        # Final kv_cache compare across all compresses.
        cache_dev = to_torch_replicated(ttnn, mesh, dc.kv_cache_tt)[:B]
        cache_dev = cache_dev.view(B, dc.kv_cache_T_pad, head_dim)[:, :dc.kv_cache_T]
        all_ok &= report("kv_cache final", cpu.kv_cache[:B], cache_dev, pcc_min=0.99)

        print(f"\n{'OK' if all_ok else 'FAIL'} -- compress events at steps {compress_steps}")
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
