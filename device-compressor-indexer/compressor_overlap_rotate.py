"""Case s2: DeviceCompressor decode, ratio=4, overlap=True, rotate=True.

This is the Compressor that lives *inside* the Indexer (head_dim=128, the
`index_head_dim`). Identical to s1 except:

  - head_dim is 128 instead of 512
  - after apply_rotary on `kv[..., -rd:]`, the full `kv` is run through
    `rotate_activation` (Walsh-Hadamard butterfly, single matmul on device)
    before being written to kv_cache.

`fp4_act_quant` from the CPU reference is *omitted* under the bf16 policy
(see ./README.md). Both fp4 and fp8 act_quant paths fold to identity once
the lossy fp* cast is dropped.

Decode-step CPU reference (inference.py:629-655 with rotate=True):

    ... s1 logic ...
    kv = norm(kv); apply_rotary_emb(kv[..., -rd:], freqs_cis[start_pos+1-ratio])
    kv = rotate_activation(kv)        # <- new vs s1
    # fp4_act_quant SKIPPED (bf16 policy)
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
    replicated, to_torch_replicated, report, open_mesh,
)
from compressor_overlap import DeviceCompressorOverlap, _gather_cpu_state

inf = load_inference()


class DeviceCompressorOverlapRotate(DeviceCompressorOverlap):
    """Decode-only DeviceCompressor for the (overlap=True, rotate=True) case.
    Reuses everything from s1 and just inserts the Walsh-Hadamard rotate
    matmul before the kv_cache write."""

    def __init__(self, mesh, comp: "inf.Compressor", wkv_dev, wgate_dev, norm_dev):
        if not comp.rotate:
            raise ValueError("DeviceCompressorOverlapRotate requires rotate=True")
        # Bypass parent's rotate=False guard.
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        self.comp = comp
        self.wkv = wkv_dev
        self.wgate = wgate_dev
        self.norm_dev = norm_dev
        if not comp.overlap:
            raise ValueError("DeviceCompressorOverlapRotate requires overlap=True")

        self.head_dim = comp.head_dim
        self.rope_head_dim = comp.rope_head_dim
        self.compress_ratio = comp.compress_ratio
        self.coff = 2
        self.cdim = self.coff * self.head_dim
        self.two_ratio = 2 * self.compress_ratio

        self.ape_tt = replicated(ttnn, mesh, comp.ape.to(torch.bfloat16))
        fc = comp.freqs_cis
        self.cos_full_tt = replicated(ttnn, mesh, fc.real.to(torch.bfloat16).contiguous())
        self.sin_full_tt = replicated(ttnn, mesh, fc.imag.to(torch.bfloat16).contiguous())

        # Hadamard table for rotate_activation; matches scripts/test_rotate_activation.py.
        h_mat = (inf._sylvester_hadamard(self.head_dim) * (self.head_dim ** -0.5)).to(torch.bfloat16)
        self.h_tt = replicated(ttnn, mesh, h_mat)

        self.kv_state_front_tt = None
        self.kv_state_back_tt = None
        self.score_state_front_tt = None
        self.score_state_back_tt = None
        self.kv_cache_tt = None

    def forward_device(self, x_tt, B: int, start_pos: int):
        ratio = self.compress_ratio
        d = self.head_dim
        c = self.cdim
        rd = self.rope_head_dim

        if self.kv_state_front_tt is None:
            self._alloc_state(B)

        # 1. Linears.
        kv_tt = self.wkv.forward_device(x_tt)
        score_tt = self.wgate.forward_device(x_tt)

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
        ttnn.kv_cache.update_cache_for_token_(self.kv_state_front_tt,
                                              ttnn.reshape(kv_front, [B, 1, 1, d]),
                                              back_slot, 0)
        ttnn.kv_cache.update_cache_for_token_(self.kv_state_back_tt,
                                              ttnn.reshape(kv_back,  [B, 1, 1, d]),
                                              back_slot, 0)
        ttnn.kv_cache.update_cache_for_token_(self.score_state_front_tt,
                                              ttnn.reshape(score_front, [B, 1, 1, d]),
                                              back_slot, 0)
        ttnn.kv_cache.update_cache_for_token_(self.score_state_back_tt,
                                              ttnn.reshape(score_back,  [B, 1, 1, d]),
                                              back_slot, 0)

        if (start_pos + 1) % ratio != 0:
            return None

        # 5+6. Two-stripe view -> weighted softmax-sum -> RMSNorm.
        # See s1 TODO comment about fusing into compressor_softmax_sum_norm.
        front_kv = ttnn.slice(self.kv_state_front_tt, [0, 0, 0, 0],     [B, 1, ratio,     d])
        back_kv  = ttnn.slice(self.kv_state_back_tt,  [0, 0, ratio, 0], [B, 1, 2 * ratio, d])
        kv_view = ttnn.concat([front_kv, back_kv], dim=2)

        front_sc = ttnn.slice(self.score_state_front_tt, [0, 0, 0, 0],     [B, 1, ratio,     d])
        back_sc  = ttnn.slice(self.score_state_back_tt,  [0, 0, ratio, 0], [B, 1, 2 * ratio, d])
        score_view = ttnn.concat([front_sc, back_sc], dim=2)

        sm_tt = ttnn.softmax(score_view, dim=-2)
        weighted = ttnn.multiply(kv_view, sm_tt)
        kv_sum = ttnn.sum(weighted, dim=-2, keepdim=True)

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

        # 8. rotate_activation (single matmul against H_d / sqrt(d)).
        kv_normed = inf._device_rotate_activation(ttnn, kv_normed, self.h_tt)

        # 9. kv_cache write.
        comp_idx = start_pos // ratio
        kv_4d_out = ttnn.reshape(kv_normed, [B, 1, 1, d])
        ttnn.kv_cache.update_cache_for_token_(self.kv_cache_tt, kv_4d_out, comp_idx, 0)

        # 10. State shift (same as s1, see TODO there).
        for buf in (self.kv_state_front_tt, self.kv_state_back_tt,
                    self.score_state_front_tt, self.score_state_back_tt):
            for i in range(ratio):
                slot_src = ttnn.slice(buf, [0, 0, ratio + i, 0],
                                           [B, 1, ratio + i + 1, d])
                ttnn.kv_cache.update_cache_for_token_(buf, slot_src, i, 0)

        return kv_normed


def main() -> int:
    print("=" * 72)
    print("s2: Compressor decode  (ratio=4, overlap=True, rotate=True)  [Indexer's compressor]")
    print("=" * 72)
    args = make_test_args(inf, max_seq_len=64)   # 64/4 = 16 compress slots
    ratio = 4
    head_dim = args.index_head_dim   # 128 (Indexer's compressor)
    cpu, dev = fresh_compressor(inf, args, compress_ratio=ratio, head_dim=head_dim,
                                rotate=True, seed=0)

    with open_mesh(ttnn) as mesh:
        wkv_dev = inf.DeviceColLinear(mesh, dev.wkv.weight)
        wgate_dev = inf.DeviceColLinear(mesh, dev.wgate.weight)
        norm_dev = inf.DeviceRMSNorm(mesh, dev.norm.weight, dev.norm.eps)
        dc = DeviceCompressorOverlapRotate(mesh, dev, wkv_dev, wgate_dev, norm_dev)

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

                ks, ss = _gather_cpu_state(dc, B)
                cpu_ss = cpu.score_state[:B].clone()
                cpu_ss[cpu_ss == float("-inf")] = 0
                dev_ss = ss.clone()
                dev_ss[dev_ss == float("-inf")] = 0
                all_ok &= report(f"kv_state@step{step}",  cpu.kv_state[:B], ks, pcc_min=0.99)
                all_ok &= report(f"score_state@step{step}", cpu_ss, dev_ss, pcc_min=0.99)

        cache_dev = to_torch_replicated(ttnn, mesh, dc.kv_cache_tt)[:B]
        cache_dev = cache_dev.view(B, dc.kv_cache_T_pad, head_dim)[:, :dc.kv_cache_T]
        all_ok &= report("kv_cache final", cpu.kv_cache[:B], cache_dev, pcc_min=0.99)

        print(f"\n{'OK' if all_ok else 'FAIL'} -- compress events at steps {compress_steps}")
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
