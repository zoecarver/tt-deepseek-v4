"""DeepSeek-V4-Flash single-file inference.

Phase 1 goal: load weights on CPU, run a forward pass, generate coherent text.

Architecture is adapted from `deepseek-ai/DeepSeek-V4-Flash/inference/model.py`.
All tilelang GPU kernels (FP8/FP4 GEMM, sparse attention, HC Sinkhorn) are
replaced with CPU-safe pure-torch equivalents in the KERNELS section below.

Script structure (nanochat style):
  - KERNELS: CPU replacements for tilelang ops
  - ARCHITECTURE: model modules (RMSNorm, MLA attention, MoE, HC block, ...)
  - TRANSFORMER: the top-level Transformer nn.Module
  - MODEL: high-level Model class with load_weights and step_decode
  - main(): load + prompt + generate
"""

from __future__ import annotations

import math
import os
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from huggingface_hub import snapshot_download


# ==============================================================================
# Global state (mirrors deepseek inference/model.py)
# ==============================================================================

world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt: Optional[str] = "ue8m0"
scale_dtype = torch.float32


# Per-phase wall-time accumulators. Populated by _phase() context manager; printed
# at end of main(). Phases nest freely — each name accumulates independently, so
# a parent phase double-counts its children. Interpret as "time attributed to X,
# inclusive of its instrumented subregions" and only compare sibling phases.
_PHASE_ACCUM: dict[str, float] = {}
_PHASE_COUNTS: dict[str, int] = {}


@contextmanager
def _phase(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _PHASE_ACCUM[name] = _PHASE_ACCUM.get(name, 0.0) + dt
        _PHASE_COUNTS[name] = _PHASE_COUNTS.get(name, 0) + 1


def _phase_report() -> str:
    if not _PHASE_ACCUM:
        return "(no phases recorded)"
    lines = ["phase                      total(s)    n    avg(ms)"]
    for k in sorted(_PHASE_ACCUM, key=lambda k: -_PHASE_ACCUM[k]):
        tot = _PHASE_ACCUM[k]
        n = _PHASE_COUNTS[k]
        avg_ms = 1000.0 * tot / n
        lines.append(f"  {k:24s}  {tot:7.2f}  {n:4d}  {avg_ms:9.1f}")
    return "\n".join(lines)


@contextmanager
def set_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# ==============================================================================
# KERNELS: CPU replacements for inference/kernel.py
# ==============================================================================

# FP4 e2m1 lookup table (value for each 4-bit pattern). Matches convert.py.
_FP4_VALUES = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _round_to_pow2(x: torch.Tensor) -> torch.Tensor:
    """Round positive float tensor up to the next power of 2 (for ue8m0 scale format)."""
    # bit trick: mantissa != 0 => exponent + 1
    u = x.float().view(torch.int32)
    exp = ((u >> 23) & 0xFF)
    man = u & ((1 << 23) - 1)
    exp = exp - 127 + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), exp)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise FP8 quantization along the last dim.

    inplace=True  -> quantize+dequantize back into x (bf16 simulation of FP8).
    inplace=False -> return (y_fp8, scales).
    """
    N = x.size(-1)
    assert N % block_size == 0, f"last dim {N} must be divisible by block_size {block_size}"

    orig_shape = x.shape
    z = x.contiguous().view(-1, N // block_size, block_size)
    amax = z.float().abs().amax(dim=-1).clamp_min(1e-4)
    fp8_max = 448.0
    s = amax / fp8_max
    if scale_fmt == "ue8m0":
        s = _round_to_pow2(s)
    y = (z.float() / s.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    y_fp8 = y.to(torch.float8_e4m3fn)

    if inplace:
        # dequantize back into x in its original dtype
        deq = (y_fp8.to(torch.float32) * s.unsqueeze(-1)).to(x.dtype)
        x.copy_(deq.view(orig_shape))
        return x

    # y shape: [..., N] in fp8; scales shape: [..., N//block_size]
    y_out = y_fp8.view(orig_shape)
    s_out = s.to(scale_dtype).view(*orig_shape[:-1], N // block_size)
    return y_out, s_out


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    """Block-wise FP4 quantization. Power-of-2 scale (ue8m0). inplace does QAT simulation."""
    N = x.size(-1)
    assert N % block_size == 0
    orig_shape = x.shape
    z = x.contiguous().view(-1, N // block_size, block_size)
    amax = z.float().abs().amax(dim=-1).clamp_min(6 * (2**-126))
    fp4_max = 6.0
    s = _round_to_pow2(amax / fp4_max)
    # quantize-dequantize via table lookup to nearest FP4 value
    scaled = (z.float() / s.unsqueeze(-1)).clamp(-fp4_max, fp4_max)
    values = _FP4_VALUES.to(scaled.device)
    # find nearest fp4 value
    diff = (scaled.unsqueeze(-1) - values).abs()
    idx = diff.argmin(dim=-1)
    quant = values[idx]
    if inplace:
        deq = (quant * s.unsqueeze(-1)).to(x.dtype)
        x.copy_(deq.view(orig_shape))
        return x
    # Pack two FP4 values per byte: low = even index, high = odd
    # But for CPU impl we just keep quant*s as bf16 since we never use the packed form in compute
    raise NotImplementedError("non-inplace fp4_act_quant not needed on CPU path")


def _dequant_fp8_weight(w_fp8: torch.Tensor, scale: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Dequantize an FP8 weight [N, K] with per-[group,group] scale to bf16."""
    N, K = w_fp8.shape
    # scale shape: [ceil(N/group), ceil(K/group)]
    gN, gK = scale.shape
    w = w_fp8.to(torch.float32)
    # expand scale to [N, K]
    s = scale.to(torch.float32).repeat_interleave(group_size, dim=0).repeat_interleave(group_size, dim=1)
    s = s[:N, :K]
    return (w * s).to(torch.bfloat16)


def _dequant_fp4_weight(w_fp4: torch.Tensor, scale: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Dequantize an FP4 (e2m1) weight to bf16 via byte-level nibble unpack.

    w_fp4 storage: [N, K/2] bytes (each byte = two packed FP4 values; low nibble = even index).
    The tensor may be stored as float4_e2m1fn_x2 OR raw int8/uint8 (torch 2.11 CPU cannot
    `.to(float32)` float4 directly, so we always go via the byte view).
    scale shape: [N, K/block_size] in float8_e8m0fnu.
    """
    # Reinterpret as uint8 bytes regardless of original dtype.
    if w_fp4.dtype == torch.float4_e2m1fn_x2:
        w_bytes = w_fp4.view(torch.uint8)
    elif w_fp4.dtype == torch.int8:
        w_bytes = w_fp4.view(torch.uint8)
    elif w_fp4.dtype == torch.uint8:
        w_bytes = w_fp4
    else:
        raise TypeError(f"unexpected FP4 storage dtype {w_fp4.dtype}")
    N, Kh = w_bytes.shape  # Kh = K // 2
    K = Kh * 2
    low = (w_bytes & 0x0F).long()
    high = ((w_bytes >> 4) & 0x0F).long()
    table = _FP4_VALUES.to(low.device)
    vals = torch.stack([table[low], table[high]], dim=-1).view(N, K)
    s = scale.to(torch.float32).repeat_interleave(block_size, dim=1)[:, :K]
    return (vals * s).to(torch.bfloat16)


def _quant_act_fp8(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation x (any dtype) to FP8 per-block along K (last) dim.
    Returns (x_fp8, scales [..., K//block]).
    """
    N = x.size(-1)
    orig_shape = x.shape
    z = x.contiguous().view(-1, N // block_size, block_size)
    amax = z.float().abs().amax(dim=-1).clamp_min(1e-4)
    fp8_max = 448.0
    s = amax / fp8_max
    y = (z.float() / s.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    y_fp8 = y.to(torch.float8_e4m3fn).view(orig_shape)
    s_out = s.view(*orig_shape[:-1], N // block_size)
    return y_fp8, s_out


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp8[N,K].T with per-block scales on both sides.
    CPU fallback: dequantize both to bf16 then use F.linear.
    """
    K = a.size(-1)
    M = a.numel() // K
    a_flat = a.view(M, K)
    a_s_flat = a_s.view(M, -1)
    # dequantize A: [M,K] from fp8 * scales [M, K//block]
    a_deq = a_flat.to(torch.float32)
    a_scale = a_s_flat.to(torch.float32).repeat_interleave(block_size, dim=1)[:, :K]
    a_deq = a_deq * a_scale
    b_deq = _dequant_fp8_weight(b, b_s, group_size=block_size)
    out = F.linear(a_deq.to(b_deq.dtype), b_deq)
    return out.view(*a.size()[:-1], b.size(0)).to(torch.get_default_dtype())


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K].T. CPU fallback: dequant both, F.linear."""
    K = a.size(-1)
    M = a.numel() // K
    a_flat = a.view(M, K)
    a_s_flat = a_s.view(M, -1)
    a_scale = a_s_flat.to(torch.float32).repeat_interleave(block_size, dim=1)[:, :K]
    a_deq = a_flat.to(torch.float32) * a_scale
    b_deq = _dequant_fp4_weight(b, b_s, block_size=fp4_block_size)
    out = F.linear(a_deq.to(b_deq.dtype), b_deq)
    return out.view(*a.size()[:-1], b.size(0)).to(torch.get_default_dtype())


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention.
    q:         [b, s, h, d]
    kv:        [b, n, d]
    attn_sink: [h]  (learned extra softmax "null" slot)
    topk_idxs: [b, s, topk] int32, -1 means masked.
    Returns o: [b, s, h, d].
    """
    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)
    # Gather kv at top-k idxs (masked positions -> zeros, score -> -inf)
    safe_idxs = topk_idxs.clamp_min(0).long()
    # kv_gather: [b, s, topk, d]
    kv_gather = torch.gather(
        kv.unsqueeze(1).expand(b, s, kv.size(1), d),
        2,
        safe_idxs.unsqueeze(-1).expand(b, s, topk, d),
    )
    valid = (topk_idxs >= 0).to(q.dtype)  # [b,s,topk]

    # scores: [b, s, h, topk]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), kv_gather.float()) * softmax_scale
    # mask invalid positions
    scores = scores + (valid.log().unsqueeze(2))  # log(0)= -inf for masked
    # include attn_sink: we treat it as extra slot with value 0 vector, score = attn_sink[h]
    sink_scores = attn_sink.view(1, 1, h, 1).expand(b, s, h, 1).float()
    full_scores = torch.cat([scores, sink_scores], dim=-1)
    probs = F.softmax(full_scores, dim=-1)
    # drop the sink slot's contribution by indexing first topk probs only (sink value is 0 so contributes 0)
    probs = probs[..., :topk]
    # weighted sum
    o = torch.einsum("bshk,bskd->bshd", probs, kv_gather.float())
    return o.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split + Sinkhorn normalization for hyper-connection mixing.
    mixes:    [*, (2+hc)*hc]
    hc_scale: [3]
    hc_base:  [(2+hc)*hc]
    Returns (pre [*, hc], post [*, hc], comb [*, hc, hc])."""
    hc = hc_mult
    pre_mix = mixes[..., :hc]
    post_mix = mixes[..., hc:2 * hc]
    comb_mix = mixes[..., 2 * hc:].view(*mixes.shape[:-1], hc, hc)

    pre = torch.sigmoid(pre_mix * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(post_mix * hc_scale[1] + hc_base[hc:2 * hc])
    comb = comb_mix * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)

    # comb = softmax(comb, -1) + eps
    comb = F.softmax(comb, dim=-1) + eps
    # comb = comb / (comb.sum(-2) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard transform along the last dim, scaled by 1/sqrt(d).
    Drop-in CPU replacement for fast_hadamard_transform.hadamard_transform.
    """
    d = x.size(-1)
    assert (d & (d - 1)) == 0 and d > 0, f"last dim {d} must be a power of 2"
    scale = d ** -0.5
    shape = x.shape
    y = x.contiguous().view(-1, d).float()
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=2).view(-1, d)
        h *= 2
    return (y * scale).view(shape).to(x.dtype)


# ==============================================================================
# ARCHITECTURE (adapted from deepseek inference/model.py, CPU-safe)
# ==============================================================================


@dataclass
class ModelArgs:
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4", "fp8"] = "fp4"
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int, ...] = field(default_factory=lambda: (
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 0,
    ))
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert bias is None
    if weight.dtype == torch.float4_e2m1fn_x2:
        # dequant w on the fly
        w = _dequant_fp4_weight(weight, weight.scale, block_size=fp4_block_size)
        return F.linear(x.to(w.dtype), w)
    if weight.dtype == torch.float8_e4m3fn:
        w = _dequant_fp8_weight(weight, weight.scale, group_size=block_size)
        return F.linear(x.to(w.dtype), w)
    return F.linear(x, weight)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2)
            )
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(out_features, in_features // fp4_block_size, dtype=torch.float8_e8m0fnu)
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(
                    (out_features + block_size - 1) // block_size,
                    (in_features + block_size - 1) // block_size,
                    dtype=torch.float8_e8m0fnu,
                )
            )
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.register_parameter("scale", None)
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, None)


class ColumnParallelLinear(Linear):
    pass  # world_size=1 -> identity shard


class RowParallelLinear(Linear):
    pass


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_v, max_v, dim):
        if min_v == max_v:
            max_v += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_v) / (max_v - min_v)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x
    x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_c.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_c.size(1), x_c.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_c.size(1), 1, x_c.size(-1))
    x_c = torch.view_as_real(x_c * freqs_cis).flatten(-2)
    y.copy_(x_c)
    return y


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0
        )
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: Optional[torch.Tensor] = None
        self.register_buffer(
            "kv_state",
            torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (args.max_batch_size, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio:cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio:cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset:offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset:offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat(
                        [self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # QAT sim on CPU: quantize+dequant q in place
        fp4_act_quant(q, fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None
        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer(
            "kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim), persistent=False
        )
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, args.max_seq_len, original_seq_len, rope_theta, args.rope_factor,
            args.beta_fast, args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = self._sparse_attn(q, kv, topk_idxs)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = self._sparse_attn(q, self.kv_cache[:bsz], topk_idxs)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        o = self._wo_a_grouped(o)
        x = self.wo_b(o.flatten(2))
        return x

    def _sparse_attn(self, q: torch.Tensor, kv: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
        """Default CPU path; offload_sparse_attn swaps for a device path."""
        return sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

    def _wo_a_grouped(self, o: torch.Tensor) -> torch.Tensor:
        """Block-diagonal per-group linear: [B,S,G,D] -> [B,S,G,R].

        Default CPU path; offload_attn_wo_a swaps this for a device path.
        """
        w = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        return torch.einsum("bsgd,grd->bsgr", o.to(w.dtype), w)


class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit: float = 0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None):
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else (
            torch.float8_e4m3fn if args.expert_dtype == "fp8" else None
        )
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit)
             for _ in range(self.n_routed_experts)]
        )
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]):
        residual = x
        with _phase("block.hc_pre"):
            x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        with _phase("block.norm"):
            x = self.attn_norm(x)
        with _phase("block.attn"):
            x = self.attn(x, start_pos)
        with _phase("block.hc_post"):
            x = self.hc_post(x, residual, post, comb)

        residual = x
        with _phase("block.hc_pre"):
            x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        with _phase("block.norm"):
            x = self.ffn_norm(x)
        with _phase("block.ffn"):
            x = self.ffn(x, input_ids)
        with _phase("block.hc_post"):
            x = self.hc_post(x, residual, post, comb)
        return x


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def get_logits(self, x):
        return F.linear(x[:, -1].float(), self.weight)

    def forward(self, x: torch.Tensor, hc_fn, hc_scale, hc_base, norm: RMSNorm):
        with _phase("head.hc_combiner"):
            x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        with _phase("head.norm_and_logits"):
            logits = self.get_logits(norm(x))
        return logits

    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


# ==============================================================================
# TRANSFORMER
# ==============================================================================


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        global default_dtype, scale_fmt, scale_dtype
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        # MTP layers intentionally dropped on the CPU path: DeepSeek-V4-Flash's weights
        # include an MTP block but it's only used for speculative decoding. Base autoregressive
        # generation runs fine without it.
        self.hc_mult = args.hc_mult
        hc_dim = args.hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(args.hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        with _phase("embed"):
            h = self.embed(input_ids)
            h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        with _phase("blocks"):
            for layer in self.layers:
                h = layer(h, start_pos, input_ids)
        with _phase("head"):
            logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits


# ==============================================================================
# DEVICE OFFLOAD: per-op wrappers for Tenstorrent mesh
# ==============================================================================


def _weight_to_bf16(cpu_weight: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8/FP4 weights with per-block scales; else cast to bf16.

    Errors if called on a quantized dtype without a .scale attribute — see the
    Parameter.data gotcha: accessing .data on an nn.Parameter strips custom
    attributes like .scale, so callers must pass the Parameter itself.
    """
    scale = getattr(cpu_weight, "scale", None)
    if cpu_weight.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise ValueError(
                "FP8 weight passed without .scale attribute; naive cast would "
                "produce unscaled bytes. Pass the nn.Parameter, not .data."
            )
        return _dequant_fp8_weight(cpu_weight, scale, group_size=block_size)
    if cpu_weight.dtype == torch.float4_e2m1fn_x2:
        if scale is None:
            raise ValueError(
                "FP4 weight passed without .scale attribute; naive cast would "
                "produce unscaled bytes. Pass the nn.Parameter, not .data."
            )
        return _dequant_fp4_weight(cpu_weight, scale, block_size=fp4_block_size)
    return cpu_weight.to(torch.bfloat16)


class DeviceLMHead:
    """Column-parallel lm_head (+ optional final RMSNorm) across a 1xN mesh.

    Replaces ParallelHead.get_logits. Shards [dim, vocab] along vocab (last dim)
    so each chip holds vocab/N cols, runs the matmul on-device, and gathers on
    the host. vocab must be divisible by N.

    If `norm_weight` is provided, also applies final RMSNorm on device before
    the matmul — one fewer CPU→device round-trip, one less CPU norm.
    """

    def __init__(self, mesh, cpu_weight: torch.Tensor,
                 norm_weight: Optional[torch.Tensor] = None,
                 norm_eps: float = 1e-6):
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        vocab, dim = cpu_weight.shape
        if vocab % self.mesh_shape[1] != 0:
            raise ValueError(f"vocab {vocab} not divisible by mesh axis 1 {self.mesh_shape[1]}")
        self.vocab = vocab
        self.dim = dim
        self.norm_eps = norm_eps
        w_dv = _weight_to_bf16(cpu_weight).transpose(0, 1).contiguous()  # [dim, vocab]
        self.w_tt = ttnn.as_tensor(
            w_dv,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, self.mesh_shape, dims=(None, -1)),
        )
        self.norm_tt = None
        if norm_weight is not None:
            # Replicate norm weight across the mesh (tiny, [dim]).
            nw = norm_weight.to(torch.bfloat16).contiguous()
            self.norm_tt = ttnn.as_tensor(
                nw,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

    def __call__(self, x_last: torch.Tensor) -> torch.Tensor:
        # x_last: [B, dim] — unnormalized if norm_tt is set, else normalized.
        ttnn = self._ttnn
        x_3d = x_last.to(torch.bfloat16).unsqueeze(1).contiguous()  # [B, 1, dim]
        x_tt = ttnn.as_tensor(
            x_3d,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        if self.norm_tt is not None:
            # ttnn.rms_norm wants 4D input; wrap [B, 1, dim] -> [1, 1, B, dim] -> back.
            B = x_last.shape[0]
            x_tt = ttnn.reshape(x_tt, (1, 1, B, self.dim))
            x_tt = ttnn.rms_norm(x_tt, weight=self.norm_tt, epsilon=self.norm_eps)
            x_tt = ttnn.reshape(x_tt, (B, 1, self.dim))
        y_tt = ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.to_torch(
            y_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(0, -1)),
        )
        if y.shape[0] != x_last.shape[0]:
            y = y[: x_last.shape[0]]
        return y[:, 0, :].float()  # [B, vocab]


class DeviceColLinear(nn.Module):
    """Column-parallel linear across a 1xN mesh.

    Weight: [out, in] (nn.Linear order). Transposed to [in, out] for ttnn,
    column-sharded on the output dim: each chip holds [in, out/N]. Output
    is gathered (concat on last dim) back to host as bf16.
    """

    def __init__(self, mesh, cpu_weight: torch.Tensor):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        out_dim, in_dim = cpu_weight.shape
        if out_dim % self.mesh_shape[1] != 0:
            raise ValueError(
                f"col-linear out_dim {out_dim} not divisible by mesh axis 1 {self.mesh_shape[1]}"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        w_io = _weight_to_bf16(cpu_weight).transpose(0, 1).contiguous()  # [in, out]
        self.w_tt = ttnn.as_tensor(
            w_io,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, self.mesh_shape, dims=(None, -1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ttnn = self._ttnn
        B, S, D = x.shape
        x_tt = ttnn.as_tensor(
            x.to(torch.bfloat16).contiguous(),
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        y_tt = ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.to_torch(
            y_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(0, -1)),
        )
        if y.shape[0] != B:
            y = y[:B]
        if y.shape[1] != S:
            y = y[:, :S]
        return y.to(torch.bfloat16)

    def forward_device(self, x_tt) -> "object":
        """Device-only path. x_tt: replicated input. Returns replicated output
        (full out_dim on every chip) via all_gather along the sharded axis."""
        ttnn = self._ttnn
        y_sharded = ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.all_gather(y_sharded, dim=-1, num_links=1)


class DeviceRowLinear(nn.Module):
    """Row-parallel linear across a 1xN mesh with all-reduce sum.

    Weight: [out, in] (nn.Linear order). Transposed to [in, out] for ttnn,
    row-sharded on the input dim: each chip holds [in/N, out]. Input must
    be sharded on its last dim (in/N per chip); we shard it on upload.
    Output is all-reduced (sum) across the mesh and returned as bf16.
    """

    def __init__(self, mesh, cpu_weight: torch.Tensor):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        out_dim, in_dim = cpu_weight.shape
        N = self.mesh_shape[1]
        if in_dim % N != 0:
            raise ValueError(
                f"row-linear in_dim {in_dim} not divisible by mesh axis 1 {N}"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        w_io = _weight_to_bf16(cpu_weight).transpose(0, 1).contiguous()  # [in, out]
        self.w_tt = ttnn.as_tensor(
            w_io,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, self.mesh_shape, dims=(None, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ttnn = self._ttnn
        B, S, D = x.shape
        x_tt = ttnn.as_tensor(
            x.to(torch.bfloat16).contiguous(),
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, self.mesh_shape, dims=(None, -1)),
        )
        y_tt = ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Bring N partial outputs back and sum. Mesh axis 0 has size 1 so its
        # concat dim is a no-op; route it to tensor dim 1 and stack axis-1
        # shards along tensor dim 0 so y.shape[0] = N*B.
        y = ttnn.to_torch(
            y_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0)),
        )
        N = self.mesh_shape[1]
        y = y.view(N, B, S, self.out_dim).sum(dim=0)
        return y.to(torch.bfloat16)


class DeviceGroupedLinear(nn.Module):
    """Block-diagonal (per-group) linear across a 1xN mesh.

    For wo_a: weight is [n_groups*out_per_group, in_per_group] where each group
    g has its own [out_per_group, in_per_group] matrix; activations are
    [B, S, n_groups, in_per_group]. Computes
        out[b,s,g,r] = sum_d x[b,s,g,d] * w[g,r,d]
    i.e. the einsum "bsgd,grd->bsgr".

    Sharding: groups are distributed across mesh axis 1 (must divide evenly).
    Each chip holds its groups' weights AND sees only its groups' activations
    -- no wasted compute. Output is gathered and permuted back to
    [B, S, n_groups, out_per_group].
    """

    def __init__(self, mesh, cpu_weight: torch.Tensor, n_groups: int):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        N = self.mesh_shape[1]
        if n_groups % N != 0:
            raise ValueError(f"n_groups {n_groups} not divisible by mesh axis 1 {N}")
        out_total, in_per_group = cpu_weight.shape
        if out_total % n_groups != 0:
            raise ValueError(f"weight out_dim {out_total} not divisible by n_groups {n_groups}")
        out_per_group = out_total // n_groups
        self.n_groups = n_groups
        self.in_per_group = in_per_group
        self.out_per_group = out_per_group

        w_bf16 = _weight_to_bf16(cpu_weight).view(n_groups, out_per_group, in_per_group)
        # Transpose to [n_groups, in_per_group, out_per_group] for A @ B matmul.
        w_gio = w_bf16.transpose(1, 2).contiguous()
        self.w_tt = ttnn.as_tensor(
            w_gio,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, self.mesh_shape, dims=(None, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, n_groups, in_per_group] -> [B, S, n_groups, out_per_group]."""
        ttnn = self._ttnn
        B, S, G, D = x.shape
        if G != self.n_groups or D != self.in_per_group:
            raise ValueError(
                f"x shape ({B},{S},{G},{D}) mismatch "
                f"expected groups={self.n_groups} in_per_group={self.in_per_group}"
            )
        # [B,S,G,D] -> [G, B*S, D] so each chip's slice on dim 0 = its groups.
        x_g = x.to(torch.bfloat16).permute(2, 0, 1, 3).reshape(G, B * S, D).contiguous()
        x_tt = ttnn.as_tensor(
            x_g,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, self.mesh_shape, dims=(None, 0)),
        )
        y_tt = ttnn.linear(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Each chip holds [G/N, B*S, out_per_group]; stack along mesh axis 1 onto
        # tensor dim 0 (ConcatMesh2dToTensor dims=(1, 0) -- mesh axis 0 is size 1,
        # routes to tensor dim 1 no-op; axis 1 stacks on dim 0).
        y = ttnn.to_torch(
            y_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0)),
        )
        y = y[:G]  # trim any trailing pad
        y = y.view(G, B, S, self.out_per_group).permute(1, 2, 0, 3).contiguous()
        return y.to(torch.bfloat16)


class DeviceMoEGate(nn.Module):
    """V4-Flash MoE gate (sqrtsoftplus score_func) on a 1xN mesh.

    Replicates the tiny gate weight (n_experts x dim bf16, ~2MB) on every chip
    and runs the full gate pipeline with ttnn primitives:

        scores  = sqrt(softplus(x @ W^T))
        indices = topk(scores + bias, k=topk)
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(-1, keepdim)       # non-softmax branch
        weights = weights * route_scale

    Output is replicated; read back via a composer that stacks 4 copies on
    tensor dim 0 and the first M rows are chip-0's result.
    """

    def __init__(self, mesh, cpu_weight: torch.Tensor, cpu_bias: torch.Tensor,
                 topk: int, route_scale: float, score_func: str):
        super().__init__()
        import ttnn
        if score_func != "sqrtsoftplus":
            raise ValueError(
                f"DeviceMoEGate v1 only supports score_func='sqrtsoftplus', got {score_func!r}"
            )
        if cpu_bias is None:
            raise ValueError("sqrtsoftplus gate requires bias; got None")
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        n_experts, dim = cpu_weight.shape
        self.n_experts = n_experts
        self.dim = dim
        self.topk = topk
        self.route_scale = float(route_scale)

        w_kn = _weight_to_bf16(cpu_weight).transpose(0, 1).contiguous()  # [dim, n_experts]
        b_row = cpu_bias.to(torch.bfloat16).view(1, n_experts).contiguous()
        common_rep = dict(
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.w_tt = ttnn.as_tensor(w_kn, **common_rep)
        self.bias_tt = ttnn.as_tensor(b_row, **common_rep)

    def forward(self, x: torch.Tensor):
        """x: [M, dim] -> (weights [M, topk] float32, indices [M, topk] int64)."""
        ttnn = self._ttnn
        M, D = x.shape
        if D != self.dim:
            raise ValueError(f"x last dim {D} mismatch gate dim {self.dim}")
        x_tt = ttnn.as_tensor(
            x.to(torch.bfloat16).contiguous(),
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        raw_tt = ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores_tt = ttnn.sqrt(ttnn.softplus(raw_tt))
        biased_tt = ttnn.add(scores_tt, self.bias_tt)
        _, indices_tt = ttnn.topk(biased_tt, k=self.topk, dim=-1, largest=True, sorted=True)
        gathered_tt = ttnn.gather(scores_tt, dim=-1, index=indices_tt)
        wsum_tt = ttnn.sum(gathered_tt, dim=-1, keepdim=True)
        normed_tt = ttnn.div(gathered_tt, wsum_tt)
        weights_tt = ttnn.multiply(normed_tt, self.route_scale)

        # Replicated readback: stack 4 copies on tensor dim 0, take first M rows.
        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        weights = ttnn.to_torch(weights_tt, mesh_composer=composer)[:M].float()
        indices = ttnn.to_torch(indices_tt, mesh_composer=composer)[:M].long()
        return weights, indices


_RMS_TILE = 32


def _pack_rms_gamma(gamma_1d: torch.Tensor) -> torch.Tensor:
    """[hidden] -> [TILE, hidden] with gamma replicated across all 32 rows."""
    (hidden,) = gamma_1d.shape
    if hidden % _RMS_TILE:
        raise ValueError(f"rms hidden={hidden} must be multiple of {_RMS_TILE}")
    return gamma_1d.unsqueeze(0).expand(_RMS_TILE, -1).to(torch.bfloat16).contiguous()


import ttl  # type: ignore[import-not-found]


def _compile_rmsnorm_kernel(num_row_tiles: int, h_tiles: int, rms_eps: float, inv_D: float):
    """Inlined from tt-lang-kernels/rmsnorm.py. Streams one row-tile (32 tokens)
    per core-iteration. Two passes over x per row-tile: sum(x^2), then
    gamma*inv_rms apply. Requires module-level `ttl` so the @ttl.operation
    decorator's source inspector sees it as a global (not a closure var)."""

    @ttl.operation(
        grid="auto",
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def rmsnorm_kernel(x, gamma, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_row_tiles // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        g_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        xsq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        red_step_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            sc = sc_dfb.wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    x0 = x_dfb.wait()
                    xsq_dfb.reserve().store(x0 * x0)
                    sq_dfb.reserve().store(
                        ttl.math.reduce_sum(xsq_dfb.wait(), sc, dims=[1])
                    )
                    for _ in range(h_tiles - 1):
                        xk = x_dfb.wait()
                        xsq_dfb.reserve().store(xk * xk)
                        red_step_dfb.reserve().store(
                            ttl.math.reduce_sum(xsq_dfb.wait(), sc, dims=[1])
                        )
                        prev = sq_dfb.wait()
                        sq_dfb.reserve().store(prev + red_step_dfb.wait())

                    sq = sq_dfb.wait()
                    inv_bc = inv_bc_dfb.reserve()
                    inv_bc.store(ttl.math.broadcast(
                        ttl.math.rsqrt(
                            sq * ttl.math.fill(sq, inv_D) + ttl.math.fill(sq, rms_eps)
                        ),
                        inv_bc, dims=[1],
                    ))
                    inv = inv_bc_dfb.wait()

                    for _ in range(h_tiles):
                        xk = x_dfb.wait()
                        gk = g_dfb.wait()
                        out_dfb.reserve().store(xk * gk * inv)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()
                        ttl.copy(gamma[0, h], g_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for h in range(h_tiles):
                        ttl.copy(out_dfb.wait(), out[global_t, h]).wait()

    return rmsnorm_kernel


class DeviceRMSNorm(nn.Module):
    """RMSNorm via the tt-lang fused kernel on a 1xN mesh (replicated gamma).

    Pads M to a multiple of TILE (32). Gamma is packed to [TILE, hidden] and
    replicated across every chip. Each chip runs the kernel on its replicated
    inputs -- output is the same on every chip; we read chip 0's rows.
    Per-(M_tiles) kernel is compiled lazily and cached.
    """

    def __init__(self, mesh, cpu_gamma: torch.Tensor, eps: float):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        g = cpu_gamma.to(torch.bfloat16).flatten().contiguous()
        (hidden,) = g.shape
        self.hidden = hidden
        self.eps = float(eps)
        self._kernels: dict = {}

        rep = dict(
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.gamma_tt = ttnn.as_tensor(_pack_rms_gamma(g), **rep)
        self.sc_tt = ttnn.as_tensor(
            torch.ones((_RMS_TILE, _RMS_TILE), dtype=torch.bfloat16), **rep)

    def _kernel(self, num_row_tiles: int):
        k = self._kernels.get(num_row_tiles)
        if k is None:
            k = _compile_rmsnorm_kernel(
                num_row_tiles=num_row_tiles,
                h_tiles=self.hidden // _RMS_TILE,
                rms_eps=self.eps,
                inv_D=1.0 / self.hidden,
            )
            self._kernels[num_row_tiles] = k
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ttnn = self._ttnn
        orig_shape = x.shape
        x2 = x.reshape(-1, self.hidden).to(torch.bfloat16)
        M = x2.shape[0]
        Mt = -(-M // _RMS_TILE)
        Mpad = Mt * _RMS_TILE
        if M < Mpad:
            pad = torch.zeros((Mpad - M, self.hidden), dtype=torch.bfloat16)
            x2 = torch.cat([x2, pad], dim=0)

        rep = dict(
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        x_tt = ttnn.as_tensor(x2.contiguous(), **rep)
        out_tt = self.forward_device(x_tt, M)

        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        y = ttnn.to_torch(out_tt, mesh_composer=composer)[:M]
        return y.view(*orig_shape).to(x.dtype)

    def forward_device(self, x_tt, num_rows: int):
        """Device-only path: x_tt must be a [Mpad, hidden] device tensor with
        Mpad rows (TILE-aligned). Returns a same-shape device tensor. Caller
        is responsible for any padding and for slicing rows back to num_rows.

        num_rows is the unpadded row count and only selects the kernel
        variant (M_tiles = ceil(num_rows / TILE)).
        """
        ttnn = self._ttnn
        Mt = -(-num_rows // _RMS_TILE)
        out_tt = ttnn.zeros(
            shape=tuple(x_tt.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._kernel(Mt)(x_tt, self.gamma_tt, self.sc_tt, out_tt)
        return out_tt


# ============================================================================
# MHC (hyper-connection mixing) — tt-lang fused kernels on a 1xN mesh
# ============================================================================

_MHC_TILE = 32


def _mhc_pack_residual(residual: torch.Tensor, num_tokens_pad: int) -> torch.Tensor:
    """[B, S, mhc, hidden] -> [num_tokens_pad, D=mhc*hidden] fp32, zero-padded rows."""
    n0, n1, mhc, hidden = residual.shape
    num_tokens = n0 * n1
    D = mhc * hidden
    flat = residual.reshape(num_tokens, D).to(torch.float32).contiguous()
    if num_tokens < num_tokens_pad:
        pad = torch.zeros(num_tokens_pad - num_tokens, D, dtype=torch.float32)
        flat = torch.cat([flat, pad], dim=0)
    return flat.contiguous()


def _mhc_pack_fn(fn: torch.Tensor, mhc_mult3: int) -> torch.Tensor:
    """[mhc_mult3, D] -> [D, TILE] fp32 (fn^T padded to TILE cols)."""
    m3, D = fn.shape
    if m3 != mhc_mult3:
        raise ValueError(f"hc_fn rows {m3} != mhc_mult3 {mhc_mult3}")
    out = torch.zeros(D, _MHC_TILE, dtype=torch.float32)
    out[:, :m3] = fn.T.to(torch.float32)
    return out.contiguous()


def _mhc_pack_x_for_apply_mix(x: torch.Tensor, num_tokens_pad: int) -> torch.Tensor:
    """[num_tokens, mhc, h] -> [num_tokens_pad * TILE, h] fp32 (rows 0..mhc-1 hold x; rest 0)."""
    num_tokens, mhc, h = x.shape
    out = torch.zeros(num_tokens_pad * _MHC_TILE, h, dtype=torch.float32)
    out.view(num_tokens_pad, _MHC_TILE, h)[:num_tokens, :mhc, :] = x.to(torch.float32)
    return out.contiguous()


def _mhc_pack_mix(mix: torch.Tensor, num_tokens_pad: int) -> torch.Tensor:
    """[num_tokens, mhc] -> [num_tokens_pad * TILE, TILE] fp32 (col 0 of rows 0..mhc-1)."""
    num_tokens, mhc = mix.shape
    out = torch.zeros(num_tokens_pad * _MHC_TILE, _MHC_TILE, dtype=torch.float32)
    out.view(num_tokens_pad, _MHC_TILE, _MHC_TILE)[:num_tokens, :mhc, 0] = mix.to(torch.float32)
    return out.contiguous()


def _mhc_pack_x_bc(x: torch.Tensor, mhc: int, num_tokens_pad: int) -> torch.Tensor:
    """[num_tokens, h] -> [num_tokens_pad * TILE, h] fp32 (rows 0..mhc-1 each hold a copy of x)."""
    num_tokens, h = x.shape
    src = x.unsqueeze(1).expand(-1, mhc, -1)
    out = torch.zeros(num_tokens_pad * _MHC_TILE, h, dtype=torch.float32)
    out.view(num_tokens_pad, _MHC_TILE, h)[:num_tokens, :mhc, :] = src.to(torch.float32)
    return out.contiguous()


def _mhc_pack_residual_for_post(residual: torch.Tensor, num_tokens_pad: int) -> torch.Tensor:
    """[num_tokens, mhc, h] -> [num_tokens_pad * TILE, h] fp32 (rows 0..mhc-1 hold residual)."""
    num_tokens, mhc, h = residual.shape
    out = torch.zeros(num_tokens_pad * _MHC_TILE, h, dtype=torch.float32)
    out.view(num_tokens_pad, _MHC_TILE, h)[:num_tokens, :mhc, :] = residual.to(torch.float32)
    return out.contiguous()


def _mhc_pack_comb_T(comb: torch.Tensor, num_tokens_pad: int) -> torch.Tensor:
    """[num_tokens, mhc, mhc] -> [num_tokens_pad * TILE, TILE] fp32 (comb^T in top-left)."""
    num_tokens, m, n = comb.shape
    if m != n:
        raise ValueError(f"comb must be square, got {m}x{n}")
    out = torch.zeros(num_tokens_pad * _MHC_TILE, _MHC_TILE, dtype=torch.float32)
    out.view(num_tokens_pad, _MHC_TILE, _MHC_TILE)[:num_tokens, :m, :m] = comb.transpose(-1, -2).to(torch.float32)
    return out.contiguous()


def _mhc_pack_post_mix(post: torch.Tensor, num_tokens_pad: int) -> torch.Tensor:
    """[num_tokens, mhc] -> [num_tokens_pad * TILE, TILE] fp32 (col 0 of rows 0..mhc-1)."""
    return _mhc_pack_mix(post, num_tokens_pad)


def _mhc_unpack_apply_mix_out(packed: torch.Tensor, num_tokens: int, h: int) -> torch.Tensor:
    """[num_tokens_pad * TILE, h] -> [num_tokens, h] (row 0 of each block)."""
    return packed.view(-1, _MHC_TILE, h)[:num_tokens, 0, :].contiguous()


def _mhc_unpack_post_out(packed: torch.Tensor, num_tokens: int, mhc: int, h: int) -> torch.Tensor:
    """[num_tokens_pad * TILE, h] -> [num_tokens, mhc, h] (rows 0..mhc-1 of each block)."""
    return packed.view(-1, _MHC_TILE, h)[:num_tokens, :mhc, :].contiguous()


def _compile_mhc_norm_fn_kernel(num_out_tiles: int, K_tiles: int, rms_eps: float, inv_D: float):
    """Inlined from tt-lang-kernels/pre_norm_fn.py.

    Computes per-token: mixes[t, m] = (residual[t] @ fn[m, :]) * rsqrt(sum(residual[t]^2)/D + eps).
    Uses ping-pong store(prev + a@b) accumulator (ksplit `c += a @ b` blocked
    inside conditional bounds by tt-lang#504); accuracy degrades modestly at
    K>>1. PCC ~0.998 at V4-Flash D=16384 (K=512); acceptable for now.
    """

    @ttl.operation(
        grid="auto",
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def norm_fn_kernel(a, b, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_out_tiles // total_cores)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        c_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        sq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        asq_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        red_step_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_out_tiles:
                    a0 = a_dfb.wait()
                    b0 = b_dfb.wait()
                    c_dfb.reserve().store(a0 @ b0)
                    asq_dfb.reserve().store(a0 * a0)
                    sq_dfb.reserve().store(
                        ttl.math.reduce_sum(asq_dfb.wait(), sc, dims=[1])
                    )

                    for _ in range(K_tiles - 1):
                        a = a_dfb.wait()
                        b = b_dfb.wait()
                        prev_c = c_dfb.wait()
                        c_dfb.reserve().store(prev_c + a @ b)

                        asq_dfb.reserve().store(a * a)
                        red_step_dfb.reserve().store(
                            ttl.math.reduce_sum(asq_dfb.wait(), sc, dims=[1])
                        )
                        prev_sq = sq_dfb.wait()
                        sq_dfb.reserve().store(prev_sq + red_step_dfb.wait())

                    sq = sq_dfb.wait()
                    inv_bc = inv_bc_dfb.reserve()
                    inv_bc.store(ttl.math.broadcast(
                        ttl.math.rsqrt(
                            sq * ttl.math.fill(sq, inv_D) + ttl.math.fill(sq, rms_eps)
                        ),
                        inv_bc, dims=[1],
                    ))

                    c = c_dfb.wait()
                    out_dfb.reserve().store(c * inv_bc_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_out_tiles:
                    for k in range(K_tiles):
                        ttl.copy(a[global_t, k], a_dfb.reserve()).wait()
                        ttl.copy(b[k, 0], b_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_out_tiles:
                    ttl.copy(out_dfb.wait(), out[global_t, 0]).wait()

    return norm_fn_kernel


def _compile_mhc_apply_mix_kernel(num_tokens: int, h_tiles: int):
    """Inlined from tt-lang-kernels/pre_apply_mix.py.

    Per-token: out[h] = sum_m x[m, :] * mix[m] (column sum across rows 0..mhc-1).
    Broadcast mix col-0 across all cols, multiply tile, reduce_sum dim=0 per h-tile.
    """

    @ttl.operation(grid="auto")
    def apply_mix_kernel(x, mix, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tokens_per_core = -(-num_tokens // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        mix_dfb = ttl.make_dataflow_buffer_like(mix, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        mix_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        prod_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    mix_raw = mix_dfb.wait()
                    mx = mix_bc_dfb.reserve()
                    mx.store(ttl.math.broadcast(mix_raw, mx, dims=[1]))
                    mix_bc = mix_bc_dfb.wait()

                    for _ in range(h_tiles):
                        x_tile = x_dfb.wait()
                        prod_dfb.reserve().store(x_tile * mix_bc)
                        red_dfb.reserve().store(
                            ttl.math.reduce_sum(prod_dfb.wait(), sc, dims=[0])
                        )
                        out_dfb.reserve().store(red_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    ttl.copy(mix[global_t, 0], mix_dfb.reserve()).wait()
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    for h in range(h_tiles):
                        ttl.copy(out_dfb.wait(), out[global_t, h]).wait()

    return apply_mix_kernel


def _compile_mhc_post_kernel(num_tokens: int, h_tiles: int):
    """Inlined from tt-lang-kernels/post.py.

    Per-token, per-h-tile: out = x * post_mix_bc + comb^T @ residual.
    `comb` is stored pre-transposed so a 32x32 tile-matmul produces the right
    row layout; `post_mix` is broadcast from col 0 across all cols.
    """

    @ttl.operation(grid="auto")
    def post_kernel(x, residual, comb_T, post_mix, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tokens_per_core = -(-num_tokens // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, 1), block_count=2)
        comb_dfb = ttl.make_dataflow_buffer_like(comb_T, shape=(1, 1), block_count=2)
        post_dfb = ttl.make_dataflow_buffer_like(post_mix, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        post_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        post_term_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        matmul_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    post = post_dfb.wait()
                    comb_t = comb_dfb.wait()

                    pbc = post_bc_dfb.reserve()
                    pbc.store(ttl.math.broadcast(post, pbc, dims=[1]))
                    post_bc = post_bc_dfb.wait()

                    for _ in range(h_tiles):
                        x_tile = x_dfb.wait()
                        res_tile = res_dfb.wait()
                        post_term_dfb.reserve().store(x_tile * post_bc)
                        matmul_dfb.reserve().store(comb_t @ res_tile)
                        out_dfb.reserve().store(
                            post_term_dfb.wait() + matmul_dfb.wait()
                        )

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    ttl.copy(post_mix[global_t, 0], post_dfb.reserve()).wait()
                    ttl.copy(comb_T[global_t, 0], comb_dfb.reserve()).wait()
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()
                        ttl.copy(residual[global_t, h], res_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tokens_per_core):
                global_t = core_idx * tokens_per_core + local_t
                if global_t < num_tokens:
                    for h in range(h_tiles):
                        ttl.copy(out_dfb.wait(), out[global_t, h]).wait()

    return post_kernel


class DeviceMHC(nn.Module):
    """MHC pre + post on a 1xN mesh via tt-lang fused kernels (fp32 throughout).

    Pipeline (per Block.hc_pre call):
      1. pre_norm_fn: [num_tokens, D] @ fn^T * inv_rms -> mixes [num_tokens, mhc_mult3]
      2. CPU split + sinkhorn: mixes -> (pre, post, comb)
      3. pre_apply_mix: x[num_tokens, mhc, h] reduced by pre[num_tokens, mhc, 1] -> y[num_tokens, h]

    Pipeline (per Block.hc_post call):
      1. post: x[num_tokens, h] * post_mix + comb^T @ residual -> [num_tokens, mhc, h]

    Sinkhorn lives on CPU because it's a tight iterative loop on a tiny tensor
    ([num_tokens, mhc, mhc]); the device split_mixes kernel would only save a
    trivial elementwise op while requiring a host round-trip anyway for the
    iterative normalization.

    Decode pads num_tokens to TILE rows (1 token -> 32 padded rows) so the
    kernels' tiles-per-core math holds. Padded rows hold zeros and contribute
    nothing to the read-back range.
    """

    def __init__(self, mesh, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                 hc_base: torch.Tensor, hc_mult: int, hc_eps: float,
                 sinkhorn_iters: int, norm_eps: float):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        self.hc_mult = int(hc_mult)
        self.mhc_mult3 = (2 + self.hc_mult) * self.hc_mult
        if hc_fn.shape[0] != self.mhc_mult3:
            raise ValueError(
                f"hc_fn rows {hc_fn.shape[0]} != mhc_mult3 {self.mhc_mult3}")
        self.D = hc_fn.shape[-1]
        if self.D % self.hc_mult:
            raise ValueError(f"D={self.D} not divisible by hc_mult={self.hc_mult}")
        self.hidden = self.D // self.hc_mult
        if self.hidden % _MHC_TILE:
            raise ValueError(f"hidden {self.hidden} must be multiple of {_MHC_TILE}")
        self.norm_eps = float(norm_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        # CPU copies of hc_scale/hc_base for the split + sinkhorn step.
        self.hc_scale_cpu = hc_scale.detach().to(torch.float32).cpu()
        self.hc_base_cpu = hc_base.detach().to(torch.float32).cpu()

        rep = dict(
            device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        fn_packed = _mhc_pack_fn(
            hc_fn.detach().to(torch.float32).cpu(), self.mhc_mult3)
        self.fn_tt = ttnn.as_tensor(fn_packed, **rep)
        self.scaler_tt = ttnn.as_tensor(
            torch.ones((_MHC_TILE, _MHC_TILE), dtype=torch.float32), **rep)

        self._kernels: dict = {}

    def _norm_fn_kernel(self, num_out_tiles: int):
        key = ("norm_fn", num_out_tiles)
        k = self._kernels.get(key)
        if k is None:
            k = _compile_mhc_norm_fn_kernel(
                num_out_tiles=num_out_tiles,
                K_tiles=self.D // _MHC_TILE,
                rms_eps=self.norm_eps,
                inv_D=1.0 / self.D,
            )
            self._kernels[key] = k
        return k

    def _apply_mix_kernel(self, num_tokens_pad: int):
        key = ("apply_mix", num_tokens_pad)
        k = self._kernels.get(key)
        if k is None:
            k = _compile_mhc_apply_mix_kernel(
                num_tokens=num_tokens_pad,
                h_tiles=self.hidden // _MHC_TILE,
            )
            self._kernels[key] = k
        return k

    def _post_kernel(self, num_tokens_pad: int):
        key = ("post", num_tokens_pad)
        k = self._kernels.get(key)
        if k is None:
            k = _compile_mhc_post_kernel(
                num_tokens=num_tokens_pad,
                h_tiles=self.hidden // _MHC_TILE,
            )
            self._kernels[key] = k
        return k

    def _rep_tensor(self, t: torch.Tensor):
        ttnn = self._ttnn
        return ttnn.as_tensor(
            t,
            device=self.mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _zeros(self, shape):
        ttnn = self._ttnn
        return ttnn.zeros(
            shape=tuple(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def hc_pre(self, x: torch.Tensor):
        """[B, S, mhc, hidden] -> (y [B, S, hidden], post [B, S, mhc], comb [B, S, mhc, mhc])."""
        ttnn = self._ttnn
        B, S, mhc, hidden = x.shape
        if mhc != self.hc_mult or hidden != self.hidden:
            raise ValueError(
                f"x shape mismatch: got mhc={mhc}, hidden={hidden}; "
                f"expected mhc={self.hc_mult}, hidden={self.hidden}")
        out_dtype = x.dtype
        num_tokens = B * S
        num_tokens_pad = -(-num_tokens // _MHC_TILE) * _MHC_TILE

        # Step 1: pre_norm_fn on device.
        a_packed = _mhc_pack_residual(x, num_tokens_pad)
        a_tt = self._rep_tensor(a_packed)
        mixes_tt = self._zeros((num_tokens_pad, _MHC_TILE))
        self._norm_fn_kernel(num_tokens_pad // _MHC_TILE)(
            a_tt, self.fn_tt, self.scaler_tt, mixes_tt)

        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        mixes_packed = ttnn.to_torch(mixes_tt, mesh_composer=composer)
        # Take chip 0's row range (replicated across chips), unpack to [num_tokens, mhc_mult3].
        mixes = mixes_packed[:num_tokens, :self.mhc_mult3].reshape(B, S, self.mhc_mult3)

        # Step 2: CPU split + sinkhorn.
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.hc_scale_cpu, self.hc_base_cpu,
            self.hc_mult, self.sinkhorn_iters, self.hc_eps,
        )

        # Step 3: pre_apply_mix on device.
        x_3d = x.view(num_tokens, mhc, hidden)
        x_packed = _mhc_pack_x_for_apply_mix(x_3d, num_tokens_pad)
        mix_packed = _mhc_pack_mix(
            pre.reshape(num_tokens, mhc), num_tokens_pad)
        x_tt = self._rep_tensor(x_packed)
        mix_tt = self._rep_tensor(mix_packed)
        out_tt = self._zeros((num_tokens_pad * _MHC_TILE, hidden))
        self._apply_mix_kernel(num_tokens_pad)(
            x_tt, mix_tt, self.scaler_tt, out_tt)

        out_packed = ttnn.to_torch(out_tt, mesh_composer=composer)
        y_flat = _mhc_unpack_apply_mix_out(out_packed, num_tokens, hidden)
        y = y_flat.view(B, S, hidden).to(out_dtype)
        return y, post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor,
                post: torch.Tensor, comb: torch.Tensor):
        """x [B, S, hidden], residual [B, S, mhc, hidden], post [B, S, mhc],
        comb [B, S, mhc, mhc] -> [B, S, mhc, hidden]."""
        ttnn = self._ttnn
        B, S, hidden = x.shape
        mhc = self.hc_mult
        out_dtype = x.dtype
        num_tokens = B * S
        num_tokens_pad = -(-num_tokens // _MHC_TILE) * _MHC_TILE

        x_2d = x.reshape(num_tokens, hidden)
        res_3d = residual.reshape(num_tokens, mhc, hidden)
        post_2d = post.reshape(num_tokens, mhc)
        comb_3d = comb.reshape(num_tokens, mhc, mhc)

        x_tt = self._rep_tensor(_mhc_pack_x_bc(x_2d, mhc, num_tokens_pad))
        res_tt = self._rep_tensor(_mhc_pack_residual_for_post(res_3d, num_tokens_pad))
        comb_tt = self._rep_tensor(_mhc_pack_comb_T(comb_3d, num_tokens_pad))
        post_tt = self._rep_tensor(_mhc_pack_post_mix(post_2d, num_tokens_pad))
        out_tt = self._zeros((num_tokens_pad * _MHC_TILE, hidden))
        self._post_kernel(num_tokens_pad)(
            x_tt, res_tt, comb_tt, post_tt, out_tt)

        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        out_packed = ttnn.to_torch(out_tt, mesh_composer=composer)
        y_flat = _mhc_unpack_post_out(out_packed, num_tokens, mhc, hidden)
        return y_flat.view(B, S, mhc, hidden).to(out_dtype)


class DeviceSharedExpert(nn.Module):
    """V4-Flash shared-expert SwiGLU on a 1xN mesh (replicated weights).

    Each chip holds full copies of w1, w2, w3 (~48MB per layer bf16) and
    runs the full forward:

        y1 = x @ w1^T
        y3 = x @ w3^T
        y1 = clamp(y1, max=limit);  y3 = clamp(y3, -limit, +limit)
        mid = silu(y1) * y3
        out = mid @ w2^T

    Activations are replicated; output read back from chip 0. This is
    correctness-first TP (no CCL); compute is duplicated 4x but the
    shared expert is small relative to the mesh's aggregate throughput.
    """

    def __init__(self, mesh, cpu_w1: torch.Tensor, cpu_w2: torch.Tensor,
                 cpu_w3: torch.Tensor, swiglu_limit: float):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        inter_dim, dim = cpu_w1.shape
        if cpu_w3.shape != (inter_dim, dim):
            raise ValueError(f"w3 shape {tuple(cpu_w3.shape)} != w1 {(inter_dim, dim)}")
        if cpu_w2.shape != (dim, inter_dim):
            raise ValueError(f"w2 shape {tuple(cpu_w2.shape)} != expected {(dim, inter_dim)}")
        self.dim = dim
        self.inter_dim = inter_dim
        self.swiglu_limit = float(swiglu_limit)

        common_rep = dict(
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.w1_tt = ttnn.as_tensor(
            _weight_to_bf16(cpu_w1).transpose(0, 1).contiguous(), **common_rep)
        self.w3_tt = ttnn.as_tensor(
            _weight_to_bf16(cpu_w3).transpose(0, 1).contiguous(), **common_rep)
        self.w2_tt = ttnn.as_tensor(
            _weight_to_bf16(cpu_w2).transpose(0, 1).contiguous(), **common_rep)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [M, dim] -> [M, dim]. `weights` kwarg matches Expert.forward signature;
        shared expert is always called with weights=None."""
        if weights is not None:
            raise ValueError("DeviceSharedExpert does not support per-token `weights`")
        ttnn = self._ttnn
        M, D = x.shape
        if D != self.dim:
            raise ValueError(f"x last dim {D} != expected {self.dim}")
        x_tt = ttnn.as_tensor(
            x.to(torch.bfloat16).contiguous(),
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        y1_tt = ttnn.matmul(x_tt, self.w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y3_tt = ttnn.matmul(x_tt, self.w3_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.swiglu_limit > 0:
            y1_tt = ttnn.clamp(y1_tt, max=self.swiglu_limit)
            y3_tt = ttnn.clamp(y3_tt, min=-self.swiglu_limit, max=self.swiglu_limit)
        mid_tt = ttnn.multiply(ttnn.silu(y1_tt), y3_tt)
        out_tt = ttnn.matmul(mid_tt, self.w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        y = ttnn.to_torch(out_tt, mesh_composer=composer)[:M]
        return y.to(x.dtype)


class DeviceSparseAttn(nn.Module):
    """V4-Flash sparse_attn on a 1xN mesh (replicated attn_sink).

    Drop-in replacement for `sparse_attn(q, kv, attn_sink, topk_idxs, scale)`.
    Inputs uploaded per-call (replicated), output read back from chip 0.

    Compute is identical to the CPU reference: gather kv at topk indices via
    ttnn.embedding, score matmul, additive mask for `-1` slots, concat sink,
    softmax, drop sink, weighted sum. ttnn.embedding sees [N, D] kv (batch=1
    squeezed) and uint32 indices.
    """

    def __init__(self, mesh, attn_sink: torch.Tensor, softmax_scale: float):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        self.softmax_scale = float(softmax_scale)
        self.n_heads = attn_sink.shape[0]
        sink_4d = attn_sink.to(torch.bfloat16).view(1, 1, self.n_heads, 1).contiguous()
        self.sink_tt = ttnn.as_tensor(
            sink_4d,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        ttnn = self._ttnn
        B, S, H, D = q.shape
        if H != self.n_heads:
            raise ValueError(f"q heads {H} != attn_sink heads {self.n_heads}")
        if B != 1:
            raise ValueError(f"DeviceSparseAttn only supports B=1, got B={B}")

        common_rep = dict(
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        q_tt = ttnn.as_tensor(q.to(torch.bfloat16).contiguous(), **common_rep)
        kv_tt = ttnn.as_tensor(kv.squeeze(0).to(torch.bfloat16).contiguous(), **common_rep)
        idxs_tt, valid_tt = self._upload_topk(topk_idxs)

        out_tt = self.forward_device(q_tt, kv_tt, idxs_tt, valid_tt, S, topk_idxs.shape[-1])
        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        out = ttnn.to_torch(out_tt, mesh_composer=composer)[:B]
        return out.to(q.dtype)

    def _upload_topk(self, topk_idxs: torch.Tensor):
        """Upload topk_idxs and the additive -inf mask. Returns (idxs_tt uint32
        [B, S*K] row-major, valid_tt bf16 [B, S, 1, K] tile)."""
        ttnn = self._ttnn
        B, S, K = topk_idxs.shape
        safe = topk_idxs.clamp_min(0).reshape(B, S * K).to(torch.int32).contiguous()
        idxs_tt = ttnn.as_tensor(
            safe,
            device=self.mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        valid = torch.where(
            topk_idxs >= 0,
            torch.zeros((), dtype=torch.bfloat16),
            torch.tensor(float("-inf"), dtype=torch.bfloat16),
        ).view(B, S, 1, K).contiguous()
        valid_tt = ttnn.as_tensor(
            valid,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        return idxs_tt, valid_tt

    def forward_device(self, q_tt, kv_tt, idxs_tt, valid_tt, S: int, K: int):
        """Device-only sparse_attn. q_tt: [1,S,H,D] tile; kv_tt: [N,D] tile;
        idxs_tt: [1, S*K] uint32 row-major; valid_tt: [1,S,1,K] tile additive
        mask. Returns o_tt [1,S,H,D] tile (replicated)."""
        ttnn = self._ttnn
        B = 1
        H = self.n_heads
        D = tuple(kv_tt.shape)[-1]
        if S == 1:
            sink_for_concat = self.sink_tt
        else:
            sink_for_concat = ttnn.repeat(self.sink_tt, ttnn.Shape([1, S, 1, 1]))

        kv_gather = ttnn.embedding(idxs_tt, kv_tt, layout=ttnn.TILE_LAYOUT)
        kv_gather = ttnn.reshape(kv_gather, [B, S, K, D])

        kv_gather_t = ttnn.transpose(kv_gather, -2, -1)
        scores = ttnn.matmul(q_tt, kv_gather_t)
        scores = ttnn.multiply(scores, self.softmax_scale)
        scores = ttnn.add(scores, valid_tt)

        full = ttnn.concat([scores, sink_for_concat], dim=-1)
        probs_full = ttnn.softmax(full, dim=-1)
        probs = ttnn.slice(probs_full, [0, 0, 0, 0], [B, S, H, K])
        return ttnn.matmul(probs, kv_gather)


def _device_apply_rotary_interleaved(ttnn, x_tt, cos_pair_tt, sin_pair_tt, inverse: bool = False):
    """Manual interleaved rotary on a device tensor.

    Matches V4-Flash's view_as_complex format: x[..., rd] is reshaped to
    [..., rd/2, 2] where the trailing 2 holds (real, imag) of one rotary
    pair; cos_pair / sin_pair are [..., rd/2] broadcast tables.

      forward:  (a, b) -> (a*cos - b*sin, a*sin + b*cos)
      inverse:  (a, b) -> (a*cos + b*sin, -a*sin + b*cos)
    """
    out_shape = list(x_tt.shape)
    rd = out_shape[-1]
    pair_shape = out_shape[:-1] + [rd // 2, 2]
    x_pairs = ttnn.reshape(x_tt, pair_shape)
    pre = [0] * (len(pair_shape) - 1)
    real = ttnn.slice(x_pairs, pre + [0], list(pair_shape[:-1]) + [1])
    imag = ttnn.slice(x_pairs, pre + [1], list(pair_shape[:-1]) + [2])
    # Add a trailing 1 dim to broadcast cos/sin across the (real, imag) pair.
    cos_shape_b = list(cos_pair_tt.shape) + [1]
    sin_shape_b = list(sin_pair_tt.shape) + [1]
    cos_b = ttnn.reshape(cos_pair_tt, cos_shape_b)
    sin_b = ttnn.reshape(sin_pair_tt, sin_shape_b)
    if inverse:
        new_real = ttnn.add(ttnn.multiply(real, cos_b), ttnn.multiply(imag, sin_b))
        new_imag = ttnn.subtract(ttnn.multiply(imag, cos_b), ttnn.multiply(real, sin_b))
    else:
        new_real = ttnn.subtract(ttnn.multiply(real, cos_b), ttnn.multiply(imag, sin_b))
        new_imag = ttnn.add(ttnn.multiply(real, sin_b), ttnn.multiply(imag, cos_b))
    paired = ttnn.concat([new_real, new_imag], dim=-1)
    return ttnn.reshape(paired, out_shape)


def _device_q_rsqrt_norm(ttnn, q_tt, eps: float):
    """Per-head rsqrt-norm: q *= rsqrt(mean(q^2, last) + eps).

    q_tt: [B, S, H, D]. mean is taken over D. Returns same shape.
    """
    sq = ttnn.multiply(q_tt, q_tt)
    mean_sq = ttnn.mean(sq, dim=-1, keepdim=True)
    inv = ttnn.rsqrt(ttnn.add(mean_sq, eps))
    return ttnn.multiply(q_tt, inv)


class DeviceAttention(nn.Module):
    """Fused MLA attention forward on a 1xN mesh (decode path device-resident).

    Owns nothing but small tables (cos/sin, attn_sink, kv_norm/q_norm) and
    references to the existing per-attn DeviceColLinear / DeviceGroupedLinear
    instances. Reuses `DeviceRMSNorm.forward_device` for q_norm/kv_norm,
    `DeviceColLinear.forward_device` for wq_a/wq_b/wkv/wo_b,
    `DeviceGroupedLinear.forward_device` for wo_a, and
    `DeviceSparseAttn.forward_device` for sparse_attn.

    Decode path (start_pos > 0, seqlen == 1): one upload of x, end-to-end
    on device, one download. CPU side effects (act_quant, kv_cache update,
    indexer/compressor) get small download/upload trips in the middle.

    Prefill path (start_pos == 0): falls back to the original CPU forward.
    """

    def __init__(self, mesh, attn, sparse_attn_dev: "DeviceSparseAttn",
                 q_norm_dev: "DeviceRMSNorm", kv_norm_dev: "DeviceRMSNorm",
                 wo_a_cpu_weight: torch.Tensor, max_seq_len: int):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        self.attn = attn
        self.sparse_attn_dev = sparse_attn_dev
        self.q_norm_dev = q_norm_dev
        self.kv_norm_dev = kv_norm_dev
        self.dim = attn.dim
        self.n_heads = attn.n_local_heads
        self.head_dim = attn.head_dim
        self.rope_head_dim = attn.rope_head_dim
        self.q_lora_rank = attn.q_lora_rank
        self.n_groups = attn.n_local_groups
        self.o_lora_rank = attn.o_lora_rank
        self.softmax_scale = float(attn.softmax_scale)
        self.eps = float(attn.eps)
        self.window_size = attn.window_size
        self.compress_ratio = attn.compress_ratio

        # cos/sin tables: real and imag of freqs_cis (complex64). Replicated.
        # Shape: [max_seq_len, rd/2]. We slice [start_pos:start_pos+1] per call.
        rd = self.rope_head_dim
        fc = attn.freqs_cis  # [max_seq_len, rd/2] complex
        cos_full = fc.real.to(torch.bfloat16).contiguous()
        sin_full = fc.imag.to(torch.bfloat16).contiguous()
        rep = dict(
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.cos_full_tt = ttnn.as_tensor(cos_full, **rep)
        self.sin_full_tt = ttnn.as_tensor(sin_full, **rep)
        self.max_seq_len = cos_full.shape[0]

        # wo_a: replicated [n_groups, in_per_group, o_lora_rank] (block-diagonal).
        # Original CPU weight: [n_groups * o_lora_rank, in_per_group].
        out_total, in_per_group = wo_a_cpu_weight.shape
        if out_total != self.n_groups * self.o_lora_rank:
            raise ValueError(
                f"wo_a shape {tuple(wo_a_cpu_weight.shape)} != expected "
                f"({self.n_groups * self.o_lora_rank}, in_per_group)"
            )
        self.wo_a_in_per_group = in_per_group
        w_g = wo_a_cpu_weight.to(torch.bfloat16).view(
            self.n_groups, self.o_lora_rank, in_per_group
        ).transpose(1, 2).contiguous()  # [G, in, R]
        self.wo_a_w_tt = ttnn.as_tensor(w_g, **rep)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Decode-only fused forward. Falls back to CPU for prefill."""
        attn = self.attn
        bsz, seqlen, _ = x.shape
        if start_pos == 0 or seqlen != 1:
            return self._cpu_forward(x, start_pos)
        return self._decode(x, start_pos)

    def _cpu_forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Fallback: invoke the original Attention.forward via the saved
        bound method. The Model wires this in offload_attn_full."""
        return self._orig_forward(x, start_pos)

    def _decode(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        ttnn = self._ttnn
        attn = self.attn
        B, S, _ = x.shape
        H, D = self.n_heads, self.head_dim
        rd = self.rope_head_dim
        win = self.window_size

        # Single upload of x.
        x_tt = ttnn.as_tensor(
            x.to(torch.bfloat16).contiguous(),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

        # Q path: wq_a -> q_norm -> wq_b -> per-head rsqrt-norm -> rotary.
        q_lora_tt = attn.wq_a.forward_device(x_tt)            # [B, S, q_lora_rank]
        # q_norm: needs a [Mpad, hidden] device tensor. Reshape and pad-on-device.
        q_lora_2d = ttnn.reshape(q_lora_tt, [B * S, self.q_lora_rank])
        qr_2d = self._rmsnorm_device(self.q_norm_dev, q_lora_2d, B * S)
        qr_tt = ttnn.reshape(qr_2d, [B, S, self.q_lora_rank])
        # Save qr for the indexer (it needs CPU qr).
        qr_cpu = self._download_replicated(qr_tt, B, S, self.q_lora_rank)

        q_full_tt = attn.wq_b.forward_device(qr_tt)           # [B, S, H*D]
        q_tt = ttnn.reshape(q_full_tt, [B, S, H, D])
        q_tt = _device_q_rsqrt_norm(ttnn, q_tt, self.eps)

        cos_q, sin_q = self._rotary_tables(start_pos, S, q_dims=4)
        # Rotate q[..., -rd:] by slicing -> rotating -> concatenating.
        q_nope = ttnn.slice(q_tt, [0, 0, 0, 0], [B, S, H, D - rd])
        q_rope = ttnn.slice(q_tt, [0, 0, 0, D - rd], [B, S, H, D])
        q_rope = _device_apply_rotary_interleaved(ttnn, q_rope, cos_q, sin_q, inverse=False)
        q_tt = ttnn.concat([q_nope, q_rope], dim=-1)

        # KV path: wkv -> kv_norm -> rotary.
        kv_tt = attn.wkv.forward_device(x_tt)                 # [B, S, D]
        kv_2d = ttnn.reshape(kv_tt, [B * S, D])
        kv_2d = self._rmsnorm_device(self.kv_norm_dev, kv_2d, B * S)
        kv_tt = ttnn.reshape(kv_2d, [B, S, D])
        cos_kv, sin_kv = self._rotary_tables(start_pos, S, q_dims=3)
        kv_nope = ttnn.slice(kv_tt, [0, 0, 0], [B, S, D - rd])
        kv_rope = ttnn.slice(kv_tt, [0, 0, D - rd], [B, S, D])
        kv_rope = _device_apply_rotary_interleaved(ttnn, kv_rope, cos_kv, sin_kv, inverse=False)
        kv_tt = ttnn.concat([kv_nope, kv_rope], dim=-1)

        # Round-trip kv to CPU for in-place act_quant and CPU side effects
        # (kv_cache update, compressor, indexer).
        kv_cpu = self._download_replicated(kv_tt, B, S, D).to(x.dtype)
        act_quant(kv_cpu[..., :-rd], 64, scale_fmt, scale_dtype, True)

        topk_idxs = get_window_topk_idxs(win, B, S, start_pos)
        if self.compress_ratio:
            offset = win
            if attn.indexer is not None:
                compress_topk_idxs = attn.indexer(x, qr_cpu, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio, B, S, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # KV cache update on CPU.
        attn.kv_cache[:B, start_pos % win] = kv_cpu.squeeze(1)
        if self.compress_ratio:
            attn.compressor(x, start_pos)
        kv_full_cpu = attn.kv_cache[:B]  # [B, kv_cache_size, D]

        # Sparse_attn back on device.
        common_rep = dict(
            device=self.mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        kv_full_tt = ttnn.as_tensor(
            kv_full_cpu.squeeze(0).to(torch.bfloat16).contiguous(), **common_rep,
        )
        idxs_tt, valid_tt = self.sparse_attn_dev._upload_topk(topk_idxs)
        K = topk_idxs.shape[-1]
        o_tt = self.sparse_attn_dev.forward_device(
            q_tt, kv_full_tt, idxs_tt, valid_tt, S, K
        )
        # o_tt: [B, S, H, D].

        # Inverse rotary on o[..., -rd:].
        cos_o, sin_o = self._rotary_tables(start_pos, S, q_dims=4)
        o_nope = ttnn.slice(o_tt, [0, 0, 0, 0], [B, S, H, D - rd])
        o_rope = ttnn.slice(o_tt, [0, 0, 0, D - rd], [B, S, H, D])
        o_rope = _device_apply_rotary_interleaved(ttnn, o_rope, cos_o, sin_o, inverse=True)
        o_tt = ttnn.concat([o_nope, o_rope], dim=-1)

        # Group reshape: [B, S, H, D] -> [G, B*S, H*D/G] for batched matmul.
        per_group = (H * D) // self.n_groups
        if per_group != self.wo_a_in_per_group:
            raise RuntimeError(
                f"wo_a in_per_group {self.wo_a_in_per_group} != H*D/G {per_group}"
            )
        o_perm = ttnn.reshape(o_tt, [B, S, self.n_groups, per_group])
        # Permute to [G, B, S, in_per_group] then merge B*S.
        o_perm = ttnn.permute(o_perm, [2, 0, 1, 3])
        o_g = ttnn.reshape(o_perm, [self.n_groups, B * S, per_group])
        # Block-diag matmul: [G, B*S, in] @ [G, in, R] -> [G, B*S, R]
        o_wo_a_g = ttnn.matmul(o_g, self.wo_a_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Reshape back to [B, S, G, R] then flatten last two dims.
        o_wo_a_rs = ttnn.reshape(o_wo_a_g, [self.n_groups, B, S, self.o_lora_rank])
        o_wo_a = ttnn.permute(o_wo_a_rs, [1, 2, 0, 3])  # [B, S, G, R]
        R = self.o_lora_rank
        o_flat = ttnn.reshape(o_wo_a, [B, S, self.n_groups * R])
        out_tt = attn.wo_b.forward_device(o_flat)             # [B, S, dim]

        # Download once.
        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        out = ttnn.to_torch(out_tt, mesh_composer=composer)[:B]
        return out.to(x.dtype)

    def _rmsnorm_device(self, norm_dev: "DeviceRMSNorm", x_2d_tt, num_rows: int):
        """Run device rmsnorm on a 2D [M, hidden] device tensor. Pads M to TILE
        on device, runs the kernel, then slices the output back to [num_rows,
        hidden]."""
        ttnn = self._ttnn
        Mpad = -(-num_rows // _RMS_TILE) * _RMS_TILE
        cur_M = tuple(x_2d_tt.shape)[0]
        hidden = tuple(x_2d_tt.shape)[1]
        if cur_M < Mpad:
            x_2d_tt = ttnn.pad(
                x_2d_tt,
                padding=[(0, Mpad - cur_M), (0, 0)],
                value=0.0,
            )
        out = norm_dev.forward_device(x_2d_tt, num_rows)
        if num_rows < Mpad:
            out = ttnn.slice(out, [0, 0], [num_rows, hidden])
        return out

    def _rotary_tables(self, start_pos: int, S: int, q_dims: int):
        """Slice cos/sin tables for [start_pos:start_pos+S] and reshape for
        broadcasting against q (4D, [B,S,H,rd/2,1] post-unsqueeze) or kv
        (3D, [B,S,rd/2,1])."""
        ttnn = self._ttnn
        rd_half = self.rope_head_dim // 2
        cos = ttnn.slice(self.cos_full_tt, [start_pos, 0], [start_pos + S, rd_half])
        sin = ttnn.slice(self.sin_full_tt, [start_pos, 0], [start_pos + S, rd_half])
        if q_dims == 4:
            cos = ttnn.reshape(cos, [1, S, 1, rd_half])
            sin = ttnn.reshape(sin, [1, S, 1, rd_half])
        else:  # q_dims == 3
            cos = ttnn.reshape(cos, [1, S, rd_half])
            sin = ttnn.reshape(sin, [1, S, rd_half])
        return cos, sin

    def _download_replicated(self, t_tt, *expected_shape):
        """Read back a replicated tensor (chip 0's copy via concat-on-dim0
        composer trick used elsewhere in this file)."""
        ttnn = self._ttnn
        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        t = ttnn.to_torch(t_tt, mesh_composer=composer)
        return t[:expected_shape[0]] if expected_shape else t


def _open_mesh(shape=(1, 4)):
    """Open a mesh device. Enables 1D fabric for CCL ops (all_gather etc.)."""
    import ttnn
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape))


def _close_mesh(mesh):
    import ttnn
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ==============================================================================
# MODEL: high-level wrapper (nanochat-style)
# ==============================================================================


# HF -> DeepSeek-inference name map (same rules as inference/convert.py, mp=1).
_HF_KEY_MAP = {
    "embed_tokens": "embed",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "q_proj": "wq",
    "q_a_proj": "wq_a",
    "q_a_layernorm": "q_norm",
    "q_b_proj": "wq_b",
    "kv_a_proj_with_mqa": "wkv_a",
    "kv_a_layernorm": "kv_norm",
    "kv_b_proj": "wkv_b",
    "o_proj": "wo",
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
    "lm_head": "head",
}


def _hf_rename(name: str) -> str:
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    parts = name.split(".")
    # Find the last segment to remap
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in _HF_KEY_MAP:
            parts[i] = _HF_KEY_MAP[parts[i]]
            break
    return ".".join(parts)


class Model:
    """High-level inference interface (nanochat-style).

    Usage:
        m = Model.from_hf("deepseek-ai/DeepSeek-V4-Flash", max_seq_len=512)
        m.load_weights()
        tokens = m.tokenizer.encode("Hello")
        for tok in m.generate(tokens, max_tokens=32):
            print(m.tokenizer.decode([tok]), end="")
    """

    def __init__(self, args: ModelArgs, tokenizer, ckpt_dir: str):
        self.args = args
        self.tokenizer = tokenizer
        self.ckpt_dir = ckpt_dir
        self.transformer = Transformer(args)

    @classmethod
    def from_hf(cls, repo_id: str, max_seq_len: int = 512, max_batch_size: int = 1):
        from transformers import PreTrainedTokenizerFast
        ckpt_dir = snapshot_download(repo_id, allow_patterns=[
            "*.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer*",
            "inference/config.json",
        ])
        with open(os.path.join(ckpt_dir, "inference", "config.json")) as f:
            cfg = json.load(f)
        # Map inference config keys -> ModelArgs fields
        args = ModelArgs(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=cfg.get("dtype", "fp8"),
            scale_fmt=cfg.get("scale_fmt", "ue8m0"),
            expert_dtype=cfg.get("expert_dtype", "fp4"),
            scale_dtype="fp8",
            vocab_size=cfg["vocab_size"],
            dim=cfg["dim"],
            moe_inter_dim=cfg["moe_inter_dim"],
            n_layers=cfg["n_layers"],
            n_hash_layers=cfg.get("n_hash_layers", 0),
            n_heads=cfg["n_heads"],
            n_routed_experts=cfg["n_routed_experts"],
            n_shared_experts=cfg["n_shared_experts"],
            n_activated_experts=cfg["n_activated_experts"],
            score_func=cfg.get("score_func", "sqrtsoftplus"),
            route_scale=cfg.get("route_scale", 1.0),
            swiglu_limit=cfg.get("swiglu_limit", 0.0),
            q_lora_rank=cfg["q_lora_rank"],
            head_dim=cfg["head_dim"],
            rope_head_dim=cfg["rope_head_dim"],
            o_groups=cfg["o_groups"],
            o_lora_rank=cfg["o_lora_rank"],
            window_size=cfg["window_size"],
            compress_ratios=tuple(cfg["compress_ratios"]),
            compress_rope_theta=cfg.get("compress_rope_theta", 160000.0),
            original_seq_len=cfg["original_seq_len"],
            rope_theta=cfg["rope_theta"],
            rope_factor=cfg["rope_factor"],
            beta_fast=cfg["beta_fast"],
            beta_slow=cfg["beta_slow"],
            index_n_heads=cfg["index_n_heads"],
            index_head_dim=cfg["index_head_dim"],
            index_topk=cfg["index_topk"],
            hc_mult=cfg["hc_mult"],
            hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(ckpt_dir, "tokenizer.json"),
            bos_token="<｜begin▁of▁sentence｜>",
            eos_token="<｜end▁of▁sentence｜>",
        )
        return cls(args, tokenizer, ckpt_dir)

    def load_weights(self, verbose: bool = False, cache_path: Optional[str] = None):
        """Load HF safetensors into the Transformer with on-the-fly key renaming and dtype fixes.

        Two-pass so we can combine FP8 wo_a weight+scale into a single bf16 tensor
        (the inference Linear for wo_a is declared bf16; convert.py does the same dequant).

        If `cache_path` is set and the file exists, loads the prebuilt state_dict
        via mmap (fast). If `cache_path` is set but the file is missing, runs the
        full safetensors path and then saves the result for next time.
        """
        if cache_path and os.path.exists(cache_path):
            print(f"[load] reading weights cache from {cache_path}")
            t0 = time.time()
            cached = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=True)
            missing, unexpected = self.transformer.load_state_dict(cached, strict=False)
            print(f"[load] cache hit: loaded {len(cached)} tensors in {time.time()-t0:.1f}s")
            if missing or unexpected:
                print(f"[load] cache: {len(missing)} missing, {len(unexpected)} unexpected (non-fatal)")
                if verbose:
                    for m in missing[:20]:
                        print(f"[load.cache.missing] {m}")
                    for u in unexpected[:20]:
                        print(f"[load.cache.unexpected] {u}")
            return

        from glob import glob
        files = sorted(glob(os.path.join(self.ckpt_dir, "*.safetensors")))
        assert files, f"No safetensors found in {self.ckpt_dir}"
        state = self.transformer.state_dict()
        target_keys = set(state.keys())

        # Index: new_name -> (file, orig_name)
        index = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    index[_hf_rename(name)] = (path, name)

        loaded = 0
        skipped = []
        handled = set()

        def _read(new_name):
            path, orig = index[new_name]
            with safe_open(path, framework="pt", device="cpu") as f:
                return f.get_tensor(orig)

        # Pass 1: dequantize fp8 wo_a weight+scale pairs into bf16
        for new_name in list(index.keys()):
            if not new_name.endswith(".wo_a.weight"):
                continue
            if new_name not in target_keys:
                continue
            target = state[new_name]
            if target.dtype != torch.bfloat16:
                continue
            scale_name = new_name[:-len(".weight")] + ".scale"
            if scale_name not in index:
                continue
            w = _read(new_name)
            s = _read(scale_name)
            if w.dtype != torch.float8_e4m3fn:
                # nothing to dequantize, just cast
                state[new_name].copy_(w.to(torch.bfloat16))
            else:
                # reshape to (out//128, 128, in//128, 128) * scale (out//128, in//128), bf16
                w = w.to(torch.float32).unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
                s = s.to(torch.float32)
                w = w * s[:, None, :, None]
                w = w.flatten(2, 3).flatten(0, 1).bfloat16()
                state[new_name].copy_(w)
            handled.add(new_name)
            handled.add(scale_name)
            loaded += 1

        # Pass 2: everything else
        for new_name, (path, orig) in index.items():
            if new_name in handled:
                continue
            if new_name.startswith("mtp."):
                skipped.append(new_name)
                continue
            if new_name not in target_keys:
                skipped.append(new_name)
                continue
            tensor = _read(new_name)
            target = state[new_name]
            if tensor.dtype != target.dtype:
                if target.dtype == torch.float4_e2m1fn_x2 and tensor.dtype == torch.int8:
                    tensor = tensor.view(torch.float4_e2m1fn_x2)
                elif target.dtype == torch.float8_e8m0fnu:
                    tensor = tensor.view(torch.float8_e8m0fnu) if tensor.dtype == torch.uint8 else tensor.to(torch.float8_e8m0fnu)
                else:
                    tensor = tensor.to(target.dtype)
            if tensor.shape != target.shape:
                skipped.append(f"{new_name} shape {tuple(tensor.shape)} != {tuple(target.shape)}")
                continue
            state[new_name].copy_(tensor)
            loaded += 1

        # Which target keys did we NOT populate? (both passes)
        written = handled | {n for n in index if n in target_keys}
        missing = sorted(k for k in target_keys if k not in written)
        print(f"[load] loaded {loaded} tensors, skipped {len(skipped)} source entries, missing {len(missing)} target params")
        if verbose:
            for s in skipped[:40]:
                print(f"[load.skip] {s}")
            for m in missing[:40]:
                print(f"[load.missing] {m}")
        else:
            if skipped:
                print(f"[load] first 10 skipped: {skipped[:10]}")
            if missing:
                print(f"[load] first 10 missing: {missing[:10]}")

        if cache_path:
            print(f"[load] writing weights cache to {cache_path} (first-run only) ...")
            t0 = time.time()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.transformer.state_dict(), cache_path)
            print(f"[load] cache written in {time.time()-t0:.1f}s")

    def offload_lm_head(self, mesh, include_final_norm: bool = True):
        """Move the lm_head matmul (and optionally the final RMSNorm) to a 1xN mesh.

        Must be called after load_weights(). Shards the lm_head column-parallel
        across vocab; replicates the tiny final-norm weight on each chip.

        When `include_final_norm=True`, ParallelHead.forward is rebound to skip
        the CPU norm and pass the unnormalized last-token hidden straight to
        the device (norm + matmul fused there).
        """
        head = self.transformer.head
        final_norm = self.transformer.norm
        norm_w = final_norm.weight.data if include_final_norm else None
        norm_eps = final_norm.eps if include_final_norm else 1e-6
        self._device_lm_head = DeviceLMHead(mesh, head.weight.data,
                                            norm_weight=norm_w, norm_eps=norm_eps)
        dlh = self._device_lm_head
        if include_final_norm:
            # Replace the whole head forward so CPU norm is skipped.
            def _device_head_forward(self_head, x, hc_fn, hc_scale, hc_base, norm):
                with _phase("head.hc_combiner"):
                    x = self_head.hc_head(x, hc_fn, hc_scale, hc_base)  # still CPU
                with _phase("head.norm_and_logits"):
                    return dlh(x[:, -1].contiguous())
            head.forward = _device_head_forward.__get__(head, type(head))
            final_norm.weight.data = torch.empty(0)  # device owns it
        else:
            def _device_get_logits(self_head, x):
                with _phase("head.norm_and_logits"):
                    return dlh(x[:, -1].contiguous())
            head.get_logits = _device_get_logits.__get__(head, type(head))
        head.weight.data = torch.empty(0)

    def offload_attn_wq_b(self, mesh):
        """Column-parallel shard wq_b (Q-LoRA up) on n_heads*head_dim."""
        self._device_wq_b = []
        for layer in self.transformer.layers:
            attn = layer.attn
            dw = DeviceColLinear(mesh, attn.wq_b.weight)
            attn.wq_b = dw
            self._device_wq_b.append(dw)

    def offload_attn_wq_a(self, mesh):
        """Column-parallel shard wq_a (Q-LoRA down) on q_lora_rank."""
        self._device_wq_a = []
        for layer in self.transformer.layers:
            attn = layer.attn
            dw = DeviceColLinear(mesh, attn.wq_a.weight)
            attn.wq_a = dw
            self._device_wq_a.append(dw)

    def offload_attn_wkv(self, mesh):
        """Column-parallel shard wkv (KV-with-MQA down) on head_dim."""
        self._device_wkv = []
        for layer in self.transformer.layers:
            attn = layer.attn
            dw = DeviceColLinear(mesh, attn.wkv.weight)
            attn.wkv = dw
            self._device_wkv.append(dw)

    def offload_attn_wo_a(self, mesh):
        """Shard wo_a block-diagonal per-group across the 1xN mesh.

        wo_a is bf16 with a block-diagonal usage: each of the n_groups groups
        has its own [o_lora_rank, in_per_group] matrix, and activations are
        reshaped to [B, S, n_groups, in_per_group] before the einsum. Groups
        divide evenly across the mesh axis 1 (8 groups / 4 chips = 2/chip).
        Swaps Attention._wo_a_grouped for a device callable.
        """
        self._device_wo_a = []
        for layer in self.transformer.layers:
            attn = layer.attn
            dg = DeviceGroupedLinear(mesh, attn.wo_a.weight, n_groups=attn.n_local_groups)

            def _device_wo_a_call(self_attn, o, _dg=dg):
                return _dg(o)

            attn._wo_a_grouped = _device_wo_a_call.__get__(attn, type(attn))
            self._device_wo_a.append(dg)

    def offload_sparse_attn(self, mesh):
        """Run sparse_attn on the 1xN mesh per layer.

        Replaces Attention._sparse_attn with a DeviceSparseAttn callable that
        uploads (q, kv, topk_idxs), runs gather+score-matmul+masked-softmax+
        weighted-sum on device, and reads back o from chip 0. Per-call round-
        trips remain — they collapse once the rest of attn moves on-device.
        """
        self._device_sparse_attn = []
        for layer in self.transformer.layers:
            attn = layer.attn
            dsa = DeviceSparseAttn(mesh, attn.attn_sink, attn.softmax_scale)

            def _device_sparse_attn_call(self_attn, q, kv, topk_idxs, _dsa=dsa):
                return _dsa(q, kv, topk_idxs)

            attn._sparse_attn = _device_sparse_attn_call.__get__(attn, type(attn))
            self._device_sparse_attn.append(dsa)

    def offload_rms_norms(self, mesh):
        """Move attn_norm, ffn_norm, q_norm, kv_norm on every Block to the 1xN mesh.

        Uses the tt-lang fused rmsnorm kernel (replicated gamma). Per-(hidden,
        M_tiles) kernel compiled lazily. Compressor's internal norm stays CPU
        for now (its inputs/outputs are still CPU; offloading would add
        round-trips with no gain until Compressor itself is on device).
        Final norm is already on device via offload_lm_head.
        """
        self._device_rms_norms = []

        def _bind(norm):
            dn = DeviceRMSNorm(mesh, norm.weight, norm.eps)
            norm.forward = dn.forward
            self._device_rms_norms.append(dn)

        for layer in self.transformer.layers:
            _bind(layer.attn_norm)
            _bind(layer.ffn_norm)
            _bind(layer.attn.q_norm)
            _bind(layer.attn.kv_norm)

    def offload_moe_shared_expert(self, mesh):
        """Move every MoE layer's shared-expert SwiGLU to the 1xN mesh.

        Weights are small per layer (~48MB bf16) and replicated on every chip;
        output is read back from chip 0. Correctness-first TP (no CCL).
        """
        self._device_shared_experts = []
        for layer in self.transformer.layers:
            ffn = layer.ffn
            if not isinstance(ffn, MoE):
                continue
            se = ffn.shared_experts
            dse = DeviceSharedExpert(
                mesh, se.w1.weight, se.w2.weight, se.w3.weight,
                swiglu_limit=se.swiglu_limit,
            )
            ffn.shared_experts = dse
            self._device_shared_experts.append(dse)

    def offload_embedding(self, mesh):
        """Move token embedding to the 1xN mesh (replicated).

        Replaces Transformer.embed.forward with a ttnn.embedding call. The
        weight is tiny-ish for V4 (vocab=129280 * dim=4096 bf16 ~1GB) and
        fits comfortably when replicated per chip.
        """
        import ttnn
        embed = self.transformer.embed
        w = _weight_to_bf16(embed.weight).contiguous()
        mesh_shape = tuple(mesh.shape)
        w_tt = ttnn.as_tensor(
            w,
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self._device_embed_weight = w_tt

        def _device_embed(self_embed, x: torch.Tensor) -> torch.Tensor:
            ids = x.to(torch.int32).contiguous()
            ids_tt = ttnn.as_tensor(
                ids,
                device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            y_tt = ttnn.embedding(ids_tt, w_tt, layout=ttnn.TILE_LAYOUT)
            composer = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape, dims=(1, 0))
            y = ttnn.to_torch(y_tt, mesh_composer=composer)
            B = x.shape[0]
            return y[:B].to(torch.bfloat16)

        embed.forward = _device_embed.__get__(embed, type(embed))
        embed.weight.data = torch.empty(0)  # device owns it

    def offload_compressor_linears(self, mesh):
        """Col-parallel shard Compressor.wkv + wgate on the 1xN mesh for every
        layer that has a compressor (attn.compress_ratio > 0 OR indexer's
        compressor). Linear compute is on device; the rest of Compressor
        stays CPU."""
        self._device_compressor_linears = []
        for layer in self.transformer.layers:
            attn = layer.attn
            for comp in self._iter_compressors(attn):
                for attr in ("wkv", "wgate"):
                    lin = getattr(comp, attr)
                    dw = DeviceColLinear(mesh, lin.weight)
                    setattr(comp, attr, dw)
                    self._device_compressor_linears.append(dw)

    def offload_indexer_linears(self, mesh):
        """Col-parallel shard Indexer.wq_b + weights_proj on the 1xN mesh for
        every layer that has an indexer (attn.compress_ratio == 4)."""
        self._device_indexer_linears = []
        for layer in self.transformer.layers:
            attn = layer.attn
            indexer = getattr(attn, "indexer", None)
            if indexer is None:
                continue
            for attr in ("wq_b", "weights_proj"):
                lin = getattr(indexer, attr)
                dw = DeviceColLinear(mesh, lin.weight)
                setattr(indexer, attr, dw)
                self._device_indexer_linears.append(dw)

    @staticmethod
    def _iter_compressors(attn):
        """Yield every Compressor reachable from an Attention (the attn's own
        compressor, plus the Indexer's compressor when the layer has one)."""
        if getattr(attn, "compress_ratio", 0):
            yield attn.compressor
        indexer = getattr(attn, "indexer", None)
        if indexer is not None:
            yield indexer.compressor

    def offload_moe_gate(self, mesh):
        """Move every non-hash MoE gate to the 1xN mesh.

        Gate is tiny (n_experts x dim bf16, ~2MB + n_experts bias) -- replicated
        on every chip. Rebinds Gate.forward to call the device gate. Hash gates
        (layer < n_hash_layers) keep their CPU lookup path.
        """
        self._device_moe_gates = []
        for layer in self.transformer.layers:
            ffn = layer.ffn
            if not isinstance(ffn, MoE):
                continue
            gate = ffn.gate
            if gate.hash:
                continue

            dg = DeviceMoEGate(
                mesh, gate.weight, gate.bias,
                topk=gate.topk, route_scale=gate.route_scale,
                score_func=gate.score_func,
            )

            def _device_gate_forward(self_gate, x, input_ids=None, _dg=dg):
                return _dg(x)

            gate.forward = _device_gate_forward.__get__(gate, type(gate))
            self._device_moe_gates.append(dg)

    def offload_attn_wo_b(self, mesh):
        """Col-parallel shard wo_b on its 4096-wide output dim.

        Row-parallel is the TP-correct pattern (shard on the 8192 input) and
        DeviceRowLinear exists, but it was producing garbage — tracked as
        follow-up tech debt. Col-parallel here is numerically equivalent
        with more activation bandwidth per chip (fine on QB fabric).
        """
        self._device_wo_b = []
        for layer in self.transformer.layers:
            attn = layer.attn
            dw = DeviceColLinear(mesh, attn.wo_b.weight)
            attn.wo_b = dw
            self._device_wo_b.append(dw)

    def offload_attn_full(self, mesh):
        """Fused end-to-end attention forward on device (decode path).

        Requires every attention sub-piece already offloaded:
          --offload-attn-wq-a / --offload-attn-wq-b / --offload-attn-wkv
          --offload-attn-wo-a / --offload-attn-wo-b
          --offload-rms-norms (q_norm/kv_norm device-resident)
          --offload-sparse-attn

        Replaces Attention.forward with DeviceAttention.forward, which
        consolidates the per-step Q/KV/O paths into a single upload+download.
        Prefill (start_pos == 0) falls back to the original CPU forward.
        """
        self._device_attn_full = []
        for layer in self.transformer.layers:
            attn = layer.attn
            for w_attr in ("wq_a", "wq_b", "wkv", "wo_b"):
                w = getattr(attn, w_attr)
                if not isinstance(w, DeviceColLinear):
                    raise RuntimeError(
                        f"offload_attn_full requires {w_attr} on device "
                        f"(use --offload-attn-{w_attr.replace('_', '-')})"
                    )
            if not hasattr(self, "_device_sparse_attn"):
                raise RuntimeError(
                    "offload_attn_full requires --offload-sparse-attn"
                )
            if not hasattr(self, "_device_rms_norms"):
                raise RuntimeError(
                    "offload_attn_full requires --offload-rms-norms (q_norm/kv_norm)"
                )
            sparse_dev = self._device_sparse_attn[layer.layer_id]
            # q_norm/kv_norm DeviceRMSNorm instances are appended in the order
            # (attn_norm, ffn_norm, q_norm, kv_norm) per layer in offload_rms_norms.
            base = layer.layer_id * 4
            q_norm_dev = self._device_rms_norms[base + 2]
            kv_norm_dev = self._device_rms_norms[base + 3]

            # wo_a: take the CPU weight as our authoritative source. This works
            # whether or not --offload-attn-wo-a was set (DeviceGroupedLinear
            # does not wipe attn.wo_a.weight).
            da = DeviceAttention(
                mesh=mesh,
                attn=attn,
                sparse_attn_dev=sparse_dev,
                q_norm_dev=q_norm_dev,
                kv_norm_dev=kv_norm_dev,
                wo_a_cpu_weight=attn.wo_a.weight.detach(),
                max_seq_len=attn.freqs_cis.shape[0],
            )
            # Save the original CPU forward so prefill can fall back.
            da._orig_forward = attn.forward
            attn.forward = da.forward
            self._device_attn_full.append(da)

    def offload_mhc(self, mesh):
        """Run Block.hc_pre + Block.hc_post on the 1xN mesh via tt-lang kernels.

        Per Block creates two DeviceMHC instances (one for hc_attn_*, one for
        hc_ffn_*) since each variant owns its own hc_fn weight. hc_post uses
        the same kernel either side and can dispatch through either DeviceMHC;
        we use the attn-side instance arbitrarily.

        ParallelHead.hc_head (the lm_head's combiner) keeps its CPU path; it
        runs once per generate call and is not on the hot decode path.
        """
        self._device_mhc = []
        for layer in self.transformer.layers:
            mhc_attn = DeviceMHC(
                mesh=mesh,
                hc_fn=layer.hc_attn_fn,
                hc_scale=layer.hc_attn_scale,
                hc_base=layer.hc_attn_base,
                hc_mult=layer.hc_mult,
                hc_eps=layer.hc_eps,
                sinkhorn_iters=layer.hc_sinkhorn_iters,
                norm_eps=layer.norm_eps,
            )
            mhc_ffn = DeviceMHC(
                mesh=mesh,
                hc_fn=layer.hc_ffn_fn,
                hc_scale=layer.hc_ffn_scale,
                hc_base=layer.hc_ffn_base,
                hc_mult=layer.hc_mult,
                hc_eps=layer.hc_eps,
                sinkhorn_iters=layer.hc_sinkhorn_iters,
                norm_eps=layer.norm_eps,
            )

            attn_fn_id = id(layer.hc_attn_fn)
            ffn_fn_id = id(layer.hc_ffn_fn)

            def _device_hc_pre(self_layer, x, hc_fn, hc_scale, hc_base,
                               _attn=mhc_attn, _ffn=mhc_ffn,
                               _attn_id=attn_fn_id, _ffn_id=ffn_fn_id):
                fid = id(hc_fn)
                if fid == _attn_id:
                    return _attn.hc_pre(x)
                if fid == _ffn_id:
                    return _ffn.hc_pre(x)
                raise RuntimeError(
                    "DeviceMHC dispatch: hc_fn identity matches neither "
                    "hc_attn_fn nor hc_ffn_fn"
                )

            def _device_hc_post(self_layer, x, residual, post, comb,
                                _mhc=mhc_attn):
                return _mhc.hc_post(x, residual, post, comb)

            layer.hc_pre = _device_hc_pre.__get__(layer, type(layer))
            layer.hc_post = _device_hc_post.__get__(layer, type(layer))
            self._device_mhc.append((mhc_attn, mhc_ffn))

    @torch.inference_mode()
    def step_decode(self, token_id: int, pos: int) -> torch.Tensor:
        """Single decode step. Returns logits [vocab_size]."""
        ids = torch.tensor([[token_id]], dtype=torch.long)
        logits = self.transformer.forward(ids, start_pos=pos)
        return logits[0]

    @torch.inference_mode()
    def prefill(self, tokens: list[int]) -> torch.Tensor:
        ids = torch.tensor([tokens], dtype=torch.long)
        logits = self.transformer.forward(ids, start_pos=0)
        return logits[0]

    @torch.inference_mode()
    def generate(self, tokens: list[int], max_tokens: int = 32, temperature: float = 1.0):
        with _phase("prefill"):
            logits = self.prefill(tokens)
        pos = len(tokens)
        for _ in range(max_tokens):
            with _phase("sample"):
                if temperature == 0:
                    nxt = int(logits.argmax().item())
                else:
                    probs = F.softmax(logits.float() / temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, num_samples=1).item())
            yield nxt
            with _phase("decode_step"):
                logits = self.step_decode(nxt, pos)
            pos += 1


def _resolve_parent(module: nn.Module, dotted: str):
    parts = dotted.split(".")
    m = module
    for p in parts[:-1]:
        m = getattr(m, p) if not p.isdigit() else m[int(p)]
    return m


# ==============================================================================
# main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--prompt-file", default=None,
                        help="Read prompt from file (bypasses shell quoting). Overrides --prompt.")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--weights-only", action="store_true",
                        help="Stop after loading weights (Phase 1 gate)")
    parser.add_argument("--verbose-load", action="store_true")
    parser.add_argument("--offload-lm-head", action="store_true",
                        help="Shard lm_head across a 1x4 Blackhole mesh (Phase 2)")
    parser.add_argument("--offload-attn-wq-b", action="store_true",
                        help="Shard wq_b (Q-LoRA up) across the 1x4 mesh (Phase 2)")
    parser.add_argument("--offload-attn-wq-a", action="store_true",
                        help="Shard wq_a (Q-LoRA down) across the 1x4 mesh (Phase 2)")
    parser.add_argument("--offload-attn-wkv", action="store_true",
                        help="Shard wkv (KV-with-MQA down) across the 1x4 mesh (Phase 2)")
    parser.add_argument("--offload-attn-wo-a", action="store_true",
                        help="Block-diagonal per-group shard wo_a across the 1x4 mesh (Phase 2)")
    parser.add_argument("--offload-attn-wo-b", action="store_true",
                        help="Row-parallel shard wo_b across the 1x4 mesh (Phase 2)")
    parser.add_argument("--offload-moe-gate", action="store_true",
                        help="Replicate MoE gate on 1x4 mesh (sqrtsoftplus); non-hash layers only")
    parser.add_argument("--offload-moe-shared-expert", action="store_true",
                        help="Replicate MoE shared-expert SwiGLU on 1x4 mesh (all layers)")
    parser.add_argument("--offload-rms-norms", action="store_true",
                        help="Run attn_norm/ffn_norm/q_norm/kv_norm via tt-lang rmsnorm kernel on 1x4 mesh")
    parser.add_argument("--offload-embedding", action="store_true",
                        help="Replicate token embedding on 1x4 mesh (ttnn.embedding)")
    parser.add_argument("--offload-compressor-linears", action="store_true",
                        help="Col-parallel shard Compressor.wkv + wgate on 1x4 mesh (per layer)")
    parser.add_argument("--offload-indexer-linears", action="store_true",
                        help="Col-parallel shard Indexer.wq_b + weights_proj on 1x4 mesh (per layer)")
    parser.add_argument("--offload-sparse-attn", action="store_true",
                        help="Run sparse_attn (gather+score+masked-softmax+weighted-sum) on 1x4 mesh per layer")
    parser.add_argument("--offload-attn-full", action="store_true",
                        help="Fused MLA attention forward on device (decode path only). "
                             "Requires all attn sub-pieces already offloaded.")
    parser.add_argument("--offload-mhc", action="store_true",
                        help="Run Block.hc_pre / hc_post on 1x4 mesh via tt-lang fused kernels "
                             "(pre_norm_fn + pre_apply_mix + post). Sinkhorn stays CPU.")
    parser.add_argument("--weights-cache",
                        default=os.environ.get("DS_WEIGHTS_CACHE",
                                               "/tmp/deepseek_v4_flash_cache/state_dict.pt"),
                        help="Path to pickled state_dict cache. First run populates it; "
                             "subsequent runs mmap it (set to empty string to disable).")
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(min(32, os.cpu_count() or 8))

    print(f"[phase] building model (max_seq_len={args.max_seq_len}) ...")
    t0 = time.time()
    model = Model.from_hf(args.repo, max_seq_len=args.max_seq_len)
    print(f"[phase] model instantiated in {time.time() - t0:.1f}s")

    print("[phase] loading weights ...")
    t0 = time.time()
    model.load_weights(verbose=args.verbose_load, cache_path=args.weights_cache or None)
    print(f"[phase] weights loaded in {time.time() - t0:.1f}s")

    mesh = None
    need_mesh = (
        args.offload_lm_head
        or args.offload_attn_wq_a
        or args.offload_attn_wq_b
        or args.offload_attn_wkv
        or args.offload_attn_wo_a
        or args.offload_attn_wo_b
        or args.offload_moe_gate
        or args.offload_moe_shared_expert
        or args.offload_rms_norms
        or args.offload_embedding
        or args.offload_compressor_linears
        or args.offload_indexer_linears
        or args.offload_sparse_attn
        or args.offload_attn_full
        or args.offload_mhc
    )
    if need_mesh:
        print("[phase] opening 1x4 mesh ...")
        t0 = time.time()
        mesh = _open_mesh(shape=(1, 4))
        print(f"[phase] mesh opened in {time.time() - t0:.1f}s")
    if args.offload_lm_head:
        print("[phase] sharding lm_head on mesh ...")
        t0 = time.time()
        model.offload_lm_head(mesh)
        print(f"[phase] lm_head offloaded in {time.time() - t0:.1f}s")
    n_layers = len(model.transformer.layers)
    if args.offload_attn_wq_a:
        print(f"[phase] sharding wq_a across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_attn_wq_a(mesh)
        print(f"[phase] wq_a offloaded in {time.time() - t0:.1f}s")
    if args.offload_attn_wq_b:
        print(f"[phase] sharding wq_b across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_attn_wq_b(mesh)
        print(f"[phase] wq_b offloaded in {time.time() - t0:.1f}s")
    if args.offload_attn_wkv:
        print(f"[phase] sharding wkv across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_attn_wkv(mesh)
        print(f"[phase] wkv offloaded in {time.time() - t0:.1f}s")
    if args.offload_attn_wo_a:
        print(f"[phase] grouped-sharding wo_a across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_attn_wo_a(mesh)
        print(f"[phase] wo_a offloaded in {time.time() - t0:.1f}s")
    if args.offload_attn_wo_b:
        print(f"[phase] row-sharding wo_b across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_attn_wo_b(mesh)
        print(f"[phase] wo_b offloaded in {time.time() - t0:.1f}s")
    if args.offload_moe_gate:
        print(f"[phase] replicating MoE gates across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_moe_gate(mesh)
        print(f"[phase] moe_gate offloaded in {time.time() - t0:.1f}s")
    if args.offload_moe_shared_expert:
        print(f"[phase] replicating MoE shared experts across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_moe_shared_expert(mesh)
        print(f"[phase] moe_shared_expert offloaded in {time.time() - t0:.1f}s")
    if args.offload_rms_norms:
        print(f"[phase] binding tt-lang rmsnorm across {n_layers} layers (attn/ffn/q/kv) ...")
        t0 = time.time()
        model.offload_rms_norms(mesh)
        print(f"[phase] rms_norms offloaded in {time.time() - t0:.1f}s")
    if args.offload_embedding:
        print(f"[phase] replicating embedding on mesh ...")
        t0 = time.time()
        model.offload_embedding(mesh)
        print(f"[phase] embedding offloaded in {time.time() - t0:.1f}s")
    if args.offload_compressor_linears:
        print(f"[phase] sharding Compressor wkv+wgate across layers on mesh ...")
        t0 = time.time()
        model.offload_compressor_linears(mesh)
        print(f"[phase] compressor_linears offloaded in {time.time() - t0:.1f}s")
    if args.offload_indexer_linears:
        print(f"[phase] sharding Indexer wq_b+weights_proj across layers on mesh ...")
        t0 = time.time()
        model.offload_indexer_linears(mesh)
        print(f"[phase] indexer_linears offloaded in {time.time() - t0:.1f}s")
    if args.offload_sparse_attn:
        print(f"[phase] binding device sparse_attn across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_sparse_attn(mesh)
        print(f"[phase] sparse_attn offloaded in {time.time() - t0:.1f}s")
    if args.offload_attn_full:
        print(f"[phase] binding fused DeviceAttention across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_attn_full(mesh)
        print(f"[phase] attn_full offloaded in {time.time() - t0:.1f}s")
    if args.offload_mhc:
        print(f"[phase] binding DeviceMHC across {n_layers} layers on mesh ...")
        t0 = time.time()
        model.offload_mhc(mesh)
        print(f"[phase] mhc offloaded in {time.time() - t0:.1f}s")

    if args.weights_only:
        print("[done] --weights-only set; stopping after weight load")
        if mesh is not None:
            _close_mesh(mesh)
        return

    print("[phase] encoding prompt ...")
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().rstrip("\n")
    tokens = model.tokenizer.encode(prompt)
    print(f"[phase] prompt = {prompt!r}, {len(tokens)} tokens")

    print("[phase] generating ...")
    t0 = time.time()
    out_tokens = []
    for tok in model.generate(tokens, max_tokens=args.max_tokens, temperature=args.temperature):
        out_tokens.append(tok)
        piece = model.tokenizer.decode([tok])
        print(piece, end="", flush=True)
    print()
    print(f"[debug] token ids: {out_tokens}")
    print(f"[debug] full decode: {model.tokenizer.decode(out_tokens)!r}")
    dt = time.time() - t0
    print(f"\n[phase] generated {len(out_tokens)} tokens in {dt:.1f}s ({len(out_tokens)/dt:.2f} tok/s)")

    print("\n[timing] per-phase breakdown (inclusive; sort by total):")
    print(_phase_report())

    if mesh is not None:
        _close_mesh(mesh)


if __name__ == "__main__":
    main()
