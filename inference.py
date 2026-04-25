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

block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16


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


def _dequant_fp8_weight(w_fp8: torch.Tensor, scale: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Dequantize an FP8 weight [N, K] with per-[group,group] scale to bf16."""
    N, K = w_fp8.shape
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


# ==============================================================================
# ARCHITECTURE (adapted from deepseek inference/model.py, CPU-safe)
# ==============================================================================


@dataclass
class ModelArgs:
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    expert_dtype: Literal[None, "fp4", "fp8"] = "fp4"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3
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

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]):
        with _phase("block.hc_pre"):
            x = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        with _phase("block.norm"):
            x = self.attn_norm(x)
        with _phase("block.attn"):
            x = self.attn(x, start_pos)
        with _phase("block.hc_post"):
            x = self.hc_post(x)

        with _phase("block.hc_pre"):
            x = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        with _phase("block.norm"):
            x = self.ffn_norm(x)
        with _phase("block.ffn"):
            x = self.ffn(x, input_ids)
        with _phase("block.hc_post"):
            x = self.hc_post(x)
        return x


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

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
        global default_dtype
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
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

        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """Allocate every per-step ttnn tensor this class uses. Called at
        construction so all decode allocations are co-located here. No ttnn
        op outside of this method is allowed to allocate a new device
        tensor; ops must write into pre-allocated outputs instead."""
        ttnn = self._ttnn
        self._x_upload_tt = ttnn.from_torch(
            torch.zeros(1, 1, self.dim, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self._matmul_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, self.vocab, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, self.mesh_shape, dims=(None, -1)),
        )
        self._upload_mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        self._trace_id = None

    def _compute_body(self):
        """Pure-device norm + matmul. Reads `_x_upload_tt`, writes
        `_matmul_out_tt`. Called once outside trace as warmup, then again
        inside `begin_trace_capture` / `end_trace_capture`."""
        ttnn = self._ttnn
        x_tt = self._x_upload_tt
        if self.norm_tt is not None:
            x_tt = ttnn.reshape(x_tt, (1, 1, 1, self.dim))
            x_tt = ttnn.rms_norm(x_tt, weight=self.norm_tt, epsilon=self.norm_eps)
            x_tt = ttnn.reshape(x_tt, (1, 1, self.dim))
        ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._matmul_out_tt)

    def _capture_trace(self):
        """Warmup outside trace (lets ttnn.rms_norm do its lazy host upload),
        then record the same op sequence into a replayable trace."""
        ttnn = self._ttnn
        self._compute_body()
        ttnn.synchronize_device(self.mesh)
        self._trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._compute_body()
        ttnn.end_trace_capture(self.mesh, self._trace_id, cq_id=0)

    def __call__(self, x_last: torch.Tensor) -> torch.Tensor:
        # x_last: [B=1, dim] — unnormalized if norm_tt is set, else normalized.
        ttnn = self._ttnn
        B = x_last.shape[0]
        if B != 1:
            raise ValueError(f"DeviceLMHead expects B=1, got B={B}")
        x_3d = x_last.to(torch.bfloat16).unsqueeze(1).contiguous()  # [1, 1, dim]
        host_mesh = ttnn.from_torch(
            x_3d,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)
        if self._trace_id is None:
            self._capture_trace()
        else:
            ttnn.execute_trace(self.mesh, self._trace_id, cq_id=0, blocking=True)
        y = ttnn.to_torch(
            self._matmul_out_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(0, -1)),
        )
        if y.shape[0] != B:
            y = y[: B]
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
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """Pre-allocate per-step output buffers for [B=1, S=1, out_dim]
        decode. forward_device's matmul + all_gather both write in place."""
        ttnn = self._ttnn
        zeros = torch.zeros(1, 1, self.out_dim, dtype=torch.bfloat16)
        self._matmul_out_tt = ttnn.from_torch(
            zeros,
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, self.mesh_shape, dims=(None, -1)),
        )
        self._gather_out_tt = ttnn.from_torch(
            zeros,
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
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
        """Device-only path. x_tt: replicated input [1, 1, in_dim]. Returns
        replicated output [1, 1, out_dim] via all_gather along the sharded
        axis. Output buffers are pre-allocated; both ops write in place."""
        ttnn = self._ttnn
        ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._matmul_out_tt)
        ttnn.all_gather(self._matmul_out_tt, dim=-1, num_links=1,
                        output_tensor=self._gather_out_tt)
        return self._gather_out_tt


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
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """Pre-allocate per-step input/output buffers (M=1). The traced
        compute body allocates intermediates (softplus/sqrt/add/topk/...)
        on the device side; trace replay reuses the captured addresses."""
        ttnn = self._ttnn
        common_rep = dict(
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self._x_upload_tt = ttnn.from_torch(
            torch.zeros(1, self.dim, dtype=torch.bfloat16), **common_rep)
        self._raw_tt = ttnn.from_torch(
            torch.zeros(1, self.n_experts, dtype=torch.bfloat16), **common_rep)
        self._upload_mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        self._trace_id = None
        self._weights_tt = None
        self._indices_tt = None

    def _compute_body(self):
        """Returns (weights_tt, indices_tt). Called outside trace as warmup
        and again during begin_trace_capture; on replay the captured
        op sequence reproduces the same output addresses."""
        ttnn = self._ttnn
        x_tt = self._x_upload_tt
        ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._raw_tt)
        scores_tt = ttnn.sqrt(ttnn.softplus(self._raw_tt))
        biased_tt = ttnn.add(scores_tt, self.bias_tt)
        _, indices_tt = ttnn.topk(biased_tt, k=self.topk, dim=-1, largest=True, sorted=True)
        gathered_tt = ttnn.gather(scores_tt, dim=-1, index=indices_tt)
        wsum_tt = ttnn.sum(gathered_tt, dim=-1, keepdim=True)
        normed_tt = ttnn.div(gathered_tt, wsum_tt)
        weights_tt = ttnn.multiply(normed_tt, self.route_scale)
        return weights_tt, indices_tt

    def _capture_trace(self):
        ttnn = self._ttnn
        self._weights_tt, self._indices_tt = self._compute_body()  # warmup
        ttnn.synchronize_device(self.mesh)
        self._trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._weights_tt, self._indices_tt = self._compute_body()
        ttnn.end_trace_capture(self.mesh, self._trace_id, cq_id=0)

    def forward(self, x: torch.Tensor):
        """x: [M=1, dim] -> (weights [1, topk] float32, indices [1, topk] int64)."""
        ttnn = self._ttnn
        M, D = x.shape
        if D != self.dim:
            raise ValueError(f"x last dim {D} mismatch gate dim {self.dim}")
        if M != 1:
            raise ValueError(f"DeviceMoEGate expects M=1, got M={M}")
        host_mesh = ttnn.from_torch(
            x.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)
        if self._trace_id is None:
            self._capture_trace()
        else:
            ttnn.execute_trace(self.mesh, self._trace_id, cq_id=0, blocking=True)

        # Replicated readback: stack 4 copies on tensor dim 0, take first M rows.
        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        weights = ttnn.to_torch(self._weights_tt, mesh_composer=composer)[:M].float()
        indices = ttnn.to_torch(self._indices_tt, mesh_composer=composer)[:M].long()
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
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """Kernel output buffer for the loop-prefill / decode case
        (num_rows=1, Mpad=TILE). The fused kernel overwrites every tile."""
        ttnn = self._ttnn
        self._out_tt = ttnn.zeros(
            shape=(_RMS_TILE, self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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
        variant (M_tiles = ceil(num_rows / TILE)). Loop-prefill keeps
        num_rows == 1 in the hot path; the pre-allocated [TILE, hidden]
        output buffer is reused across calls.
        """
        Mt = -(-num_rows // _RMS_TILE)
        if Mt != 1:
            raise RuntimeError(
                f"DeviceRMSNorm.forward_device: num_rows={num_rows} requires "
                f"Mt={Mt} tiles, but the pre-allocated output buffer is sized "
                f"for Mt=1. Loop-prefill keeps num_rows=1; lift the buffer "
                f"sizing if multi-row decode is reintroduced.")
        out_tt = self._out_tt
        self._kernel(Mt)(x_tt, self.gamma_tt, self.sc_tt, out_tt)
        return out_tt


# ============================================================================
# MHC (hyper-connection mixing) — tt-lang fused kernels on a 1xN mesh
# ============================================================================

_MHC_TILE = 32
# Softmax-sentinel for sinkhorn kernel: exp(PAD_SENTINEL - row_max) underflows
# to 0 in fp32 so padded cells don't leak into the normalization.
_MHC_PAD_SENTINEL = -1e4
# Fixed by the V4-Flash hc_split_sinkhorn reference (post = 2 * sigmoid(...)).
_MHC_POST_MULT = 2.0


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


def _mhc_pack_x_bc(x: torch.Tensor, mhc: int, num_tokens_pad: int) -> torch.Tensor:
    """[num_tokens, h] -> [num_tokens_pad * TILE, h] fp32 (rows 0..mhc-1 each hold a copy of x)."""
    num_tokens, h = x.shape
    src = x.unsqueeze(1).expand(-1, mhc, -1)
    out = torch.zeros(num_tokens_pad * _MHC_TILE, h, dtype=torch.float32)
    out.view(num_tokens_pad, _MHC_TILE, h)[:num_tokens, :mhc, :] = src.to(torch.float32)
    return out.contiguous()


def _mhc_unpack_apply_mix_out(packed: torch.Tensor, num_tokens: int, h: int) -> torch.Tensor:
    """[num_tokens_pad * TILE, h] -> [num_tokens, h] (row 0 of each block)."""
    return packed.view(-1, _MHC_TILE, h)[:num_tokens, 0, :].contiguous()


def _mhc_unpack_post_out(packed: torch.Tensor, num_tokens: int, mhc: int, h: int) -> torch.Tensor:
    """[num_tokens_pad * TILE, h] -> [num_tokens, mhc, h] (rows 0..mhc-1 of each block)."""
    return packed.view(-1, _MHC_TILE, h)[:num_tokens, :mhc, :].contiguous()


def _mhc_broadcast_row_to_tile(row: torch.Tensor) -> torch.Tensor:
    """Tile a [TILE] row to [TILE, TILE] by repeating it across all TILE rows."""
    return row.unsqueeze(0).expand(_MHC_TILE, _MHC_TILE).contiguous()


def _mhc_build_split_constant_tiles(
    hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int,
    post_mult: float, pre_eps: float,
):
    """Build the six [TILE, TILE] fp32 constant tiles for the split_mixes kernel.

    Returns (scale_tile, base_tile, pre_mask, pre_eps_tile, post_mult_mask, comb_mask).
    Each tile's rows are identical; a row has `mhc_mult3` valid cols with zeros
    in the rest. Matches tt-lang-kernels/pre_split_mixes.make_constant_tiles.
    """
    mhc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    sc = hc_scale.to(torch.float32)
    scale_vec = torch.cat([
        sc[0].expand(hc_mult),
        sc[1].expand(hc_mult),
        sc[2].expand(hc_mult * hc_mult),
    ])

    scale_row = torch.zeros(_MHC_TILE, dtype=torch.float32)
    scale_row[:mhc_mult3] = scale_vec
    base_row = torch.zeros(_MHC_TILE, dtype=torch.float32)
    base_row[:mhc_mult3] = hc_base.to(torch.float32)[:mhc_mult3]

    pre_mask_row = torch.zeros(_MHC_TILE, dtype=torch.float32)
    pre_mask_row[:hc_mult] = 1.0
    pre_eps_row = torch.zeros(_MHC_TILE, dtype=torch.float32)
    pre_eps_row[:hc_mult] = pre_eps
    post_mult_mask_row = torch.zeros(_MHC_TILE, dtype=torch.float32)
    post_mult_mask_row[hc_mult: 2 * hc_mult] = post_mult
    comb_mask_row = torch.zeros(_MHC_TILE, dtype=torch.float32)
    comb_mask_row[2 * hc_mult: mhc_mult3] = 1.0

    return (
        _mhc_broadcast_row_to_tile(scale_row),
        _mhc_broadcast_row_to_tile(base_row),
        _mhc_broadcast_row_to_tile(pre_mask_row),
        _mhc_broadcast_row_to_tile(pre_eps_row),
        _mhc_broadcast_row_to_tile(post_mult_mask_row),
        _mhc_broadcast_row_to_tile(comb_mask_row),
    )


def _mhc_sinkhorn_mask_tile(hc_mult: int) -> torch.Tensor:
    """[TILE, TILE] with 1s in the top-left hc_mult x hc_mult region, 0s elsewhere."""
    m = torch.zeros(_MHC_TILE, _MHC_TILE, dtype=torch.float32)
    m[:hc_mult, :hc_mult] = 1.0
    return m


def _mhc_sinkhorn_eps_mask_tile(hc_mult: int, eps: float) -> torch.Tensor:
    """[TILE, TILE] with `eps` in the top-left hc_mult x hc_mult region, 0s elsewhere.

    Adds eps only inside the valid region so the padded sentinel cells stay pad.
    """
    m = torch.zeros(_MHC_TILE, _MHC_TILE, dtype=torch.float32)
    m[:hc_mult, :hc_mult] = eps
    return m


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


def _compile_mhc_split_mixes_kernel(num_tiles: int):
    """Inlined from tt-lang-kernels/pre_split_mixes.py.

    Per-token elementwise split of `mixes[t, :mhc_mult3]` into three sections,
    each preserving the [num_tokens_pad, TILE] layout:
      - pre_out[t, :mhc]              = sigmoid(in * scale + base) + pre_eps
      - post_out[t, mhc:2*mhc]        = sigmoid(in * scale + base) * post_mult
      - comb_out[t, 2*mhc:2*mhc+mhc*mhc] = (in * scale + base) raw (no softmax)

    Constants `scale`/`base`/masks are each [TILE, TILE] with every row
    identical so the row-broadcasted elementwise ops all see the same values.
    """

    @ttl.operation(grid="auto")
    def split_mixes_kernel(
        input_mixes, scale_tile, base_tile,
        pre_mask, pre_eps_tile, post_mult_mask, comb_mask,
        pre_out, post_out, comb_out,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_tiles // total_cores)

        in_dfb = ttl.make_dataflow_buffer_like(input_mixes, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
        base_dfb = ttl.make_dataflow_buffer_like(base_tile, shape=(1, 1), block_count=1)
        prem_dfb = ttl.make_dataflow_buffer_like(pre_mask, shape=(1, 1), block_count=1)
        preeps_dfb = ttl.make_dataflow_buffer_like(pre_eps_tile, shape=(1, 1), block_count=1)
        postmm_dfb = ttl.make_dataflow_buffer_like(post_mult_mask, shape=(1, 1), block_count=1)
        combm_dfb = ttl.make_dataflow_buffer_like(comb_mask, shape=(1, 1), block_count=1)

        sig_dfb = ttl.make_dataflow_buffer_like(input_mixes, shape=(1, 1), block_count=2)

        pre_out_dfb = ttl.make_dataflow_buffer_like(pre_out, shape=(1, 1), block_count=2)
        post_out_dfb = ttl.make_dataflow_buffer_like(post_out, shape=(1, 1), block_count=2)
        comb_out_dfb = ttl.make_dataflow_buffer_like(comb_out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()
            base = base_dfb.wait()
            prem = prem_dfb.wait()
            preeps = preeps_dfb.wait()
            postmm = postmm_dfb.wait()
            combm = combm_dfb.wait()

            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    inp = in_dfb.wait()
                    sig_dfb.reserve().store(ttl.math.sigmoid(inp * sc + base))
                    comb_out_dfb.reserve().store((inp * sc + base) * combm)

                    sig = sig_dfb.wait()
                    pre_out_dfb.reserve().store(sig * prem + preeps)
                    post_out_dfb.reserve().store(sig * postmm)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scale_tile[0, 0], sc_dfb.reserve()).wait()
            ttl.copy(base_tile[0, 0], base_dfb.reserve()).wait()
            ttl.copy(pre_mask[0, 0], prem_dfb.reserve()).wait()
            ttl.copy(pre_eps_tile[0, 0], preeps_dfb.reserve()).wait()
            ttl.copy(post_mult_mask[0, 0], postmm_dfb.reserve()).wait()
            ttl.copy(comb_mask[0, 0], combm_dfb.reserve()).wait()
            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    ttl.copy(input_mixes[global_i, 0], in_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    ttl.copy(pre_out_dfb.wait(), pre_out[global_i, 0]).wait()
                    ttl.copy(post_out_dfb.wait(), post_out[global_i, 0]).wait()
                    ttl.copy(comb_out_dfb.wait(), comb_out[global_i, 0]).wait()

    return split_mixes_kernel


def _compile_mhc_sinkhorn_kernel(num_slices: int, repeat: int, eps: float):
    """Inlined from tt-lang-kernels/sinkhorn.py.

    Per-slice iterative normalize (softmax -> mask+eps -> col-norm ->
    (repeat-1) alternating row/col norms). Each 32x32 tile holds one 4x4 slice
    in its top-left, with `_MHC_PAD_SENTINEL` elsewhere so softmax pad cells
    underflow. Uses the scaler-style reduce/broadcast pattern; broadcasts and
    state carry full-tile shape.

    Compiler bug workaround: `options="--no-ttl-reduce-full-fp32"` — see
    tt-lang-kernels README on fp32 reduce(dim=1) returning zeros.
    """

    @ttl.operation(grid="auto", options="--no-ttl-reduce-full-fp32")
    def sinkhorn_kernel(x, mask, eps_mask, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        slices_per_core = -(-num_slices // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), block_count=1)
        em_dfb = ttl.make_dataflow_buffer_like(eps_mask, shape=(1, 1), block_count=1)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        state_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        state_copy_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        exp_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()
            m = m_dfb.wait()
            em = em_dfb.wait()

            for local_i in range(slices_per_core):
                global_i = core_idx * slices_per_core + local_i
                if global_i < num_slices:
                    # Row softmax: state := softmax(x, dim=-1)
                    x_in = x_dfb.wait()
                    red_dfb.reserve().store(ttl.math.reduce_max(x_in, sc, dims=[1]))
                    rmx = bc_dfb.reserve()
                    rmx.store(ttl.math.broadcast(red_dfb.wait(), rmx, dims=[1]))
                    exp_dfb.reserve().store(ttl.math.exp(x_in - bc_dfb.wait()))

                    ex = exp_dfb.wait()
                    red_dfb.reserve().store(ttl.math.reduce_sum(ex, sc, dims=[1]))
                    state_copy_dfb.reserve().store(ex)
                    rinv = bc_dfb.reserve()
                    rinv.store(ttl.math.broadcast(ttl.math.recip(red_dfb.wait()), rinv, dims=[1]))
                    state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                    # Mask + eps: state := state * mask + eps_mask
                    state_copy_dfb.reserve().store(state_dfb.wait() * m + em)
                    state_dfb.reserve().store(state_copy_dfb.wait())

                    # First col-normalize: state := state / (col_sum + eps)
                    s = state_dfb.wait()
                    red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[0]))
                    state_copy_dfb.reserve().store(s)
                    cinv = bc_dfb.reserve()
                    csum = red_dfb.wait()
                    cinv.store(ttl.math.broadcast(
                        ttl.math.recip(csum + ttl.math.fill(csum, eps)),
                        cinv, dims=[0]))
                    state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                    for _ in range(repeat - 1):
                        s = state_dfb.wait()
                        red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[1]))
                        state_copy_dfb.reserve().store(s)
                        rinv = bc_dfb.reserve()
                        rsum = red_dfb.wait()
                        rinv.store(ttl.math.broadcast(
                            ttl.math.recip(rsum + ttl.math.fill(rsum, eps)),
                            rinv, dims=[1]))
                        state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                        s = state_dfb.wait()
                        red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[0]))
                        state_copy_dfb.reserve().store(s)
                        cinv = bc_dfb.reserve()
                        csum = red_dfb.wait()
                        cinv.store(ttl.math.broadcast(
                            ttl.math.recip(csum + ttl.math.fill(csum, eps)),
                            cinv, dims=[0]))
                        state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                    out_dfb.reserve().store(state_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(mask[0, 0], m_dfb.reserve()).wait()
            ttl.copy(eps_mask[0, 0], em_dfb.reserve()).wait()
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_i in range(slices_per_core):
                global_i = core_idx * slices_per_core + local_i
                if global_i < num_slices:
                    ttl.copy(x[global_i, 0], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_i in range(slices_per_core):
                global_i = core_idx * slices_per_core + local_i
                if global_i < num_slices:
                    ttl.copy(out_dfb.wait(), out[global_i, 0]).wait()

    return sinkhorn_kernel


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
      2. split_mixes (device): mixes -> (pre_raw, post, comb_raw), all in
         [num_tokens_pad, TILE] column-packed layout.
      3. Host round-trip on comb: unpack to [num_tokens, mhc, mhc], re-pack into
         the sinkhorn per-slice tile layout (4x4 in top-left, PAD_SENTINEL
         elsewhere). pre and post also come back with comb on the same read.
      4. sinkhorn (device): iterative softmax/normalize -> comb [num_tokens, mhc, mhc]
      5. pre_apply_mix (device): x[num_tokens, mhc, h] reduced by pre[num_tokens, mhc, 1].

    Pipeline (per Block.hc_post call):
      1. post: x[num_tokens, h] * post_mix + comb^T @ residual -> [num_tokens, mhc, h]

    The host repack between split_mixes and sinkhorn bridges two incompatible
    tile layouts (`[num_tokens_pad, TILE]` vs `[num_tokens_pad*TILE, TILE]`).
    A future device repack kernel could replace it without changing either
    endpoint. Decode pads num_tokens to TILE rows so the kernels'
    tiles-per-core math holds; padded rows hold zeros and contribute nothing.
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

        # split_mixes constant tiles (all [TILE, TILE] fp32, row-broadcast).
        (scale_tile, base_tile, pre_mask_tile, pre_eps_tile,
         post_mult_mask_tile, comb_mask_tile) = _mhc_build_split_constant_tiles(
            hc_scale.detach().to(torch.float32).cpu(),
            hc_base.detach().to(torch.float32).cpu(),
            self.hc_mult, _MHC_POST_MULT, self.hc_eps,
        )
        self.split_scale_tt = ttnn.as_tensor(scale_tile, **rep)
        self.split_base_tt = ttnn.as_tensor(base_tile, **rep)
        self.split_pre_mask_tt = ttnn.as_tensor(pre_mask_tile, **rep)
        self.split_pre_eps_tt = ttnn.as_tensor(pre_eps_tile, **rep)
        self.split_post_mult_mask_tt = ttnn.as_tensor(post_mult_mask_tile, **rep)
        self.split_comb_mask_tt = ttnn.as_tensor(comb_mask_tile, **rep)

        # sinkhorn constant tiles.
        self.sk_mask_tt = ttnn.as_tensor(
            _mhc_sinkhorn_mask_tile(self.hc_mult), **rep)
        self.sk_eps_mask_tt = ttnn.as_tensor(
            _mhc_sinkhorn_eps_mask_tile(self.hc_mult, self.hc_eps), **rep)

        self._kernels: dict = {}

        # Device tensors stashed by hc_pre and consumed by the matching
        # hc_post call. None when no hc_pre has run yet.
        self._stash_a_tt = None
        self._stash_post_tt = None
        self._stash_comb_sk_tt = None

        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """Per-step output buffers for the tt-lang kernels and upload
        buffers for the residual / x feeds. Allocated once at num_tokens=1
        (loop-prefill: every step is a single token). The kernels
        overwrite every tile they own, so no per-call zeroing."""
        ttnn = self._ttnn
        num_tokens_pad = _MHC_TILE
        num_tokens = 1
        self._mixes_tt = self._zeros((num_tokens_pad, _MHC_TILE))
        self._pre_tt = self._zeros((num_tokens_pad, _MHC_TILE))
        self._post_tt = self._zeros((num_tokens_pad, _MHC_TILE))
        self._comb_tt = self._zeros((num_tokens_pad, _MHC_TILE))
        self._comb_sk_out_tt = self._zeros((num_tokens * _MHC_TILE, _MHC_TILE))
        self._apply_mix_out_tt = self._zeros((num_tokens * _MHC_TILE, self.hidden))
        self._post_out_tt = self._zeros((num_tokens * _MHC_TILE, self.hidden))

        # Upload buffers. hc_pre uploads residual packed as [num_tokens_pad,
        # mhc*hidden]; hc_post uploads x packed as [num_tokens * TILE, hidden].
        self._a_upload_tt = ttnn.from_torch(
            torch.zeros(num_tokens_pad, self.hc_mult * self.hidden,
                        dtype=torch.float32),
            device=self.mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self._x_upload_tt = ttnn.from_torch(
            torch.zeros(num_tokens * _MHC_TILE, self.hidden,
                        dtype=torch.float32),
            device=self.mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self._upload_mapper = ttnn.ReplicateTensorToMesh(self.mesh)

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

    def _split_mixes_kernel(self, num_out_tiles: int):
        key = ("split_mixes", num_out_tiles)
        k = self._kernels.get(key)
        if k is None:
            k = _compile_mhc_split_mixes_kernel(num_tiles=num_out_tiles)
            self._kernels[key] = k
        return k

    def _sinkhorn_kernel(self, num_slices: int):
        key = ("sinkhorn", num_slices, self.sinkhorn_iters)
        k = self._kernels.get(key)
        if k is None:
            k = _compile_mhc_sinkhorn_kernel(
                num_slices=num_slices,
                repeat=self.sinkhorn_iters,
                eps=self.hc_eps,
            )
            self._kernels[key] = k
        return k

    def _apply_mix_kernel(self, num_tokens: int):
        key = ("apply_mix", num_tokens)
        k = self._kernels.get(key)
        if k is None:
            k = _compile_mhc_apply_mix_kernel(
                num_tokens=num_tokens,
                h_tiles=self.hidden // _MHC_TILE,
            )
            self._kernels[key] = k
        return k

    def _post_kernel(self, num_tokens: int):
        key = ("post", num_tokens)
        k = self._kernels.get(key)
        if k is None:
            k = _compile_mhc_post_kernel(
                num_tokens=num_tokens,
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
        """[B, S, mhc, hidden] -> y [B, S, hidden].

        post and comb stay on device (stashed for the matching hc_post call);
        only y, the apply_mix output that downstream norm/attn/ffn need, is
        downloaded.
        """
        ttnn = self._ttnn
        B, S, mhc, hidden = x.shape
        if mhc != self.hc_mult or hidden != self.hidden:
            raise ValueError(
                f"x shape mismatch: got mhc={mhc}, hidden={hidden}; "
                f"expected mhc={self.hc_mult}, hidden={self.hidden}")
        out_dtype = x.dtype
        num_tokens = B * S
        num_tokens_pad = -(-num_tokens // _MHC_TILE) * _MHC_TILE

        # Step 1: pre_norm_fn on device -> mixes_tt [num_tokens_pad, TILE].
        # norm_fn / split_mixes process 32 tokens per tile, so they keep the
        # num_tokens_pad sizing. Per-token kernels (sinkhorn, apply_mix, post)
        # use real num_tokens to avoid 32x over-allocation on decode.
        a_packed = _mhc_pack_residual(x, num_tokens_pad)
        host_mesh = ttnn.from_torch(
            a_packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._a_upload_tt)
        a_tt = self._a_upload_tt
        mixes_tt = self._mixes_tt
        self._norm_fn_kernel(num_tokens_pad // _MHC_TILE)(
            a_tt, self.fn_tt, self.scaler_tt, mixes_tt)

        # Step 2: split_mixes on device -> pre/post/comb_raw (each
        # [num_tokens_pad, TILE], column-packed sections).
        pre_tt = self._pre_tt
        post_tt = self._post_tt
        comb_tt = self._comb_tt
        self._split_mixes_kernel(num_tokens_pad // _MHC_TILE)(
            mixes_tt, self.split_scale_tt, self.split_base_tt,
            self.split_pre_mask_tt, self.split_pre_eps_tt,
            self.split_post_mult_mask_tt, self.split_comb_mask_tt,
            pre_tt, post_tt, comb_tt,
        )

        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))

        # Step 3: build the sinkhorn input on device. comb_tt has 32 tokens per
        # tile with each token's flat 4x4 in cols 2*mhc..2*mhc+mhc*mhc. Sinkhorn
        # expects one tile per token with the 4x4 in the top-left and
        # PAD_SENTINEL elsewhere. slice -> reshape -> pad -> reshape replaces
        # the prior CPU repack round-trip.
        comb_sliced = ttnn.slice(
            comb_tt, [0, 2 * mhc], [num_tokens, 2 * mhc + mhc * mhc])
        comb_3d = ttnn.reshape(comb_sliced, [num_tokens, mhc, mhc])
        comb_padded = ttnn.pad(
            comb_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, _MHC_TILE - mhc)],
            value=_MHC_PAD_SENTINEL,
        )
        comb_sk_in_tt = ttnn.reshape(
            comb_padded, [num_tokens * _MHC_TILE, _MHC_TILE])
        comb_sk_out_tt = self._comb_sk_out_tt
        self._sinkhorn_kernel(num_tokens)(
            comb_sk_in_tt, self.sk_mask_tt, self.sk_eps_mask_tt, self.scaler_tt,
            comb_sk_out_tt,
        )

        # Step 4: pre_apply_mix on device. mix layout is built from pre_tt via
        # slice + reshape + pad (col 0 of rows 0..mhc-1 holds pre[t, m]).
        pre_sliced = ttnn.slice(pre_tt, [0, 0], [num_tokens, mhc])
        pre_3d = ttnn.reshape(pre_sliced, [num_tokens, mhc, 1])
        mix_padded = ttnn.pad(
            pre_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, _MHC_TILE - 1)],
            value=0.0,
        )
        mix_tt = ttnn.reshape(mix_padded, [num_tokens * _MHC_TILE, _MHC_TILE])

        # Build apply_mix x layout from the already-uploaded a_tt (the
        # norm_fn input) instead of re-uploading the residual in a different
        # shape. a_tt is [num_tokens_pad, mhc*hidden]; we need [num_tokens *
        # TILE, hidden] with rows 0..mhc-1 of each TILE-block holding x[t,m,:].
        a_sliced = ttnn.slice(a_tt, [0, 0], [num_tokens, mhc * hidden])
        a_3d = ttnn.reshape(a_sliced, [num_tokens, mhc, hidden])
        x_padded = ttnn.pad(
            a_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, 0)],
            value=0.0,
        )
        x_tt = ttnn.reshape(x_padded, [num_tokens * _MHC_TILE, hidden])
        out_tt = self._apply_mix_out_tt
        self._apply_mix_kernel(num_tokens)(
            x_tt, mix_tt, self.scaler_tt, out_tt)

        # Stash device tensors hc_post consumes directly.
        self._stash_a_tt = a_tt
        self._stash_post_tt = post_tt
        self._stash_comb_sk_tt = comb_sk_out_tt

        out_packed = ttnn.to_torch(out_tt, mesh_composer=composer)
        y_flat = _mhc_unpack_apply_mix_out(out_packed, num_tokens, hidden)
        return y_flat.view(B, S, hidden).to(out_dtype)

    def hc_post(self, x: torch.Tensor):
        """x [B, S, hidden] -> [B, S, mhc, hidden].

        residual/post/comb are consumed from device tensors stashed by the
        matching hc_pre call. Only x (post-attn or post-FFN) is uploaded.
        """
        ttnn = self._ttnn
        B, S, hidden = x.shape
        mhc = self.hc_mult
        out_dtype = x.dtype
        num_tokens = B * S

        a_tt_stashed = self._stash_a_tt
        post_tt_stashed = self._stash_post_tt
        comb_sk_tt_stashed = self._stash_comb_sk_tt
        if a_tt_stashed is None or post_tt_stashed is None or comb_sk_tt_stashed is None:
            raise RuntimeError("hc_post called without matching hc_pre stash")
        # Forbid reuse: a stash belongs to one hc_pre/hc_post pair.
        self._stash_a_tt = None
        self._stash_post_tt = None
        self._stash_comb_sk_tt = None

        x_2d = x.reshape(num_tokens, hidden)
        x_packed = _mhc_pack_x_bc(x_2d, mhc, num_tokens)
        host_mesh = ttnn.from_torch(
            x_packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)
        x_tt = self._x_upload_tt

        # Build res_tt from the stashed a_tt (norm_fn input == hc_pre's
        # residual, packed [num_tokens_pad, mhc*hidden]). Slice -> reshape ->
        # pad -> reshape gives [num_tokens * TILE, hidden] with rows 0..mhc-1
        # of each TILE-block holding residual[t, m, :].
        a_sliced = ttnn.slice(a_tt_stashed, [0, 0], [num_tokens, mhc * hidden])
        a_3d = ttnn.reshape(a_sliced, [num_tokens, mhc, hidden])
        res_padded = ttnn.pad(
            a_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, 0)],
            value=0.0,
        )
        res_tt = ttnn.reshape(res_padded, [num_tokens * _MHC_TILE, hidden])

        # Build post_tt from stashed split_mixes output. post values live in
        # cols mhc..2*mhc of post_tt_stashed [num_tokens_pad, TILE].
        post_sliced = ttnn.slice(
            post_tt_stashed, [0, mhc], [num_tokens, 2 * mhc])
        post_3d = ttnn.reshape(post_sliced, [num_tokens, mhc, 1])
        post_padded = ttnn.pad(
            post_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, _MHC_TILE - 1)],
            value=0.0,
        )
        post_tt = ttnn.reshape(post_padded, [num_tokens * _MHC_TILE, _MHC_TILE])

        # Build comb_tt (transposed) from stashed sinkhorn output. The post
        # kernel reads only the top-left mhc x mhc of each TILE-block, so
        # transposing the full TILE x TILE is correct: the new top-left
        # mhc x mhc is the transpose of the original top-left mhc x mhc.
        comb_3d = ttnn.reshape(
            comb_sk_tt_stashed, [num_tokens, _MHC_TILE, _MHC_TILE])
        comb_T_3d = ttnn.transpose(comb_3d, -2, -1)
        comb_tt = ttnn.reshape(
            comb_T_3d, [num_tokens * _MHC_TILE, _MHC_TILE])

        out_tt = self._post_out_tt
        self._post_kernel(num_tokens)(
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
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """Pre-allocate per-step buffers for the M=1 SwiGLU pipeline. Every
        ttnn op below feeds output_tensor= / optional_output_tensor=."""
        ttnn = self._ttnn
        common_rep = dict(
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self._x_upload_tt = ttnn.from_torch(
            torch.zeros(1, self.dim, dtype=torch.bfloat16), **common_rep)
        self._y1_tt = ttnn.from_torch(
            torch.zeros(1, self.inter_dim, dtype=torch.bfloat16), **common_rep)
        self._y3_tt = ttnn.from_torch(
            torch.zeros(1, self.inter_dim, dtype=torch.bfloat16), **common_rep)
        self._silu_tt = ttnn.from_torch(
            torch.zeros(1, self.inter_dim, dtype=torch.bfloat16), **common_rep)
        self._mid_tt = ttnn.from_torch(
            torch.zeros(1, self.inter_dim, dtype=torch.bfloat16), **common_rep)
        self._out_tt = ttnn.from_torch(
            torch.zeros(1, self.dim, dtype=torch.bfloat16), **common_rep)
        self._upload_mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        self._trace_id = None

    def _compute_body(self):
        """Pure-device SwiGLU pipeline. All ops write into pre-allocated
        buffers so trace replay reuses the same addresses."""
        ttnn = self._ttnn
        x_tt = self._x_upload_tt
        ttnn.matmul(x_tt, self.w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._y1_tt)
        ttnn.matmul(x_tt, self.w3_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._y3_tt)
        if self.swiglu_limit > 0:
            ttnn.clamp(self._y1_tt, max=self.swiglu_limit, output_tensor=self._y1_tt)
            ttnn.clamp(self._y3_tt, min=-self.swiglu_limit, max=self.swiglu_limit,
                       output_tensor=self._y3_tt)
        ttnn.silu(self._y1_tt, output_tensor=self._silu_tt)
        ttnn.multiply(self._silu_tt, self._y3_tt, output_tensor=self._mid_tt)
        ttnn.matmul(self._mid_tt, self.w2_tt,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._out_tt)

    def _capture_trace(self):
        ttnn = self._ttnn
        self._compute_body()  # warmup
        ttnn.synchronize_device(self.mesh)
        self._trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._compute_body()
        ttnn.end_trace_capture(self.mesh, self._trace_id, cq_id=0)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [M=1, dim] -> [M=1, dim]. weights matches Expert.forward
        signature; shared expert is always called with weights=None."""
        if weights is not None:
            raise ValueError("DeviceSharedExpert does not support per-token `weights`")
        ttnn = self._ttnn
        M, D = x.shape
        if D != self.dim:
            raise ValueError(f"x last dim {D} != expected {self.dim}")
        if M != 1:
            raise ValueError(f"DeviceSharedExpert expects M=1, got M={M}")
        host_mesh = ttnn.from_torch(
            x.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)
        if self._trace_id is None:
            self._capture_trace()
        else:
            ttnn.execute_trace(self.mesh, self._trace_id, cq_id=0, blocking=True)

        composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(1, 0))
        y = ttnn.to_torch(self._out_tt, mesh_composer=composer)[:M]
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
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """No per-step buffers yet; intermediates (gather/scores/probs)
        still allocate per-call. Reserved entry point for follow-up
        hoists once tracing exposes the fixed-shape pattern."""
        return

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

    def _idxs_int_tile_to_idxs_and_mask(self, idxs_int_tt, B: int, S: int, K: int):
        """Convert a device-resident [B, S, K] int32 TILE topk index tensor
        (with -1 sentinels) into (idxs_uint32_rm [B, S*K], valid_bf16_tile
        [B, S, 1, K]). Used by the device-indexer path to avoid a CPU round
        trip on the indices."""
        ttnn = self._ttnn
        cond = ttnn.lt(idxs_int_tt, 0)
        cond_4d = ttnn.reshape(cond, [B, S, 1, K])
        valid_tt = ttnn.where(cond_4d, float("-inf"), 0.0)
        safe = ttnn.clamp(idxs_int_tt, min=0)
        safe = ttnn.reshape(safe, [B, S * K])
        safe = ttnn.typecast(safe, dtype=ttnn.uint32)
        safe = ttnn.to_layout(safe, ttnn.ROW_MAJOR_LAYOUT)
        return safe, valid_tt

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


def _device_act_quant_block(ttnn, x_tt, block_size: int,
                            fp8_max: float = 448.0, eps: float = 1e-4):
    """Block-wise act_quant along last dim, bf16 round-trip.

    Mirrors CPU `act_quant(..., inplace=True)` without the fp8 e4m3 cast
    (bf16 precision policy). Algorithm stays on device so a future fp8
    emulation kernel can slot in here.

      z = reshape(x, [..., N/block, block])
      amax = max(|z|, -1).clamp(min=eps)
      s = amax / fp8_max
      out = clamp(z/s, -fp8_max, fp8_max) * s
      return out reshaped back

    TODO: top tt-lang kernel candidate. All elementwise + one reduce on a
    [..., nb, block] reshape; fuses cleanly into a single compute body.
    """
    orig_shape = list(x_tt.shape)
    N = orig_shape[-1]
    if N % block_size != 0:
        raise ValueError(f"last dim {N} must be divisible by block_size {block_size}")
    nb = N // block_size
    blocked = ttnn.reshape(x_tt, orig_shape[:-1] + [nb, block_size])
    amax = ttnn.max(ttnn.abs(blocked), dim=-1, keepdim=True)
    amax = ttnn.maximum(amax, eps)
    s = ttnn.multiply(amax, 1.0 / fp8_max)
    y = ttnn.divide(blocked, s)
    y = ttnn.clamp(y, -fp8_max, fp8_max)
    out = ttnn.multiply(y, s)
    return ttnn.reshape(out, orig_shape)


def _sylvester_hadamard(d: int) -> torch.Tensor:
    """Sylvester-construction Hadamard matrix H_d of size [d, d] with +/-1
    entries. Symmetric. Matches the natural-order butterfly produced by
    rotate_activation."""
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"d {d} must be a positive power of 2")
    H = torch.tensor([[1.0]])
    n = 1
    while n < d:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        n *= 2
    return H


def _device_rotate_activation(ttnn, x_tt, h_tt):
    """Walsh-Hadamard transform along last dim, scaled by 1/sqrt(d).

    h_tt is a precomputed device tensor of shape [d, d] holding H_d/sqrt(d).
    Single matmul replaces the 7-stage butterfly (for d=128) and broadcasts
    over arbitrary leading dims.
    """
    return ttnn.matmul(x_tt, h_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class DeviceCompressor(nn.Module):
    """Decode-only Compressor on a 1xN mesh.

    Holds device-resident state (kv_state, score_state, kv_cache) and runs
    the full Compressor.forward decode branch on device. Validated cases
    (see ./device-compressor-indexer/):
      - overlap=False, rotate=False  (Attention layers with ratio=128)
      - overlap=True,  rotate=False  (Attention layers with ratio=4)
      - overlap=True,  rotate=True   (Indexer's internal compressor; ratio=4)

    fp4_act_quant / act_quant are omitted (bf16 policy; identity in bf16).

    If `slot_offset` is non-zero, kv_cache writes target the *slot_offset+i*-th
    row of `kv_cache_tt`. This lets the Attention layer point us at a slice
    of its joint window+compress kv_cache buffer (slot_offset == window_size)
    so compressed tokens land in the right place without an extra mirror.

    State buffers split the CPU `[B, ratio_pad, coff*head_dim]` shape into
    front (..., :head_dim) and back (..., head_dim:) buffers when overlap=True.
    Without the split, the last-dim 2*head_dim trips an L1 overflow in
    `update_cache_for_token_`. See compressor_overlap.py for details.
    """

    def __init__(self, mesh, comp, wkv_dev, wgate_dev, norm_dev,
                 slot_offset: int = 0, kv_cache_tt=None):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        self.comp = comp
        self.wkv = wkv_dev
        self.wgate = wgate_dev
        self.norm_dev = norm_dev
        self.head_dim = comp.head_dim
        self.rope_head_dim = comp.rope_head_dim
        self.compress_ratio = comp.compress_ratio
        self.overlap = bool(comp.overlap)
        self.rotate = bool(comp.rotate)
        self.coff = 2 if self.overlap else 1
        self.cdim = self.coff * self.head_dim
        self.row_count = 2 * self.compress_ratio if self.overlap else self.compress_ratio
        self.slot_offset = slot_offset

        rep = dict(
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.ape_tt = ttnn.as_tensor(comp.ape.to(torch.bfloat16).contiguous(), **rep)

        fc = comp.freqs_cis
        self.cos_full_tt = ttnn.as_tensor(fc.real.to(torch.bfloat16).contiguous(), **rep)
        self.sin_full_tt = ttnn.as_tensor(fc.imag.to(torch.bfloat16).contiguous(), **rep)

        if self.rotate:
            h_mat = (_sylvester_hadamard(self.head_dim) *
                     (self.head_dim ** -0.5)).to(torch.bfloat16)
            self.h_tt = ttnn.as_tensor(h_mat, **rep)
        else:
            self.h_tt = None

        self.kv_state_front_tt = None
        self.kv_state_back_tt = None
        self.score_state_front_tt = None
        self.score_state_back_tt = None
        self.kv_cache_tt = kv_cache_tt
        self._owns_kv_cache = kv_cache_tt is None

    def bind_kv_cache_tt(self, kv_cache_tt, slot_offset: int):
        """Attach a pre-existing kv_cache buffer (e.g. the Attention's joint
        window+compress kv_cache_tt). Call before the first forward_device.
        Compressed-token writes will land at slot_offset + (start_pos // ratio).
        """
        self.kv_cache_tt = kv_cache_tt
        self.slot_offset = slot_offset
        self._owns_kv_cache = False

    def _alloc_decode_tensors(self, B: int = 1):
        """Allocate per-step state buffers (kv_state/score_state/kv_cache).
        Decode runs B=1; the orchestrator calls this with the default.
        Idempotent — re-running rebinds fresh device buffers from the
        current CPU state."""
        ttnn = self._ttnn
        comp = self.comp
        d = self.head_dim
        ratio_pad = -(-self.row_count // 32) * 32
        self.ratio_pad = ratio_pad

        kv_init = comp.kv_state[:B].to(torch.bfloat16)               # [B, row_count, c]
        score_init = comp.score_state[:B].to(torch.bfloat16)
        if self.overlap:
            kv_init_front = kv_init[..., :d]
            kv_init_back = kv_init[..., d:]
            score_init_front = score_init[..., :d]
            score_init_back = score_init[..., d:]
        else:
            kv_init_front = kv_init
            score_init_front = score_init
            kv_init_back = score_init_back = None

        if ratio_pad != self.row_count:
            zero_pad = torch.zeros(B, ratio_pad - self.row_count, d, dtype=torch.bfloat16)
            ninf_pad = torch.full_like(zero_pad, float("-inf"))
            kv_init_front = torch.cat([kv_init_front, zero_pad], dim=1)
            score_init_front = torch.cat([score_init_front, ninf_pad], dim=1)
            if self.overlap:
                kv_init_back = torch.cat([kv_init_back, zero_pad], dim=1)
                score_init_back = torch.cat([score_init_back, ninf_pad], dim=1)

        rep = dict(
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self.kv_state_front_tt = ttnn.as_tensor(
            kv_init_front.view(B, 1, ratio_pad, d).contiguous(), **rep)
        self.score_state_front_tt = ttnn.as_tensor(
            score_init_front.view(B, 1, ratio_pad, d).contiguous(), **rep)
        if self.overlap:
            self.kv_state_back_tt = ttnn.as_tensor(
                kv_init_back.view(B, 1, ratio_pad, d).contiguous(), **rep)
            self.score_state_back_tt = ttnn.as_tensor(
                score_init_back.view(B, 1, ratio_pad, d).contiguous(), **rep)

        if self._owns_kv_cache:
            T = comp.kv_cache.shape[1]
            T_pad = -(-T // 32) * 32
            kv_cache_init = comp.kv_cache[:B].to(torch.bfloat16)
            if T_pad != T:
                pad = torch.zeros(B, T_pad - T, d, dtype=torch.bfloat16)
                kv_cache_init = torch.cat([kv_cache_init, pad], dim=1)
            self.kv_cache_tt = ttnn.as_tensor(
                kv_cache_init.view(B, 1, T_pad, d).contiguous(), **rep)

    def forward_device(self, x_tt, B: int, start_pos: int):
        ttnn = self._ttnn
        ratio = self.compress_ratio
        d = self.head_dim
        c = self.cdim
        rd = self.rope_head_dim

        if self.kv_state_front_tt is None:
            self._alloc_decode_tensors(B)

        kv_tt = self.wkv.forward_device(x_tt)
        score_tt = self.wgate.forward_device(x_tt)

        slot_in_ape = start_pos % ratio
        ape_slot = ttnn.slice(self.ape_tt, [slot_in_ape, 0], [slot_in_ape + 1, c])
        score_tt = ttnn.add(score_tt, ttnn.reshape(ape_slot, [1, 1, c]))

        if self.overlap:
            kv_front = ttnn.slice(kv_tt, [0, 0, 0], [B, 1, d])
            kv_back  = ttnn.slice(kv_tt, [0, 0, d], [B, 1, c])
            score_front = ttnn.slice(score_tt, [0, 0, 0], [B, 1, d])
            score_back  = ttnn.slice(score_tt, [0, 0, d], [B, 1, c])
            slot = ratio + slot_in_ape
            ttnn.kv_cache.update_cache_for_token_(self.kv_state_front_tt,
                                                  ttnn.reshape(kv_front, [B, 1, 1, d]), slot, 0)
            ttnn.kv_cache.update_cache_for_token_(self.kv_state_back_tt,
                                                  ttnn.reshape(kv_back, [B, 1, 1, d]), slot, 0)
            ttnn.kv_cache.update_cache_for_token_(self.score_state_front_tt,
                                                  ttnn.reshape(score_front, [B, 1, 1, d]), slot, 0)
            ttnn.kv_cache.update_cache_for_token_(self.score_state_back_tt,
                                                  ttnn.reshape(score_back, [B, 1, 1, d]), slot, 0)
        else:
            slot = slot_in_ape
            ttnn.kv_cache.update_cache_for_token_(self.kv_state_front_tt,
                                                  ttnn.reshape(kv_tt, [B, 1, 1, d]), slot, 0)
            ttnn.kv_cache.update_cache_for_token_(self.score_state_front_tt,
                                                  ttnn.reshape(score_tt, [B, 1, 1, d]), slot, 0)

        if (start_pos + 1) % ratio != 0:
            return None

        # TODO(tt-lang): fuse the slice/concat view + softmax-sum + RMSNorm
        # into one `compressor_softmax_sum_norm` kernel. See
        # device-compressor-indexer/README.md candidate #1.
        if self.overlap:
            front_kv = ttnn.slice(self.kv_state_front_tt, [0, 0, 0, 0],     [B, 1, ratio,     d])
            back_kv  = ttnn.slice(self.kv_state_back_tt,  [0, 0, ratio, 0], [B, 1, 2 * ratio, d])
            kv_view = ttnn.concat([front_kv, back_kv], dim=2)
            front_sc = ttnn.slice(self.score_state_front_tt, [0, 0, 0, 0],     [B, 1, ratio,     d])
            back_sc  = ttnn.slice(self.score_state_back_tt,  [0, 0, ratio, 0], [B, 1, 2 * ratio, d])
            score_view = ttnn.concat([front_sc, back_sc], dim=2)
        else:
            kv_view = self.kv_state_front_tt
            score_view = self.score_state_front_tt

        sm_tt = ttnn.softmax(score_view, dim=-2)
        weighted = ttnn.multiply(kv_view, sm_tt)
        kv_sum = ttnn.sum(weighted, dim=-2, keepdim=True)

        kv_2d = ttnn.reshape(kv_sum, [B, d])
        if B < _RMS_TILE:
            kv_2d = ttnn.pad(kv_2d, padding=[(0, _RMS_TILE - B), (0, 0)], value=0.0)
        kv_2d = self.norm_dev.forward_device(kv_2d, B)
        if B < _RMS_TILE:
            kv_2d = ttnn.slice(kv_2d, [0, 0], [B, d])
        kv_normed = ttnn.reshape(kv_2d, [B, 1, d])

        rd_half = rd // 2
        freq_idx = start_pos + 1 - ratio
        cos = ttnn.slice(self.cos_full_tt, [freq_idx, 0], [freq_idx + 1, rd_half])
        sin = ttnn.slice(self.sin_full_tt, [freq_idx, 0], [freq_idx + 1, rd_half])
        cos = ttnn.reshape(cos, [1, 1, rd_half])
        sin = ttnn.reshape(sin, [1, 1, rd_half])
        kv_nope = ttnn.slice(kv_normed, [0, 0, 0],     [B, 1, d - rd])
        kv_rope = ttnn.slice(kv_normed, [0, 0, d - rd], [B, 1, d])
        kv_rope = _device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)
        kv_normed = ttnn.concat([kv_nope, kv_rope], dim=-1)

        if self.rotate:
            kv_normed = _device_rotate_activation(ttnn, kv_normed, self.h_tt)

        comp_idx = start_pos // ratio
        kv_4d_out = ttnn.reshape(kv_normed, [B, 1, 1, d])
        ttnn.kv_cache.update_cache_for_token_(self.kv_cache_tt, kv_4d_out,
                                              self.slot_offset + comp_idx, 0)

        # TODO(tt-lang): the per-buffer ratio-many slot-shift writes below are
        # the second tt-lang candidate (compressor_state_shift). One kernel
        # that copies [B, ratio, d] from slots ratio..2*ratio-1 to 0..ratio-1
        # in a single dispatch is much cleaner than the loop below.
        if self.overlap:
            for buf in (self.kv_state_front_tt, self.kv_state_back_tt,
                        self.score_state_front_tt, self.score_state_back_tt):
                for i in range(ratio):
                    slot_src = ttnn.slice(buf, [0, 0, ratio + i, 0],
                                               [B, 1, ratio + i + 1, d])
                    ttnn.kv_cache.update_cache_for_token_(buf, slot_src, i, 0)

        return kv_normed


class DeviceIndexer(nn.Module):
    """Decode-only Indexer on a 1xN mesh.

    Owns:
      - inner DeviceCompressor (overlap=True, rotate=True)
      - DeviceColLinear for wq_b and weights_proj
      - Hadamard table for q rotate

    Returns post-reduce score [B, 1, T_pad] as a *device* tensor; caller
    pulls it to CPU, masks slots >= end_pos // ratio, and runs topk. Topk
    on device is the future fusion target (README candidate #4).
    """

    def __init__(self, mesh, indexer, dc: DeviceCompressor,
                 wq_b_dev, weights_proj_dev):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.indexer = indexer
        self.dc = dc
        self.wq_b = wq_b_dev
        self.weights_proj = weights_proj_dev

        self.head_dim = indexer.head_dim
        self.rope_head_dim = indexer.rope_head_dim
        self.n_heads = indexer.n_heads
        self.compress_ratio = indexer.compress_ratio
        self.softmax_scale = float(indexer.softmax_scale)
        self.index_topk = indexer.index_topk

        rep = dict(
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        h_mat = (_sylvester_hadamard(self.head_dim) *
                 (self.head_dim ** -0.5)).to(torch.bfloat16)
        self.h_tt = ttnn.as_tensor(h_mat, **rep)
        fc = indexer.freqs_cis
        self.cos_full_tt = ttnn.as_tensor(fc.real.to(torch.bfloat16).contiguous(), **rep)
        self.sin_full_tt = ttnn.as_tensor(fc.imag.to(torch.bfloat16).contiguous(), **rep)
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """No per-step buffers yet; q/cos/sin slices and score temporaries
        still allocate per-call. Reserved entry point for the future
        `indexer_score_reduce` fused kernel."""
        return

    def forward_device_score(self, x_tt, qr_tt, B: int, start_pos: int):
        ttnn = self._ttnn
        H = self.n_heads
        D = self.head_dim
        rd = self.rope_head_dim

        q_tt = self.wq_b.forward_device(qr_tt)
        q_tt = ttnn.reshape(q_tt, [B, 1, H, D])

        rd_half = rd // 2
        cos = ttnn.slice(self.cos_full_tt, [start_pos, 0], [start_pos + 1, rd_half])
        sin = ttnn.slice(self.sin_full_tt, [start_pos, 0], [start_pos + 1, rd_half])
        cos = ttnn.reshape(cos, [1, 1, 1, rd_half])
        sin = ttnn.reshape(sin, [1, 1, 1, rd_half])
        q_nope = ttnn.slice(q_tt, [0, 0, 0, 0],     [B, 1, H, D - rd])
        q_rope = ttnn.slice(q_tt, [0, 0, 0, D - rd], [B, 1, H, D])
        q_rope = _device_apply_rotary_interleaved(ttnn, q_rope, cos, sin, inverse=False)
        q_tt = ttnn.concat([q_nope, q_rope], dim=-1)
        q_tt = _device_rotate_activation(ttnn, q_tt, self.h_tt)

        # fp4_act_quant SKIPPED (bf16 policy).
        self.dc.forward_device(x_tt, B, start_pos)

        scale = self.softmax_scale * (H ** -0.5)
        w_tt = self.weights_proj.forward_device(x_tt)
        w_tt = ttnn.multiply(w_tt, scale)

        kv_T_tt = ttnn.transpose(self.dc.kv_cache_tt, -2, -1)   # [B, 1, D, T_pad]
        score = ttnn.matmul(q_tt, kv_T_tt)                        # [B, 1, H, T_pad]

        # TODO(tt-lang): fuse relu * weights * sum + topk into a single
        # `indexer_score_reduce` kernel (README candidate #4).
        score = ttnn.relu(score)
        score_t = ttnn.transpose(score, -2, -1)                   # [B, 1, T_pad, H]
        w_b = ttnn.reshape(w_tt, [B, 1, 1, H])
        score_t = ttnn.multiply(score_t, w_b)
        return ttnn.sum(score_t, dim=-1, keepdim=False)           # [B, 1, T_pad]


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

        # Persistent device-side joint MLA kv_cache. Shape on device:
        # [1, 1, kv_cache_size_pad, head_dim] (B=1, num_heads=1) so
        # ttnn.kv_cache.update_cache_for_token_ can write a single slot. The
        # seq dim is padded up to a tile boundary; padding slots are never
        # addressed by topk_idxs and stay zero.
        kv_shape = tuple(attn.kv_cache.shape)
        if len(kv_shape) != 3:
            raise ValueError(
                f"expected attn.kv_cache to be 3D [B, kv_cache_size, D], got {kv_shape}"
            )
        if kv_shape[2] != self.head_dim:
            raise ValueError(
                f"attn.kv_cache last dim {kv_shape[2]} != head_dim {self.head_dim}"
            )
        if self.head_dim % 32 != 0:
            raise ValueError(
                f"head_dim {self.head_dim} must be a multiple of 32 for tile layout"
            )
        self.kv_cache_size = kv_shape[1]
        self.kv_cache_size_pad = -(-self.kv_cache_size // 32) * 32
        self.kv_cache_tt = None
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self, B: int = 1):
        """Allocate the joint MLA kv_cache buffer eagerly so first decode
        does not trigger a host->device transfer. Inherits whatever CPU
        state is in attn.kv_cache at construction time (zeros pre-prefill,
        prefill state otherwise). Also hoists the per-step x upload
        buffer; per-call intermediates (q/kv/rotary) still allocate."""
        ttnn = self._ttnn
        attn = self.attn
        D = self.head_dim
        cache_3d = attn.kv_cache[:B].to(torch.bfloat16).contiguous()
        if self.kv_cache_size_pad != self.kv_cache_size:
            pad = torch.zeros(
                B, self.kv_cache_size_pad - self.kv_cache_size, D,
                dtype=torch.bfloat16,
            )
            cache_3d = torch.cat([cache_3d, pad], dim=1)
        cache_4d = cache_3d.view(B, 1, self.kv_cache_size_pad, D)
        self.kv_cache_tt = ttnn.as_tensor(
            cache_4d,
            device=self.mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

        self._upload_mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        self._x_upload_tt = ttnn.from_torch(
            torch.zeros(B, 1, self.dim, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        # Window topk indices [B, S, win] int32. Always present on the
        # device-indexer path; the per-step values change but shape is
        # pinned by the model's window_size.
        self._win_idxs_upload_tt = ttnn.from_torch(
            torch.zeros(B, 1, self.window_size, dtype=torch.int32),
            device=self.mesh, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Single-token forward. Prefill is driven externally as a per-token
        loop, so seqlen is always 1 and start_pos==0 is handled by _decode."""
        bsz, seqlen, _ = x.shape
        if seqlen != 1:
            raise RuntimeError(
                f"DeviceAttention expects seqlen=1; got seqlen={seqlen}. "
                "Prefill must loop one token at a time."
            )
        return self._decode(x, start_pos)

    def _decode(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        ttnn = self._ttnn
        attn = self.attn
        B, S, _ = x.shape
        H, D = self.n_heads, self.head_dim
        rd = self.rope_head_dim
        win = self.window_size

        # Single upload of x via the hoisted upload buffer.
        host_mesh = ttnn.from_torch(
            x.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)
        x_tt = self._x_upload_tt

        device_comp = getattr(attn, "_device_compressor", None)
        device_indexer = getattr(attn, "_device_indexer", None)

        # Q path: wq_a -> q_norm -> wq_b -> per-head rsqrt-norm -> rotary.
        q_lora_tt = attn.wq_a.forward_device(x_tt)            # [B, S, q_lora_rank]
        # q_norm: needs a [Mpad, hidden] device tensor. Reshape and pad-on-device.
        q_lora_2d = ttnn.reshape(q_lora_tt, [B * S, self.q_lora_rank])
        qr_2d = self._rmsnorm_device(self.q_norm_dev, q_lora_2d, B * S)
        qr_tt = ttnn.reshape(qr_2d, [B, S, self.q_lora_rank])

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

        # Device-side act_quant on the nope region (bf16 round-trip; no fp8
        # cast under the bf16-everywhere precision policy). After this, kv_tt
        # mirrors what CPU `act_quant(..., inplace=True)` would produce in bf16.
        kv_nope_q = _device_act_quant_block(
            ttnn,
            ttnn.slice(kv_tt, [0, 0, 0], [B, S, D - rd]),
            block_size=64,
        )
        kv_rope_only = ttnn.slice(kv_tt, [0, 0, D - rd], [B, S, D])
        kv_tt = ttnn.concat([kv_nope_q, kv_rope_only], dim=-1)

        # CPU side still needs kv on host when the compressor is CPU-resident.
        # Device path keeps kv_tt on device for both window-slot and compress-
        # slot writes (no round-trip).
        if device_comp is None:
            kv_cpu = self._download_replicated(kv_tt, B, S, D).to(x.dtype)

        topk_idxs = get_window_topk_idxs(win, B, S, start_pos)
        topk_idxs_dev = None  # int32 TILE [B, S, K] if device path, else None
        topk_idxs_dev_K = None
        if self.compress_ratio:
            offset = win
            if device_indexer is not None:
                # Device indexer: runs inner compressor + score reduce on
                # device, then ttnn.topk on device. When T_active > 0 we keep
                # the indices fully on device through typecast+add+concat to
                # avoid the small but synchronizing CPU round-trip.
                score_tt = device_indexer.forward_device_score(
                    x_tt, qr_tt, B, start_pos
                )
                end_pos = start_pos + 1
                T_active = end_pos // self.compress_ratio
                if T_active == 0:
                    compress_topk_idxs = get_compress_topk_idxs(
                        self.compress_ratio, B, S, start_pos, offset)
                    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
                else:
                    k = min(device_indexer.index_topk, T_active)
                    score_valid_tt = ttnn.slice(
                        score_tt, [0, 0, 0], [B, S, T_active])
                    _, cmp_idxs_tt = ttnn.topk(
                        score_valid_tt, k=k, dim=-1,
                        largest=True, sorted=True)
                    cmp_idxs_int = ttnn.typecast(cmp_idxs_tt, dtype=ttnn.int32)
                    cmp_idxs_int = ttnn.add(cmp_idxs_int, offset)
                    win_idxs_torch = topk_idxs.to(torch.int32).contiguous()
                    win_host_mesh = ttnn.from_torch(
                        win_idxs_torch, dtype=ttnn.int32,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=self._upload_mapper,
                    )
                    ttnn.copy_host_to_device_tensor(
                        win_host_mesh, self._win_idxs_upload_tt
                    )
                    topk_idxs_dev = ttnn.concat(
                        [self._win_idxs_upload_tt, cmp_idxs_int], dim=-1
                    )
                    topk_idxs_dev_K = win + k
                    topk_idxs = None
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio, B, S, start_pos, offset)
                topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        if topk_idxs is not None:
            topk_idxs = topk_idxs.int()

        # KV cache update.
        common_rep = dict(
            device=self.mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        if device_comp is None:
            attn.kv_cache[:B, start_pos % win] = kv_cpu.squeeze(1)
            if self.compress_ratio:
                attn.compressor(x, start_pos)
            window_slot = (
                attn.kv_cache[:B, start_pos % win]
                .to(torch.bfloat16)
                .contiguous()
                .view(B, 1, 1, D)
            )
            window_slot_tt = ttnn.as_tensor(window_slot, **common_rep)
            ttnn.kv_cache.update_cache_for_token_(
                self.kv_cache_tt, window_slot_tt, start_pos % win, 0
            )
            if self.compress_ratio and (start_pos + 1) % self.compress_ratio == 0:
                comp_idx = win + start_pos // self.compress_ratio
                comp_slot = (
                    attn.kv_cache[:B, comp_idx]
                    .to(torch.bfloat16)
                    .contiguous()
                    .view(B, 1, 1, D)
                )
                comp_slot_tt = ttnn.as_tensor(comp_slot, **common_rep)
                ttnn.kv_cache.update_cache_for_token_(
                    self.kv_cache_tt, comp_slot_tt, comp_idx, 0
                )
        else:
            # Device path: write window slot directly from kv_tt; compressor
            # writes compress slot at win + comp_idx into the same buffer.
            kv_4d = ttnn.reshape(kv_tt, [B, 1, 1, D])
            ttnn.kv_cache.update_cache_for_token_(
                self.kv_cache_tt, kv_4d, start_pos % win, 0
            )
            device_comp.forward_device(x_tt, B, start_pos)

        kv_full_tt = ttnn.reshape(self.kv_cache_tt, [self.kv_cache_size_pad, D])
        if topk_idxs_dev is not None:
            K = topk_idxs_dev_K
            idxs_tt, valid_tt = self.sparse_attn_dev._idxs_int_tile_to_idxs_and_mask(
                topk_idxs_dev, B, S, K
            )
        else:
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


def _open_mesh(shape=(1, 4), trace_region_size: int = 100_000_000):
    """Open a mesh device. Enables 1D fabric for CCL ops (all_gather etc.)
    and reserves trace_region_size bytes (default 100MB) for captured
    traces; required before any begin_trace_capture call."""
    import ttnn
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    return ttnn.open_mesh_device(
        ttnn.MeshShape(*shape), trace_region_size=trace_region_size
    )


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
            expert_dtype=cfg.get("expert_dtype", "fp4"),
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

    def offload_lm_head(self, mesh):
        """Move the lm_head matmul and the final RMSNorm to a 1xN mesh.

        Must be called after load_weights(). Shards the lm_head column-parallel
        across vocab; replicates the tiny final-norm weight on each chip.
        ParallelHead.forward is rebound to skip the CPU norm and pass the
        unnormalized last-token hidden straight to the device (norm + matmul
        fused there).
        """
        head = self.transformer.head
        final_norm = self.transformer.norm
        self._device_lm_head = DeviceLMHead(mesh, head.weight.data,
                                            norm_weight=final_norm.weight.data,
                                            norm_eps=final_norm.eps)
        dlh = self._device_lm_head

        def _device_head_forward(self_head, x, hc_fn, hc_scale, hc_base, norm):
            with _phase("head.hc_combiner"):
                x = self_head.hc_head(x, hc_fn, hc_scale, hc_base)  # still CPU
            with _phase("head.norm_and_logits"):
                return dlh(x[:, -1].contiguous())
        head.forward = _device_head_forward.__get__(head, type(head))
        final_norm.weight.data = torch.empty(0)  # device owns it
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

    def offload_sparse_attn(self, mesh):
        """Build DeviceSparseAttn instances; consumed by offload_attn_full
        to run gather+score-matmul+masked-softmax+weighted-sum on device."""
        self._device_sparse_attn = []
        for layer in self.transformer.layers:
            attn = layer.attn
            dsa = DeviceSparseAttn(mesh, attn.attn_sink, attn.softmax_scale)
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

    def offload_compressor_indexer(self, mesh):
        """Replace every Attention.compressor and Attention.indexer with
        DeviceCompressor + DeviceIndexer instances.

        Requires --offload-attn-full (DeviceAttention._decode is what dispatches
        to the device versions) and --offload-indexer-linears (we share the
        already-on-device wq_b / weights_proj). Per layer:
          - attn._device_compressor: DeviceCompressor for attn.compressor.
            Bound to attn._device_attention.kv_cache_tt at first decode.
          - attn._device_indexer (when ratio=4): DeviceIndexer wrapping
            attn.indexer with its own DeviceCompressor (rotate=True).
        """
        if not hasattr(self, "_device_attn_full") or not self._device_attn_full:
            raise RuntimeError(
                "offload_compressor_indexer requires --offload-attn-full"
            )
        self._device_compressors = []
        self._device_indexers = []
        for layer, da in zip(self.transformer.layers, self._device_attn_full):
            attn = layer.attn
            if not attn.compress_ratio:
                continue
            # Eagerly wire compressor/indexer freqs_cis and kv_cache from the
            # owning module. CPU Attention.forward / Indexer.forward set these
            # lazily on first call, but with prefill on device the lazy paths
            # never run.
            if attn.compressor.freqs_cis is None:
                attn.compressor.freqs_cis = attn.freqs_cis
            if attn.compressor.kv_cache is None:
                attn.compressor.kv_cache = attn.kv_cache[:, attn.window_size:]
            if attn.indexer is not None:
                if attn.indexer.freqs_cis is None:
                    attn.indexer.freqs_cis = attn.freqs_cis
                if attn.indexer.compressor.freqs_cis is None:
                    attn.indexer.compressor.freqs_cis = attn.indexer.freqs_cis
                if attn.indexer.compressor.kv_cache is None:
                    attn.indexer.compressor.kv_cache = attn.indexer.kv_cache
            wkv = attn.compressor.wkv if isinstance(
                attn.compressor.wkv, DeviceColLinear
            ) else DeviceColLinear(mesh, attn.compressor.wkv.weight)
            wgate = attn.compressor.wgate if isinstance(
                attn.compressor.wgate, DeviceColLinear
            ) else DeviceColLinear(mesh, attn.compressor.wgate.weight)
            norm_dev = DeviceRMSNorm(
                mesh, attn.compressor.norm.weight, attn.compressor.norm.eps
            )
            dc = DeviceCompressor(
                mesh=mesh,
                comp=attn.compressor,
                wkv_dev=wkv,
                wgate_dev=wgate,
                norm_dev=norm_dev,
            )
            dc.bind_kv_cache_tt(da.kv_cache_tt, slot_offset=da.window_size)
            attn._device_compressor = dc
            self._device_compressors.append(dc)

            indexer = getattr(attn, "indexer", None)
            if indexer is None:
                continue
            if not isinstance(indexer.wq_b, DeviceColLinear):
                raise RuntimeError(
                    "offload_compressor_indexer requires --offload-indexer-linears"
                )
            ix_wkv = indexer.compressor.wkv if isinstance(
                indexer.compressor.wkv, DeviceColLinear
            ) else DeviceColLinear(mesh, indexer.compressor.wkv.weight)
            ix_wgate = indexer.compressor.wgate if isinstance(
                indexer.compressor.wgate, DeviceColLinear
            ) else DeviceColLinear(mesh, indexer.compressor.wgate.weight)
            ix_norm = DeviceRMSNorm(
                mesh, indexer.compressor.norm.weight, indexer.compressor.norm.eps
            )
            ix_dc = DeviceCompressor(
                mesh=mesh,
                comp=indexer.compressor,
                wkv_dev=ix_wkv,
                wgate_dev=ix_wgate,
                norm_dev=ix_norm,
            )
            di = DeviceIndexer(
                mesh=mesh,
                indexer=indexer,
                dc=ix_dc,
                wq_b_dev=indexer.wq_b,
                weights_proj_dev=indexer.weights_proj,
            )
            attn._device_indexer = di
            self._device_indexers.append(di)
            self._device_compressors.append(ix_dc)

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
        """Fused end-to-end attention forward on device.

        Replaces Attention.forward with DeviceAttention.forward. Prefill is
        driven by Model.prefill as a per-token loop through this same path,
        so DeviceAttention only needs to handle seqlen=1.
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
            # active[0] holds the DeviceMHC instance whose stash the next
            # hc_post call must consume. hc_pre sets it; hc_post reads + clears.
            active = [None]

            def _device_hc_pre(self_layer, x, hc_fn, hc_scale, hc_base,
                               _attn=mhc_attn, _ffn=mhc_ffn,
                               _attn_id=attn_fn_id, _ffn_id=ffn_fn_id,
                               _active=active):
                fid = id(hc_fn)
                if fid == _attn_id:
                    _active[0] = _attn
                    return _attn.hc_pre(x)
                if fid == _ffn_id:
                    _active[0] = _ffn
                    return _ffn.hc_pre(x)
                raise RuntimeError(
                    "DeviceMHC dispatch: hc_fn identity matches neither "
                    "hc_attn_fn nor hc_ffn_fn"
                )

            def _device_hc_post(self_layer, x, _active=active):
                mhc = _active[0]
                if mhc is None:
                    raise RuntimeError(
                        "DeviceMHC dispatch: hc_post called before hc_pre")
                _active[0] = None
                return mhc.hc_post(x)

            layer.hc_pre = _device_hc_pre.__get__(layer, type(layer))
            layer.hc_post = _device_hc_post.__get__(layer, type(layer))
            self._device_mhc.append((mhc_attn, mhc_ffn))

    def _iter_device_modules(self):
        """Yield every Device* instance the model has constructed. Single
        source of truth for the orchestrator and any future tracing wiring."""
        if hasattr(self, "_device_lm_head"):
            yield self._device_lm_head
        for attr in ("_device_wq_a", "_device_wq_b", "_device_wkv",
                     "_device_wo_b", "_device_compressor_linears",
                     "_device_indexer_linears"):
            for inst in getattr(self, attr, ()):
                yield inst
        for attr in ("_device_rms_norms", "_device_shared_experts",
                     "_device_moe_gates", "_device_sparse_attn",
                     "_device_attn_full", "_device_compressors",
                     "_device_indexers"):
            for inst in getattr(self, attr, ()):
                yield inst
        for pair in getattr(self, "_device_mhc", ()):
            for inst in pair:
                yield inst

    def allocate_decode_tensors(self):
        """Single entry point that re-runs every Device* instance's
        `_alloc_decode_tensors()`. Each per-step ttnn tensor lives on its
        owning class and is referenced by every ttnn op inside that class
        via output_tensor= / optional_output_tensor=. Currently invoked at
        construction; this orchestrator exists so tracing (which forbids
        per-call allocations) can re-run it once before begin_trace_capture."""
        n = 0
        for mod in self._iter_device_modules():
            alloc = getattr(mod, "_alloc_decode_tensors", None)
            if alloc is not None:
                alloc()
                n += 1
        return n

    @torch.inference_mode()
    def step_decode(self, token_id: int, pos: int) -> torch.Tensor:
        """Single decode step. Returns logits [vocab_size]."""
        ids = torch.tensor([[token_id]], dtype=torch.long)
        logits = self.transformer.forward(ids, start_pos=pos)
        return logits[0]

    @torch.inference_mode()
    def prefill(self, tokens: list[int]) -> torch.Tensor:
        """Loop the device decode path over every prompt token. Slow per
        token, but reuses the decode kernels and keeps prefill on device."""
        logits = None
        for i, tok in enumerate(tokens):
            ids = torch.tensor([[tok]], dtype=torch.long)
            logits = self.transformer.forward(ids, start_pos=i)
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

    print("[phase] opening 1x4 mesh ...")
    t0 = time.time()
    mesh = _open_mesh(shape=(1, 4))
    print(f"[phase] mesh opened in {time.time() - t0:.1f}s")

    n_layers = len(model.transformer.layers)
    print("[phase] sharding lm_head on mesh ...")
    t0 = time.time()
    model.offload_lm_head(mesh)
    print(f"[phase] lm_head offloaded in {time.time() - t0:.1f}s")
    for label, method in (
        ("wq_a", model.offload_attn_wq_a),
        ("wq_b", model.offload_attn_wq_b),
        ("wkv", model.offload_attn_wkv),
        ("wo_b", model.offload_attn_wo_b),
        ("moe_gate", model.offload_moe_gate),
        ("moe_shared_expert", model.offload_moe_shared_expert),
        ("rms_norms", model.offload_rms_norms),
        ("embedding", model.offload_embedding),
        ("compressor_linears", model.offload_compressor_linears),
        ("indexer_linears", model.offload_indexer_linears),
        ("sparse_attn", model.offload_sparse_attn),
        ("attn_full", model.offload_attn_full),
        ("mhc", model.offload_mhc),
        ("compressor_indexer", model.offload_compressor_indexer),
    ):
        print(f"[phase] offloading {label} on mesh ...")
        t0 = time.time()
        method(mesh)
        print(f"[phase] {label} offloaded in {time.time() - t0:.1f}s")

    n_alloc_modules = sum(1 for _ in model._iter_device_modules())
    print(f"[phase] {n_alloc_modules} device modules registered for "
          f"Model.allocate_decode_tensors")

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

    _close_mesh(mesh)


if __name__ == "__main__":
    main()
