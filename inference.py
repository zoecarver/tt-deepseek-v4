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
from pathlib import Path
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
# When set, called at the start and end of every _phase() to flush the device
# command queue. Without this, ttnn ops are non-blocking and a phase's wall
# time captures dispatch overhead instead of compute, with the actual work
# bleeding into the next phase. Costs ~one synchronize_device per region but
# makes the per-phase shares trustworthy. Set by main() once the mesh opens.
_PHASE_SYNC = None


@contextmanager
def _phase(name: str):
    if _PHASE_SYNC is not None:
        _PHASE_SYNC()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if _PHASE_SYNC is not None:
            _PHASE_SYNC()
        dt = time.perf_counter() - t0
        _PHASE_ACCUM[name] = _PHASE_ACCUM.get(name, 0.0) + dt
        _PHASE_COUNTS[name] = _PHASE_COUNTS.get(name, 0) + 1


def _set_phase_sync(fn):
    global _PHASE_SYNC
    _PHASE_SYNC = fn


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


def _build_window_topk_table(window_size: int) -> torch.Tensor:
    """Precomputed [2*window_size, window_size] int32 lookup table for
    per-step window topk indices:
      - rows [0, window_size): forward-pad at p < window_size
        (`[0, 1, ..., p, -1, ...]`).
      - rows [window_size, 2*window_size): wraparound at p >= window_size
        (cat of `arange(p%win + 1, win)` and `arange(0, p%win + 1)`).
    Indexed via `_window_topk_row_for_pos` and sliced on device per step
    (loop-prefill keeps seqlen=1, so we never need the seqlen>1 form)."""
    table = torch.full(
        (2 * window_size, window_size), -1, dtype=torch.int32
    )
    for s in range(window_size):
        table[s, : s + 1] = torch.arange(s + 1, dtype=torch.int32)
    for s in range(window_size):
        table[window_size + s] = torch.cat(
            [
                torch.arange(s + 1, window_size, dtype=torch.int32),
                torch.arange(0, s + 1, dtype=torch.int32),
            ]
        )
    return table


def _window_topk_row_for_pos(start_pos: int, window_size: int) -> int:
    if start_pos < window_size:
        return start_pos
    return window_size + (start_pos % window_size)


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

    def forward_device(self, x_tt, input_ids_tt):
        """Device-in / device-out MoE for the chained block path.

        x_tt: [M, dim] bf16 device tensor (replicated). Must be M=1 (decode).
        input_ids_tt: [1, 1] uint32 device tensor (pre-uploaded once per step).
        Returns y_tt: [M, dim] bf16 device tensor.

        Routed experts go through Path D (cached, on-device): per-chip 8-expert
        grouped MLP + selection-mask sum + ttnn.all_reduce. The host fallback
        was removed once Hash-MoE landed on-device (commit 7492c50); every
        layer must have offload_moe_routed_experts attached.
        """
        if getattr(self, "_w1_tt", None) is None:
            raise RuntimeError(
                "MoE.forward_device requires routed-expert cache; "
                "offload_moe_routed_experts must run before forward_device."
            )

        shared_dev = self.shared_experts
        ttnn = shared_dev._ttnn

        M = int(x_tt.shape[0])
        if M != 1:
            raise ValueError(
                f"MoE.forward_device expects M=1 (decode), got M={M}"
            )
        if int(x_tt.shape[-1]) != self.dim:
            raise ValueError(
                f"x_tt last dim {int(x_tt.shape[-1])} != self.dim {self.dim}"
            )

        shared_out_tt = shared_dev.forward_device(x_tt)

        gate_dev = self.gate._device_gate
        if self.gate.hash:
            weights_tt, indices_tt = gate_dev.forward_device(x_tt, input_ids_tt)
        else:
            weights_tt, indices_tt = gate_dev.forward_device(x_tt)
        return self._forward_device_routed_cached(
            x_tt, weights_tt, indices_tt, shared_out_tt)

    def _forward_device_routed_cached(
        self, x_tt, weights_tt, indices_tt, shared_out_tt
    ):
        """Path D body: per-chip 8-expert grouped MLP + selection mask.

        Inputs (all device tensors):
          x_tt: bf16 [1, dim] replicated
          weights_tt: bf16 [1, topk] replicated (gate output, normalized * route_scale)
          indices_tt: int32 [1, topk] replicated
          shared_out_tt: bf16 [1, dim] replicated (shared expert output)

        Cached weights (attached by Model.offload_moe_routed_experts;
        natively quantized as bfp4_b — ttnn.matmul dequantizes each tile
        on the fly using the format's intrinsic per-tile exponent):
          self._w1_tt:        bfp4_b [1, per_chip, dim, inter] sharded
          self._w3_tt:        bfp4_b [1, per_chip, dim, inter] sharded
          self._w2_tt:        bfp4_b [1, per_chip, inter, dim] sharded
          self._chip_local_ids_tt: int32 [1, 1, per_chip] sharded
          self._n_per_chip: experts per chip (== n_routed_experts // mesh_size)

        Returns: bf16 [1, dim] replicated.
        """
        ttnn = self.shared_experts._ttnn
        per_chip = self._n_per_chip
        topk = self.n_activated_experts
        dim = self.dim

        # Rank 4 throughout: ttnn.eq / ttnn.clamp only support ranks 2/3/4.

        # Selection mask:
        #   chip_ids   [1, per_chip, 1, 1]   sharded (pre-shaped at offload)
        #   indices    [1, 1, 1, topk]       replicated
        #   eq         [1, per_chip, 1, topk]
        #   * weights  [1, per_chip, 1, topk]
        #   sum -1     [1, per_chip, 1, 1]   = per-chip mask, already aligned
        weights_4d = ttnn.reshape(weights_tt, [1, 1, 1, topk])
        indices_int32 = (
            indices_tt if indices_tt.dtype == ttnn.int32
            else ttnn.typecast(indices_tt, ttnn.int32))
        indices_4d = ttnn.reshape(indices_int32, [1, 1, 1, topk])
        match = ttnn.eq(indices_4d, self._chip_ids_4d_tt)
        match_bf16 = ttnn.typecast(match, ttnn.bfloat16)
        ttnn.multiply(weights_4d, match_bf16, output_tensor=match_bf16)
        mask = ttnn.sum(match_bf16, dim=-1, keepdim=True)

        # Grouped MLP. Weights are [1, per_chip, K, N] sharded; ttnn.matmul
        # won't broadcast batch dims when one operand is all-1 and the
        # other isn't, so we repeat x along the per_chip dim explicitly.
        x_4d = ttnn.reshape(x_tt, [1, 1, 1, dim])
        x_grouped = ttnn.repeat(x_4d, [1, per_chip, 1, 1])  # [1, per_chip, 1, dim]
        y1 = ttnn.matmul(x_grouped, self._w1_tt,
                         optional_output_tensor=self._fp4_y1_scratch_tt)
        y3 = ttnn.matmul(x_grouped, self._w3_tt,
                         optional_output_tensor=self._fp4_y3_scratch_tt)
        if self.experts[0].swiglu_limit > 0:
            limit = float(self.experts[0].swiglu_limit)
            ttnn.clamp(y1, max=limit, output_tensor=y1)
            ttnn.clamp(y3, min=-limit, max=limit, output_tensor=y3)
        ttnn.silu(y1, output_tensor=y1)
        ttnn.multiply(y1, y3, output_tensor=y1)  # glu = silu(y1) * y3 (in-place)

        y = ttnn.matmul(y1, self._w2_tt,
                        optional_output_tensor=self._fp4_y2_scratch_tt)

        ttnn.multiply(y, mask, output_tensor=y)  # y_masked (in-place)

        # Sum across local experts: keepdim=True so result stays rank 4.
        y_local = ttnn.sum(y, dim=1, keepdim=True)  # [1, 1, 1, dim]

        # All-reduce across the (rows*cols) mesh: full 256-expert sum.
        y_full = ttnn.all_reduce(y_local)

        y_2d = ttnn.reshape(y_full, [1, dim])
        return ttnn.add(y_2d, shared_out_tt)


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


def _block_forward(layer, x, start_pos: int,
                   input_ids_tt,
                   B: int, S: int, mhc: int, hidden: int,
                   orig_dtype):
    """Vanilla block forward. Calls the bound (device or CPU) sub-modules
    directly so the chain is one flat sequence of explicit calls — no
    nn.Module.__call__ traversal between phases.

    x is the residual stream as a device tensor in hc_pre_device input
    format: [num_tokens_pad, mhc*hidden] fp32. Returns the same format
    (output of hc_post → packed for the next layer's hc_pre).

    input_ids_tt is the [1, 1] uint32 device tensor uploaded once per step
    by the caller; consumed by hash-gate MoE layers.

    Method dispatch follows the offload-time bindings:
      - layer._device_mhc_attn / layer._device_mhc_ffn: DeviceMHC instances
      - layer.attn_norm.forward / layer.ffn_norm.forward: DeviceRMSNorm
      - layer.attn.forward: DeviceAttention.forward (decode-only)
      - layer.ffn.forward_device: cached Path D for all 43 layers (hash + non-hash).
    """
    mhc_attn = layer._device_mhc_attn
    mhc_ffn = layer._device_mhc_ffn
    attn_norm_dn = layer.attn_norm.forward.__self__
    ffn_norm_dn = layer.ffn_norm.forward.__self__
    attn_dev = layer.attn.forward.__self__
    ttnn = attn_norm_dn._ttnn
    num_tokens = B * S
    num_tokens_pad = -(-num_tokens // _MHC_TILE) * _MHC_TILE

    with _phase("block.hc_pre"):
        hc_out_fp32 = mhc_attn.hc_pre_device(
            num_tokens, num_tokens_pad, a_tt=x)
    with _phase("block.norm"):
        # DeviceRMSNorm.forward_device reads its x_tt arg directly and writes
        # _out_tt; _x_upload_tt is only used by the host-upload wrapper. Reuse
        # it here as the typecast destination so we don't alloc a fresh
        # [Mpad, hidden] bf16 buffer per call.
        ttnn.typecast(hc_out_fp32, dtype=ttnn.bfloat16,
                      output_tensor=attn_norm_dn._x_upload_tt)
        norm_out_tt = attn_norm_dn.forward_device(
            attn_norm_dn._x_upload_tt, num_tokens)
    with _phase("block.attn"):
        ttnn.slice(norm_out_tt, [0, 0], [num_tokens, hidden],
                   output_tensor=mhc_attn._norm_slice_tt)
        bridge_tt = ttnn.reshape(mhc_attn._norm_slice_tt, [B, S, hidden])
        attn_out_tt = attn_dev.forward_device(bridge_tt, start_pos)
    with _phase("block.hc_post"):
        x_2d = ttnn.reshape(attn_out_tt, [num_tokens, 1, hidden])
        x_repeated = ttnn.repeat(x_2d, ttnn.Shape([1, mhc, 1]))
        x_padded = ttnn.pad(
            x_repeated,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, 0)],
            value=0.0,
        )
        x_post_input = ttnn.reshape(
            x_padded, [num_tokens * _MHC_TILE, hidden])
        ttnn.typecast(x_post_input, dtype=ttnn.float32,
                      output_tensor=mhc_attn._x_upload_tt)
        post_out_tt = mhc_attn.hc_post_device(num_tokens)

    with _phase("block.hc_pre"):
        a_input_tt = _mhc_post_to_a_tt(
            ttnn, post_out_tt, num_tokens, num_tokens_pad, mhc, hidden)
        ffn_hc_out_fp32 = mhc_ffn.hc_pre_device(
            num_tokens, num_tokens_pad, a_tt=a_input_tt)
    with _phase("block.norm"):
        ttnn.typecast(ffn_hc_out_fp32, dtype=ttnn.bfloat16,
                      output_tensor=ffn_norm_dn._x_upload_tt)
        ffn_norm_out_tt = ffn_norm_dn.forward_device(
            ffn_norm_dn._x_upload_tt, num_tokens)
    with _phase("block.ffn"):
        ttnn.slice(ffn_norm_out_tt, [0, 0], [num_tokens, hidden],
                   output_tensor=mhc_ffn._norm_slice_tt)
        moe_out_tt = layer.ffn.forward_device(
            mhc_ffn._norm_slice_tt, input_ids_tt)
    with _phase("block.hc_post"):
        x_2d = ttnn.reshape(moe_out_tt, [num_tokens, 1, hidden])
        x_repeated = ttnn.repeat(x_2d, ttnn.Shape([1, mhc, 1]))
        x_padded = ttnn.pad(
            x_repeated,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, 0)],
            value=0.0,
        )
        x_post_input = ttnn.reshape(
            x_padded, [num_tokens * _MHC_TILE, hidden])
        ttnn.typecast(x_post_input, dtype=ttnn.float32,
                      output_tensor=mhc_ffn._x_upload_tt)
        ffn_post_out_tt = mhc_ffn.hc_post_device(num_tokens)
        return _mhc_post_to_a_tt(
            ttnn, ffn_post_out_tt, num_tokens, num_tokens_pad, mhc, hidden)


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
        mhc_attn = self.layers[0]._device_mhc_attn
        ttnn = mhc_attn._ttnn
        mhc = self.hc_mult
        hidden = mhc_attn.hidden
        B, S = input_ids.shape
        num_tokens = B * S
        num_tokens_pad = -(-num_tokens // _MHC_TILE) * _MHC_TILE
        orig_dtype = torch.bfloat16

        # Upload token ids ONCE per step into the pre-allocated [B, S] uint32
        # device tensor. _device_embed and any hash-gate MoE layer consumes
        # this same tensor; no other host->device transfers exist in the
        # decode hot path.
        ids_host = ttnn.from_torch(
            input_ids.to(torch.int32).contiguous(),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._input_ids_upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(ids_host, self._input_ids_tt)
        input_ids_tt = self._input_ids_tt

        with _phase("embed"):
            embed_tt = self.embed(input_ids_tt)
            e_2d = ttnn.reshape(embed_tt, [num_tokens, hidden])
            e_fp32 = ttnn.typecast(e_2d, dtype=ttnn.float32)
            e_repeated = ttnn.repeat(e_fp32, ttnn.Shape([1, mhc]))
            a_tt = ttnn.pad(
                e_repeated,
                padding=[(0, num_tokens_pad - num_tokens), (0, 0)],
                value=0.0,
            )
        with _phase("blocks"):
            for layer in self.layers:
                a_tt = _block_forward(layer, a_tt, start_pos, input_ids_tt,
                                      B, S, mhc, hidden, orig_dtype)
        with _phase("head"):
            logits = self.head(a_tt, num_tokens)
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
                 norm_eps: float = 1e-6,
                 hc_fn: Optional[torch.Tensor] = None,
                 hc_scale: Optional[torch.Tensor] = None,
                 hc_base: Optional[torch.Tensor] = None,
                 hc_eps: float = 1e-6,
                 hc_mult: int = 1):
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        vocab, dim = cpu_weight.shape
        # Shard vocab across all chips (1D flat over the 2D mesh). Pad to
        # num_chips * TILE so each chip's slice is tile-aligned.
        TILE = 32
        num_chips = self.mesh_shape[0] * self.mesh_shape[1]
        chunk = num_chips * TILE
        vocab_padded = -(-vocab // chunk) * chunk
        self.vocab = vocab
        self.vocab_padded = vocab_padded
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        w_dv = _weight_to_bf16(cpu_weight).transpose(0, 1).contiguous()  # [dim, vocab]
        if vocab_padded != vocab:
            w_dv = F.pad(w_dv, (0, vocab_padded - vocab))
        self.w_tt = ttnn.as_tensor(
            w_dv,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
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

        self.hc_fn_t_tt = None
        self.hc_scale_tt = None
        self.hc_base_tt = None
        if hc_fn is not None:
            # hc_fn: [mhc, mhc*hidden] fp32. Transpose to [mhc*hidden, mhc] for matmul.
            hc_fn_t = hc_fn.to(torch.float32).transpose(0, 1).contiguous()
            self.hc_fn_t_tt = ttnn.as_tensor(
                hc_fn_t, device=mesh, layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            # hc_scale: [1] fp32 -> broadcastable [1, 1].
            scale_2d = hc_scale.to(torch.float32).reshape(1, 1).contiguous()
            self.hc_scale_tt = ttnn.as_tensor(
                scale_2d, device=mesh, layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            # hc_base: [mhc] fp32 -> [1, mhc] for elementwise add against [num_tokens, mhc].
            base_2d = hc_base.to(torch.float32).reshape(1, hc_mult).contiguous()
            self.hc_base_tt = ttnn.as_tensor(
                base_2d, device=mesh, layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            torch.zeros(1, 1, self.vocab_padded, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=-1),
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
        self._compute_body()
        y = ttnn.to_torch(
            self._matmul_out_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1),
        )
        return y[:1, 0, :self.vocab].float()  # [B, vocab]

    def forward_a_tt(self, a_tt, num_tokens: int) -> torch.Tensor:
        """Pure-device head: a_tt [num_tokens_pad, mhc*hidden] fp32 -> logits torch [B=1, vocab].
        Runs hc_combiner + final RMSNorm + lm_head matmul on device."""
        ttnn = self._ttnn
        if self.hc_fn_t_tt is None:
            raise ValueError("forward_a_tt requires hc_fn/scale/base; pass them to DeviceLMHead.")
        mhc = self.hc_mult
        hidden = self.dim
        D = mhc * hidden
        last = num_tokens - 1
        # Slice last token row: [1, mhc*hidden] fp32.
        x_2d = ttnn.slice(a_tt, [last, 0], [last + 1, D])
        # mixes = x_2d @ hc_fn_t_tt -> [1, mhc] fp32.
        mixes = ttnn.matmul(x_2d, self.hc_fn_t_tt)
        # rsqrt(mean(x^2) + eps): mean over last dim of [1, D].
        sq = ttnn.multiply(x_2d, x_2d)
        sq_mean = ttnn.mean(sq, dim=-1, keepdim=True)  # [1, 1]
        sq_mean_eps = ttnn.add(sq_mean, self.norm_eps)
        rsqrt_val = ttnn.rsqrt(sq_mean_eps)  # [1, 1]
        # pre = sigmoid(mixes * rsqrt * hc_scale + hc_base) + hc_eps -> [1, mhc].
        scaled = ttnn.multiply(mixes, rsqrt_val)
        scaled = ttnn.multiply(scaled, self.hc_scale_tt)
        scaled = ttnn.add(scaled, self.hc_base_tt)
        pre = ttnn.sigmoid(scaled)
        pre = ttnn.add(pre, self.hc_eps)
        # y = pre @ x_view, where x_view is x_2d viewed as [mhc, hidden].
        x_3d = ttnn.reshape(x_2d, [1, mhc, hidden])
        pre_3d = ttnn.reshape(pre, [1, 1, mhc])
        y_3d = ttnn.matmul(pre_3d, x_3d)  # [1, 1, hidden] fp32
        y_bf16 = ttnn.typecast(y_3d, dtype=ttnn.bfloat16)
        # RMSNorm + lm_head matmul.
        if self.norm_tt is not None:
            y_4d = ttnn.reshape(y_bf16, (1, 1, 1, hidden))
            y_normed = ttnn.rms_norm(y_4d, weight=self.norm_tt, epsilon=self.norm_eps)
            y_normed = ttnn.reshape(y_normed, (1, 1, hidden))
        else:
            y_normed = y_bf16
        ttnn.matmul(y_normed, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._matmul_out_tt)
        out = ttnn.to_torch(
            self._matmul_out_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1),
        )
        return out[:1, 0, :self.vocab].float()  # [1, vocab]


class DeviceColLinear(nn.Module):
    """Column-parallel linear across the mesh.

    Weight: [out, in] (nn.Linear order). Transposed to [in, out] for ttnn.
    Auto-selects sharding based on out_dim:
      - **Full-mesh (TP=N)**: when `out_dim % (num_chips * TILE) == 0`, shard
        the out axis 1D-flat across all 32 chips. Each chip holds
        [in, out/N]. After matmul, two sequential all_gathers (cluster_axis=1
        then cluster_axis=0) reconstruct the full out_dim replicated across
        the mesh.
      - **Cols-only (TP=cols)**: otherwise, fall back to the original 8-chip
        col-shard. Each chip holds [in, out/cols], with 4x replication on
        the row axis. After matmul, one all_gather on cluster_axis=1.
    Output is replicated [1, 1, out_dim] in either mode, same calling
    contract.
    """

    _TILE = 32

    def __init__(self, mesh, cpu_weight: torch.Tensor):
        super().__init__()
        import ttnn
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        n_rows, n_cols = self.mesh_shape
        num_chips = n_rows * n_cols
        out_dim, in_dim = cpu_weight.shape
        if out_dim % n_cols != 0:
            raise ValueError(
                f"col-linear out_dim {out_dim} not divisible by mesh cols {n_cols}"
            )
        # Pick the finest shard that keeps each chip's slice tile-aligned.
        full_mesh = (out_dim % (num_chips * self._TILE) == 0)
        self._full_mesh = full_mesh
        self._n_shards = num_chips if full_mesh else n_cols
        self.in_dim = in_dim
        self.out_dim = out_dim
        w_io = _weight_to_bf16(cpu_weight).transpose(0, 1).contiguous()  # [in, out]
        self.w_tt = ttnn.as_tensor(
            w_io,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._w_mapper(),
        )
        self._alloc_decode_tensors()

    def _w_mapper(self):
        ttnn = self._ttnn
        if self._full_mesh:
            return ttnn.ShardTensorToMesh(self.mesh, dim=-1)
        return ttnn.ShardTensor2dMesh(self.mesh, self.mesh_shape, dims=(None, -1))

    def _alloc_decode_tensors(self):
        """Pre-allocate per-step output buffers for [B=1, S=1, out_dim]
        decode. The intermediate `_row_gathered_tt` is only used in the
        full-mesh path (after all_gather cluster_axis=1, before
        cluster_axis=0); skip it in the cols-only path."""
        ttnn = self._ttnn
        zeros = torch.zeros(1, 1, self.out_dim, dtype=torch.bfloat16)
        self._matmul_out_tt = ttnn.from_torch(
            zeros,
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._w_mapper(),
        )
        self._gather_out_tt = ttnn.from_torch(
            zeros,
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        if self._full_mesh:
            # After cluster_axis=1 gather, each row holds n_cols × per-chip
            # contiguous out values, replicated across cols within the row.
            # Shard tensor dim -1 across rows (axis 0 of mesh), replicate on
            # cols. The next all_gather (cluster_axis=0) collapses this into
            # _gather_out_tt.
            self._row_gathered_tt = ttnn.from_torch(
                zeros,
                device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh, self.mesh_shape, dims=(-1, None)),
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
        if self._full_mesh:
            composer = ttnn.ConcatMeshToTensor(self.mesh, dim=-1)
        else:
            composer = ttnn.ConcatMesh2dToTensor(self.mesh, self.mesh_shape, dims=(0, -1))
        y = ttnn.to_torch(y_tt, mesh_composer=composer)
        if y.shape[0] != B:
            y = y[:B]
        if y.shape[1] != S:
            y = y[:, :S]
        return y.to(torch.bfloat16)

    def forward_device(self, x_tt) -> "object":
        """Device-only path. x_tt: replicated input [1, 1, in_dim]. Returns
        replicated output [1, 1, out_dim]. In TP=N mode the gather is two
        sequential all_gathers (cluster_axis=1 then cluster_axis=0); in
        TP=cols mode it's a single cluster_axis=1 gather. Output buffers
        are pre-allocated; all ops write in place."""
        ttnn = self._ttnn
        ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._matmul_out_tt)
        if self._full_mesh:
            ttnn.all_gather(self._matmul_out_tt, dim=-1, cluster_axis=1,
                            num_links=1, output_tensor=self._row_gathered_tt)
            ttnn.all_gather(self._row_gathered_tt, dim=-1, cluster_axis=0,
                            num_links=1, output_tensor=self._gather_out_tt)
        else:
            ttnn.all_gather(self._matmul_out_tt, dim=-1, cluster_axis=1,
                            num_links=1, output_tensor=self._gather_out_tt)
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

    def forward_device(self, x_tt):
        """Pure-device gate body: x_tt [M=1, dim] bf16 device tensor in.
        Returns (weights_tt, indices_tt) device tensors. No host transfers."""
        ttnn = self._ttnn
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

    def forward(self, x: torch.Tensor):
        """x: [M=1, dim] -> (weights [1, topk] float32, indices [1, topk] int64).

        Wrapper: uploads x, runs forward_device, downloads (weights, indices).
        Use forward_device when chaining on device.
        """
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
        weights_tt, indices_tt = self.forward_device(self._x_upload_tt)

        weights = _readback_replicated_2d(
            ttnn, weights_tt, self.mesh, self.mesh_shape)[:M].float()
        indices = _readback_replicated_2d(
            ttnn, indices_tt, self.mesh, self.mesh_shape)[:M].long()
        return weights, indices


class HashDeviceMoEGate(nn.Module):
    """Hash-routed MoE gate (vocab-lookup indices) on a 1xN mesh.

    Hash-gate layers (layer_id < n_hash_layers) score with the same
    sqrt(softplus(x @ W^T)) as the regular gate, but pick indices from a
    fixed [vocab, topk] table indexed by input_ids — no topk(scores), no
    bias. This class runs the whole pipeline on device so MoE.forward_device
    can take Path D (cached bfp4_b experts) for hash layers too.

    Parallel to DeviceMoEGate; we keep them as separate classes since the
    bias/topk plumbing diverges enough to make a unified version messy.
    """

    def __init__(self, mesh, cpu_weight: torch.Tensor,
                 cpu_tid2eid: torch.Tensor,
                 topk: int, route_scale: float, score_func: str,
                 n_routed_experts: int):
        super().__init__()
        import ttnn
        if score_func != "sqrtsoftplus":
            raise ValueError(
                f"HashDeviceMoEGate v1 only supports score_func="
                f"'sqrtsoftplus', got {score_func!r}")
        if cpu_tid2eid.dtype != torch.int32:
            raise ValueError(
                f"tid2eid expected int32, got {cpu_tid2eid.dtype}")
        if int(cpu_tid2eid.max().item()) >= n_routed_experts:
            raise ValueError("tid2eid has out-of-range expert id")
        self._ttnn = ttnn
        self.mesh = mesh
        self.mesh_shape = tuple(mesh.shape)
        n_experts, dim = cpu_weight.shape
        self.n_experts = n_experts
        self.dim = dim
        self.topk = topk
        self.route_scale = float(route_scale)

        common_rep = dict(
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        w_kn = _weight_to_bf16(cpu_weight).transpose(0, 1).contiguous()
        self.w_tt = ttnn.as_tensor(w_kn, **common_rep)
        # tid2eid stored as bf16: expert ids 0..255 round-trip exactly through
        # bf16 (bf16 represents every integer up to 256), so the embedding
        # lookup is lossless. Cast back to int32 after the lookup so the
        # downstream gather/eq sees the right dtype.
        tid2eid_bf16 = cpu_tid2eid.to(torch.bfloat16).contiguous()
        self.tid2eid_tt = ttnn.as_tensor(
            tid2eid_bf16,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
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
            torch.zeros(1, self.n_experts, dtype=torch.bfloat16),
            **common_rep)

    def forward_device(self, x_tt, input_ids_tt):
        """Pure-device hash-gate body.

        x_tt: [1, dim] bf16 device tensor.
        input_ids_tt: [1, 1] uint32 device tensor (pre-uploaded once per step).
        Returns (weights_tt [1, topk] bf16, indices_tt [1, topk] uint32)
        replicated. No host transfers."""
        ttnn = self._ttnn
        ttnn.matmul(x_tt, self.w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._raw_tt)
        scores_tt = ttnn.sqrt(ttnn.softplus(self._raw_tt))
        idx_lookup = ttnn.embedding(
            input_ids_tt, self.tid2eid_tt, layout=ttnn.TILE_LAYOUT)
        # ttnn.gather requires uint32/uint16 index. Path D's selection mask
        # typecasts to int32 anyway, so returning uint32 is fine downstream.
        idx_u32 = ttnn.typecast(idx_lookup, ttnn.uint32)
        idx_2d = ttnn.reshape(idx_u32, [1, self.topk])
        gathered_tt = ttnn.gather(scores_tt, dim=-1, index=idx_2d)
        wsum_tt = ttnn.sum(gathered_tt, dim=-1, keepdim=True)
        normed_tt = ttnn.div(gathered_tt, wsum_tt)
        weights_tt = ttnn.multiply(normed_tt, self.route_scale)
        return weights_tt, idx_2d


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
        """Kernel output buffer + traced-input buffer for the loop-prefill
        / decode case (num_rows=1, Mpad=TILE). The fused kernel overwrites
        every tile of _out_tt; _x_upload_tt is filled via
        copy_host_to_device_tensor on every forward() call."""
        ttnn = self._ttnn
        self._out_tt = ttnn.zeros(
            shape=(_RMS_TILE, self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._x_upload_tt = ttnn.from_torch(
            torch.zeros(_RMS_TILE, self.hidden, dtype=torch.bfloat16),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self._upload_mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        self._trace_id = None

    def _compute_body(self):
        """Replay-shape: the fused kernel writes _out_tt from _x_upload_tt."""
        self._kernel(1)(self._x_upload_tt, self.gamma_tt, self.sc_tt, self._out_tt)

    def _capture_trace(self):
        ttnn = self._ttnn
        self._compute_body()  # warmup so any lazy host uploads happen first
        ttnn.synchronize_device(self.mesh)
        self._trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._compute_body()
        ttnn.end_trace_capture(self.mesh, self._trace_id, cq_id=0)

    def _kernel(self, num_row_tiles: int):
        # At decode (num_row_tiles=1) prefer the pre-built named global for
        # this hidden size. For prefill (num_row_tiles > 1) the shape isn't
        # pre-built; fall through to the cache helper, which still shares
        # the compile across all 86 layers in the same prefill pass.
        if num_row_tiles == 1:
            kernel = self._decode_kernel
            if kernel is not None:
                return kernel
        return _get_ttl_rmsnorm_kernel(
            num_row_tiles, self.hidden // _RMS_TILE, self.eps, 1.0 / self.hidden,
        )

    @property
    def _decode_kernel(self):
        """Named decode-path global matching this instance's hidden size,
        or None if no pre-built variant matches (e.g. an unusual hidden
        encountered before prebuild_ttl_decode_kernels ran)."""
        h = self.hidden
        if h == 4096:
            return ttl_rmsnorm_M32_h4096
        if h == 1024:
            return ttl_rmsnorm_M32_h1024
        if h == 512:
            return ttl_rmsnorm_M32_h512
        if h == 128:
            return ttl_rmsnorm_M32_h128
        return None

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

        if Mt == 1:
            host_mesh = ttnn.from_torch(
                x2.contiguous(),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=self._upload_mapper,
            )
            ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)
            self._compute_body()
            out_tt = self._out_tt
        else:
            rep = dict(
                device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            x_tt = ttnn.as_tensor(x2.contiguous(), **rep)
            out_tt = self.forward_device(x_tt, M)

        y = _readback_replicated_2d(
            ttnn, out_tt, self.mesh, self.mesh_shape)[:M]
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
# K-split fanout for the decode-only norm_fn ksplit kernel. 8 is the optimal
# benchmarked point; higher Kp plateaus on speedup and PCC degrades.
_MHC_NORM_FN_KP = 8


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


def _mhc_unpack_a_tt(packed: torch.Tensor, num_tokens: int, mhc: int, h: int) -> torch.Tensor:
    """[num_tokens_pad, mhc*h] (a_tt format) -> [num_tokens, mhc, h]."""
    return packed.view(-1, mhc, h)[:num_tokens, :, :].contiguous()


def _mhc_post_to_a_tt(ttnn, post_out_tt, num_tokens: int, num_tokens_pad: int,
                      mhc: int, hidden: int):
    """On-device transform: hc_post_device output [num_tokens*TILE, hidden]
    -> hc_pre_device input [num_tokens_pad, mhc*hidden] fp32. Slices off the
    TILE-padding rows, flattens (mhc, hidden) into a row, then zero-pads to
    num_tokens_pad rows. No host transfers."""
    post_3d = ttnn.reshape(post_out_tt, [num_tokens, _MHC_TILE, hidden])
    post_sliced = ttnn.slice(post_3d, [0, 0, 0], [num_tokens, mhc, hidden])
    post_flat = ttnn.reshape(post_sliced, [num_tokens, mhc * hidden])
    return ttnn.pad(
        post_flat,
        padding=[(0, num_tokens_pad - num_tokens), (0, 0)],
        value=0.0,
    )


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


def _compile_mhc_norm_fn_ksplit_kernel(K_tiles: int, Kp: int, rms_eps: float,
                                       inv_D: float, *, grid_cols: int = 8):
    """Inlined from tt-lang-kernels/pre_norm_fn_ksplit.py.

    K-axis-parallel version of pre_norm_fn for num_out_tiles=1 (decode).
    Splits the K-loop across Kp cores; root gathers matmul + ssq partials,
    finalizes inv_rms and writes the row. Optimal Kp=8 at V4-Flash decode
    (D=16384, K_tiles=512): 0.25ms vs 0.69ms single-core (~2.8x speedup,
    PCC 0.989).
    """
    if Kp < 2:
        raise ValueError(f"ksplit kernel requires Kp >= 2, got {Kp}")
    if K_tiles % Kp:
        raise ValueError(f"K_tiles={K_tiles} not divisible by Kp={Kp}")
    if Kp < grid_cols:
        grid_cols = Kp
    if Kp % grid_cols:
        raise ValueError(f"Kp={Kp} not divisible by grid_cols={grid_cols}")
    if Kp - 1 > 32:
        raise ValueError(f"Kp={Kp} requires block_count={Kp-1} > 32")

    K_BPN = K_tiles // Kp
    COL = grid_cols
    ROW = Kp // grid_cols

    @ttl.operation(
        grid=(COL, ROW),
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def norm_fn_ksplit_kernel(a, b, scaler, out):
        c_reduce_pipes = [
            ttl.Pipe(src=(col, row), dst=(0, 0))
            for row in range(ROW)
            for col in range(COL)
            if not (col == 0 and row == 0)
        ]
        c_reduce_net = ttl.PipeNet(c_reduce_pipes)

        sq_reduce_pipes = [
            ttl.Pipe(src=(col, row), dst=(0, 0))
            for row in range(ROW)
            for col in range(COL)
            if not (col == 0 and row == 0)
        ]
        sq_reduce_net = ttl.PipeNet(sq_reduce_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        c_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        sq_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_bc_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        asq_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        red_step_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        recv_c_cb = ttl.make_dataflow_buffer_like(
            out, shape=(1, 1), block_count=max(2, Kp - 1))
        recv_sq_cb = ttl.make_dataflow_buffer_like(
            scaler, shape=(1, 1), block_count=max(2, Kp - 1))

        @ttl.compute()
        def compute():
            col_c, row_c = ttl.node(dims=2)
            sc = sc_cb.wait()

            a0 = a_cb.wait()
            b0 = b_cb.wait()
            c_cb.reserve().store(a0 @ b0)
            asq_cb.reserve().store(a0 * a0)
            sq_cb.reserve().store(
                ttl.math.reduce_sum(asq_cb.wait(), sc, dims=[1])
            )

            for _ in range(K_BPN - 1):
                ak = a_cb.wait()
                bk = b_cb.wait()
                prev_c = c_cb.wait()
                c_cb.reserve().store(prev_c + ak @ bk)

                asq_cb.reserve().store(ak * ak)
                red_step_cb.reserve().store(
                    ttl.math.reduce_sum(asq_cb.wait(), sc, dims=[1])
                )
                prev_sq = sq_cb.wait()
                sq_cb.reserve().store(prev_sq + red_step_cb.wait())

            if col_c == 0 and row_c == 0:
                for _ in range(Kp - 1):
                    prev_c = c_cb.wait()
                    r = recv_c_cb.wait()
                    new_c = c_cb.reserve()
                    new_c.store(prev_c + r)
                for _ in range(Kp - 1):
                    prev_sq = sq_cb.wait()
                    r = recv_sq_cb.wait()
                    new_sq = sq_cb.reserve()
                    new_sq.store(prev_sq + r)

                sq_total = sq_cb.wait()
                inv_bc = inv_bc_cb.reserve()
                inv_bc.store(ttl.math.broadcast(
                    ttl.math.rsqrt(
                        sq_total * ttl.math.fill(sq_total, inv_D)
                        + ttl.math.fill(sq_total, rms_eps)
                    ),
                    inv_bc, dims=[1],
                ))
                c_total = c_cb.wait()
                out_cb.reserve().store(c_total * inv_bc_cb.wait())

        @ttl.datamovement()
        def dm_read():
            col_c, row_c = ttl.node(dims=2)
            k_p = row_c * COL + col_c

            ttl.copy(scaler[0, 0], sc_cb.reserve()).wait()

            for kb_local in range(K_BPN):
                kc = k_p * K_BPN + kb_local
                ttl.copy(a[0, kc], a_cb.reserve()).wait()
                ttl.copy(b[kc, 0], b_cb.reserve()).wait()

            is_root = (col_c == 0 and row_c == 0)
            if is_root:
                def recv_c(pipe):
                    blk = recv_c_cb.reserve()
                    ttl.copy(pipe, blk).wait()
                c_reduce_net.if_dst(recv_c)

                def recv_sq(pipe):
                    blk = recv_sq_cb.reserve()
                    ttl.copy(pipe, blk).wait()
                sq_reduce_net.if_dst(recv_sq)
            else:
                p_c = c_cb.wait()
                def send_c(pipe):
                    ttl.copy(p_c, pipe).wait()
                c_reduce_net.if_src(send_c)

                p_sq = sq_cb.wait()
                def send_sq(pipe):
                    ttl.copy(p_sq, pipe).wait()
                sq_reduce_net.if_src(send_sq)

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            if col_c == 0 and row_c == 0:
                ttl.copy(out_cb.wait(), out[0, 0]).wait()

    return norm_fn_ksplit_kernel


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


def _compile_mhc_apply_mix_h_kernel(num_tokens: int, h_tiles: int):
    """Inlined from tt-lang-kernels/pre_apply_mix_h.py.

    h-axis-sharded variant: flattens (token, h_tile) into one work axis so
    `grid="auto"` distributes across all cores. At decode (num_tokens=1,
    h_tiles=128) this saturates ~130 BH cores (~0.18ms vs 0.25ms single-core,
    1.36x, PCC 1.0). Each h-tile is independent — no reduce needed.
    """
    total_work = num_tokens * h_tiles

    @ttl.operation(grid="auto")
    def apply_mix_h_kernel(x, mix, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

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

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    mix_raw = mix_dfb.wait()
                    mx = mix_bc_dfb.reserve()
                    mx.store(ttl.math.broadcast(mix_raw, mx, dims=[1]))
                    mix_bc = mix_bc_dfb.wait()

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
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    global_t = global_w // h_tiles
                    global_h = global_w % h_tiles
                    ttl.copy(mix[global_t, 0], mix_dfb.reserve()).wait()
                    ttl.copy(x[global_t, global_h], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    global_t = global_w // h_tiles
                    global_h = global_w % h_tiles
                    ttl.copy(out_dfb.wait(), out[global_t, global_h]).wait()

    return apply_mix_h_kernel


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


def _compile_compressor_slot_shift_kernel(num_buffers: int, ratio_pad: int, d: int):
    """Inlined from tt-lang-kernels/compressor_slot_shift.py.

    Replaces the per-buffer ratio-many `ttnn.kv_cache.update_cache_for_token_`
    calls in DeviceCompressor.forward_device (overlap=True, ratio=4):

        for buf in (kv_state_front, kv_state_back, score_state_front, score_state_back):
            for i in range(ratio):
                slot_src = ttnn.slice(buf, [0,0,ratio+i,0], [B,1,ratio+i+1,d])
                ttnn.kv_cache.update_cache_for_token_(buf, slot_src, i, 0)

    16 ttnn dispatches per emit step (4 buffers x 4 slot writes), each
    serialized on host. We collapse to one tt-lang kernel per buffer (4
    dispatches total).

    Math: out[i, :] = buf[i + ratio, :] for i < ratio
          out[i, :] = buf[i, :]         for i >= ratio
    Expressed as a tile matmul: out = P @ buf where P is the [ratio_pad,
    ratio_pad] shift matrix (P[i, i+ratio]=1 for i<ratio, P[i,i]=1 else).

    V4-Flash decode: ratio=4, ratio_pad=32, num_buffers=1 (per-call), d=512
    for the attention compressor and d=128 for the indexer's compressor.
    Constraint: ratio_pad == TILE == 32 (within-tile shift via matmul; multi-
    tile P would need different math).
    """
    if ratio_pad % _RMS_TILE != 0:
        raise ValueError(f"ratio_pad={ratio_pad} not multiple of TILE={_RMS_TILE}")
    if d % _RMS_TILE != 0:
        raise ValueError(f"d={d} not multiple of TILE={_RMS_TILE}")
    if ratio_pad != _RMS_TILE:
        raise ValueError(
            f"ratio_pad={ratio_pad} > TILE={_RMS_TILE} unsupported "
            f"(V4-Flash uses ratio_pad=32; multi-tile P needs different math)"
        )

    M_tiles = num_buffers
    N_tiles = d // _RMS_TILE
    total_work = M_tiles * N_tiles

    @ttl.operation(grid="auto")
    def slot_shift_kernel(buf, P, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        buf_dfb = ttl.make_dataflow_buffer_like(buf, shape=(1, 1), block_count=2)
        P_dfb = ttl.make_dataflow_buffer_like(P, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            P_tile = P_dfb.wait()

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    buf_tile = buf_dfb.wait()
                    out_dfb.reserve().store(P_tile @ buf_tile)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(P[0, 0], P_dfb.reserve()).wait()
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    m = global_w // N_tiles
                    n = global_w % N_tiles
                    ttl.copy(buf[m, n], buf_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    m = global_w // N_tiles
                    n = global_w % N_tiles
                    ttl.copy(out_dfb.wait(), out[m, n]).wait()

    return slot_shift_kernel


def _compressor_shift_matrix(ratio: int, ratio_pad: int) -> torch.Tensor:
    """Build the [ratio_pad, ratio_pad] shift matrix P for slot-shift kernel.
    P @ buf produces:
        out[i] = buf[i + ratio] for i < ratio
        out[i] = buf[i]         for i >= ratio   (incl. padding rows)
    """
    if ratio_pad < 2 * ratio:
        raise ValueError(f"ratio_pad={ratio_pad} must be >= 2*ratio={2 * ratio}")
    P = torch.zeros(ratio_pad, ratio_pad, dtype=torch.bfloat16)
    for i in range(ratio):
        P[i, i + ratio] = 1.0
    for i in range(ratio, ratio_pad):
        P[i, i] = 1.0
    return P


def _compile_compressor_softmax_sum_norm_kernel(ratio: int, ratio_pad: int,
                                                d: int, rms_eps: float):
    """Inlined from tt-lang-kernels/compressor_softmax_sum_norm.py.

    Fuses the slice+concat view + softmax + weighted-sum + RMSNorm for the
    Compressor emit branch (overlap=True) into one tt-lang kernel:
        score_view = mf*sc_front + mb*sc_back + mp*sc_front  # padding rows pull -inf
        kv_view    = mf*kv_front + mb*kv_back                # padding rows zero
        sm         = softmax(score_view, dim=0)              # over all 32 rows
        kv_sum     = sum(kv_view * sm, dim=0)                # row 0 of [TILE, d]
        rms        = rsqrt(mean(kv_sum^2) + rms_eps)
        out        = kv_sum * gamma * rms                    # row 0 only

    Replaces 8 ttnn ops (slice x4, concat x2, softmax, multiply, sum,
    reshape, pad, RMSNorm) with one dispatch. Single core (grid=(1,1));
    work is small (n_tiles=d//TILE) so ttnn-chain dispatch overhead
    dominates the kernel's compute cost.
    """
    if ratio_pad != _RMS_TILE:
        raise ValueError(
            f"ratio_pad={ratio_pad} != TILE={_RMS_TILE} unsupported "
            f"(V4-Flash uses 32)")
    if d % _RMS_TILE != 0:
        raise ValueError(f"d={d} not multiple of TILE={_RMS_TILE}")
    if 2 * ratio > ratio_pad:
        raise ValueError(f"2*ratio={2 * ratio} > ratio_pad={ratio_pad}")

    n_tiles = d // _RMS_TILE
    inv_d = 1.0 / d

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
    def cssn_kernel(kv_front, kv_back, sc_front, sc_back,
                    mask_front, mask_back, mask_pad,
                    gamma, scaler, out):
        kvf_dfb = ttl.make_dataflow_buffer_like(kv_front, shape=(1, 1), block_count=2)
        kvb_dfb = ttl.make_dataflow_buffer_like(kv_back, shape=(1, 1), block_count=2)
        scf_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        scb_dfb = ttl.make_dataflow_buffer_like(sc_back, shape=(1, 1), block_count=2)
        mf_dfb = ttl.make_dataflow_buffer_like(mask_front, shape=(1, 1), block_count=1)
        mb_dfb = ttl.make_dataflow_buffer_like(mask_back, shape=(1, 1), block_count=1)
        mp_dfb = ttl.make_dataflow_buffer_like(mask_pad, shape=(1, 1), block_count=1)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        gamma_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        sv_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        kv_dfb = ttl.make_dataflow_buffer_like(kv_front, shape=(1, 1), block_count=2)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        max_bc_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        exp_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        invsum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        invsum_bc_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        sm_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        weighted_dfb = ttl.make_dataflow_buffer_like(kv_front, shape=(1, 1), block_count=2)
        kv_sum_partial_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        kv_sum_stash_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=n_tiles)
        ks_sq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        ssq_step_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        ssq_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rms_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rms_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            sc = sc_dfb.wait()
            mf = mf_dfb.wait()
            mb = mb_dfb.wait()
            mp = mp_dfb.wait()

            for ct in range(n_tiles):
                kvf = kvf_dfb.wait()
                kvb = kvb_dfb.wait()
                scf = scf_dfb.wait()
                scb = scb_dfb.wait()

                sv_dfb.reserve().store(mf * scf + mb * scb + mp * scf)
                sv = sv_dfb.wait()

                kv_dfb.reserve().store(mf * kvf + mb * kvb)
                kvv = kv_dfb.wait()

                max_dfb.reserve().store(ttl.math.reduce_max(sv, sc, dims=[0]))
                row_max = max_dfb.wait()
                max_bc = max_bc_dfb.reserve()
                max_bc.store(ttl.math.broadcast(row_max, max_bc, dims=[0]))
                row_max_bc = max_bc_dfb.wait()

                exp_dfb.reserve().store(ttl.math.exp(sv - row_max_bc))
                exp_view = exp_dfb.wait()

                sum_dfb.reserve().store(ttl.math.reduce_sum(exp_view, sc, dims=[0]))
                row_sum = sum_dfb.wait()
                invsum_dfb.reserve().store(ttl.math.recip(row_sum))
                row_invsum = invsum_dfb.wait()
                invsum_bc = invsum_bc_dfb.reserve()
                invsum_bc.store(ttl.math.broadcast(row_invsum, invsum_bc, dims=[0]))
                row_invsum_bc = invsum_bc_dfb.wait()

                sm_dfb.reserve().store(exp_view * row_invsum_bc)
                sm = sm_dfb.wait()

                weighted_dfb.reserve().store(kvv * sm)
                w = weighted_dfb.wait()

                kv_sum_partial_dfb.reserve().store(
                    ttl.math.reduce_sum(w, sc, dims=[0])
                )
                ks = kv_sum_partial_dfb.wait()
                kv_sum_stash_dfb.reserve().store(ks)

                ks_sq_dfb.reserve().store(ks * ks)
                ssq_step_dfb.reserve().store(
                    ttl.math.reduce_sum(ks_sq_dfb.wait(), sc, dims=[0, 1])
                )
                step = ssq_step_dfb.wait()
                if ct == 0:
                    ssq_acc_dfb.reserve().store(step)
                else:
                    prev = ssq_acc_dfb.wait()
                    ssq_acc_dfb.reserve().store(prev + step)

            ssq = ssq_acc_dfb.wait()
            rms_dfb.reserve().store(
                ttl.math.rsqrt(
                    ssq * ttl.math.fill(ssq, inv_d) + ttl.math.fill(ssq, rms_eps)
                )
            )
            rms_scalar = rms_dfb.wait()
            rms_bc = rms_bc_dfb.reserve()
            rms_bc.store(ttl.math.broadcast(rms_scalar, rms_bc, dims=[0, 1]))
            rms_full = rms_bc_dfb.wait()

            for ct in range(n_tiles):
                ks = kv_sum_stash_dfb.wait()
                g = gamma_dfb.wait()
                out_dfb.reserve().store(ks * g * rms_full)

        @ttl.datamovement()
        def dm_read():
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            ttl.copy(mask_front[0, 0], mf_dfb.reserve()).wait()
            ttl.copy(mask_back[0, 0], mb_dfb.reserve()).wait()
            ttl.copy(mask_pad[0, 0], mp_dfb.reserve()).wait()

            for ct in range(n_tiles):
                ttl.copy(kv_front[0, ct], kvf_dfb.reserve()).wait()
                ttl.copy(kv_back[0, ct], kvb_dfb.reserve()).wait()
                ttl.copy(sc_front[0, ct], scf_dfb.reserve()).wait()
                ttl.copy(sc_back[0, ct], scb_dfb.reserve()).wait()

            for ct in range(n_tiles):
                ttl.copy(gamma[0, ct], gamma_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            for ct in range(n_tiles):
                ttl.copy(out_dfb.wait(), out[0, ct]).wait()

    return cssn_kernel


def _compressor_softmax_sum_norm_masks(ratio: int):
    """Build the three [TILE, TILE] mask tiles used by cssn_kernel.

    mask_front[i, j] = 1 if i < ratio
    mask_back [i, j] = 1 if ratio <= i < 2*ratio
    mask_pad  [i, j] = 1 if i >= 2*ratio
    """
    mf = torch.zeros(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16)
    mb = torch.zeros(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16)
    mp = torch.zeros(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16)
    mf[:ratio, :] = 1.0
    mb[ratio:2 * ratio, :] = 1.0
    mp[2 * ratio:, :] = 1.0
    return mf, mb, mp


def _compile_act_quant_block_kernel(M_pad: int, N: int, BLOCK: int = 64,
                                     fp8_max: float = 448.0, eps: float = 1e-4):
    """Inlined from tt-lang-kernels/act_quant_block.py.

    Per-block max-abs scale round-trip in bf16:
        amax = max(|x|, dim=last(block))
        s    = max(amax, eps) / fp8_max
        out  = clamp(x / s, -fp8_max, fp8_max) * s
    Replaces 9 ttnn ops in `_device_act_quant_block` (reshape, abs, max,
    maximum, multiply, divide, clamp, multiply, reshape) with 1 dispatch.
    Block matches the V4-Flash decode shape M_pad=32 N=448 (head_dim - rope).
    """
    if N % BLOCK != 0:
        raise ValueError(f"N={N} not divisible by BLOCK={BLOCK}")
    if BLOCK % _RMS_TILE != 0:
        raise ValueError(f"BLOCK={BLOCK} not multiple of TILE={_RMS_TILE}")
    if M_pad % _RMS_TILE != 0:
        raise ValueError(f"M_pad={M_pad} not multiple of TILE={_RMS_TILE}")

    M_tiles = M_pad // _RMS_TILE
    NB = N // BLOCK
    BLOCK_TILES = BLOCK // _RMS_TILE
    total_work = M_tiles * NB
    inv_fp8_max = 1.0 / fp8_max

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def act_quant_kernel(x, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BLOCK_TILES), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_TILES), block_count=2)

        abs_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BLOCK_TILES), block_count=2)
        amax_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        amax_eps_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_s_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BLOCK_TILES), block_count=2)
        s_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BLOCK_TILES), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            sc = sc_dfb.wait()

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    x_blk = x_dfb.wait()
                    abs_dfb.reserve().store(ttl.math.abs(x_blk))
                    amax_dfb.reserve().store(
                        ttl.math.reduce_max(abs_dfb.wait(), sc, dims=[1])
                    )
                    amax = amax_dfb.wait()
                    amax_eps_dfb.reserve().store(
                        amax + ttl.math.fill(amax, eps)
                    )
                    amax_eps = amax_eps_dfb.wait()

                    inv_s_dfb.reserve().store(
                        ttl.math.recip(amax_eps)
                        * ttl.math.fill(amax_eps, fp8_max)
                    )
                    inv_s = inv_s_dfb.wait()
                    inv_s_bc_blk = inv_s_bc_dfb.reserve()
                    inv_s_bc_blk.store(
                        ttl.math.broadcast(inv_s, inv_s_bc_blk, dims=[1])
                    )
                    inv_s_bc = inv_s_bc_dfb.wait()

                    s_dfb.reserve().store(
                        amax_eps * ttl.math.fill(amax_eps, inv_fp8_max)
                    )
                    s = s_dfb.wait()
                    s_bc_blk = s_bc_dfb.reserve()
                    s_bc_blk.store(
                        ttl.math.broadcast(s, s_bc_blk, dims=[1])
                    )
                    s_bc = s_bc_dfb.wait()

                    out_dfb.reserve().store(x_blk * inv_s_bc * s_bc)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    m = global_w // NB
                    nb = global_w % NB
                    c0 = nb * BLOCK_TILES
                    c1 = c0 + BLOCK_TILES
                    blk = x_dfb.reserve()
                    ttl.copy(x[m:m+1, c0:c1], blk).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    m = global_w // NB
                    nb = global_w % NB
                    c0 = nb * BLOCK_TILES
                    c1 = c0 + BLOCK_TILES
                    ttl.copy(out_dfb.wait(), out[m:m+1, c0:c1]).wait()

    return act_quant_kernel


# ============================================================================
# tt-lang kernel cache + named decode-path globals
# ============================================================================
# Each `_compile_*_kernel` factory above returns a fresh `@ttl.operation`
# wrapper, and ttl's JIT keys its compiled-program cache off the wrapper
# identity. So calling a factory N times at the same shape yields N cold JIT
# compiles. Compile cost dominates wall time at K=512, h=128 (fold of 84s/2tok
# -> 322s/2tok was traced to the previous per-instance cache being removed).
#
# Strategy:
#   1. Module-level dict `_TTL_KERNEL_CACHE` keyed on the full factory
#      signature. One compile per unique signature, shared across every
#      DeviceRMSNorm and every DeviceMHC instance.
#   2. Named globals (`ttl_rmsnorm_M32_h4096`, etc.) for each known decode-path
#      kernel. Populated by `prebuild_ttl_decode_kernels(args)` after the
#      model is constructed. Call sites reference these names directly so
#      code flow is explicit.
#   3. `_get_*` cache helpers for prefill and any shape not pre-built — they
#      hit the same module-level dict, so a prefill compile is shared across
#      all 86 layers.
#
# Naming convention: `ttl_<kernel-name>_<key-shape-fields>` in element units
# (M32_h4096 = 32 rows by 4096-wide hidden). This matches how the kernels
# are debugged at the source level.

_TTL_KERNEL_CACHE: dict = {}


def _cached_kernel(key, factory):
    k = _TTL_KERNEL_CACHE.get(key)
    if k is None:
        k = factory()
        _TTL_KERNEL_CACHE[key] = k
    return k


def _get_ttl_rmsnorm_kernel(num_row_tiles: int, h_tiles: int,
                             rms_eps: float, inv_D: float):
    return _cached_kernel(
        ("rmsnorm", num_row_tiles, h_tiles, rms_eps, inv_D),
        lambda: _compile_rmsnorm_kernel(num_row_tiles, h_tiles, rms_eps, inv_D),
    )


def _get_ttl_mhc_norm_fn_kernel(num_out_tiles: int, K_tiles: int,
                                 rms_eps: float, inv_D: float):
    return _cached_kernel(
        ("mhc_norm_fn", num_out_tiles, K_tiles, rms_eps, inv_D),
        lambda: _compile_mhc_norm_fn_kernel(
            num_out_tiles, K_tiles, rms_eps, inv_D),
    )


def _get_ttl_mhc_norm_fn_ksplit_kernel(K_tiles: int, Kp: int,
                                        rms_eps: float, inv_D: float):
    return _cached_kernel(
        ("mhc_norm_fn_ksplit", K_tiles, Kp, rms_eps, inv_D),
        lambda: _compile_mhc_norm_fn_ksplit_kernel(
            K_tiles, Kp, rms_eps, inv_D),
    )


def _get_ttl_mhc_split_mixes_kernel(num_tiles: int):
    return _cached_kernel(
        ("mhc_split_mixes", num_tiles),
        lambda: _compile_mhc_split_mixes_kernel(num_tiles),
    )


def _get_ttl_mhc_sinkhorn_kernel(num_slices: int, repeat: int, eps: float):
    return _cached_kernel(
        ("mhc_sinkhorn", num_slices, repeat, eps),
        lambda: _compile_mhc_sinkhorn_kernel(num_slices, repeat, eps),
    )


def _get_ttl_mhc_apply_mix_kernel(num_tokens: int, h_tiles: int):
    return _cached_kernel(
        ("mhc_apply_mix", num_tokens, h_tiles),
        lambda: _compile_mhc_apply_mix_kernel(num_tokens, h_tiles),
    )


def _get_ttl_mhc_apply_mix_h_kernel(num_tokens: int, h_tiles: int):
    return _cached_kernel(
        ("mhc_apply_mix_h", num_tokens, h_tiles),
        lambda: _compile_mhc_apply_mix_h_kernel(num_tokens, h_tiles),
    )


def _get_ttl_mhc_post_kernel(num_tokens: int, h_tiles: int):
    return _cached_kernel(
        ("mhc_post", num_tokens, h_tiles),
        lambda: _compile_mhc_post_kernel(num_tokens, h_tiles),
    )


def _get_ttl_compressor_slot_shift_kernel(num_buffers: int, ratio_pad: int, d: int):
    return _cached_kernel(
        ("compressor_slot_shift", num_buffers, ratio_pad, d),
        lambda: _compile_compressor_slot_shift_kernel(num_buffers, ratio_pad, d),
    )


def _get_ttl_compressor_softmax_sum_norm_kernel(ratio: int, ratio_pad: int,
                                                 d: int, rms_eps: float):
    return _cached_kernel(
        ("compressor_softmax_sum_norm", ratio, ratio_pad, d, rms_eps),
        lambda: _compile_compressor_softmax_sum_norm_kernel(
            ratio, ratio_pad, d, rms_eps),
    )


def _get_ttl_act_quant_block_kernel(M_pad: int, N: int, BLOCK: int = 64,
                                     fp8_max: float = 448.0, eps: float = 1e-4):
    return _cached_kernel(
        ("act_quant_block", M_pad, N, BLOCK, fp8_max, eps),
        lambda: _compile_act_quant_block_kernel(M_pad, N, BLOCK, fp8_max, eps),
    )


# Named decode-path kernels. Set by `prebuild_ttl_decode_kernels(args)`.
# Each name encodes the shape it serves so call sites read like
# `ttl_rmsnorm_M32_h4096(x, gamma, sc, out)`.

ttl_rmsnorm_M32_h4096 = None    # attn_norm, ffn_norm (hidden=4096)
ttl_rmsnorm_M32_h1024 = None    # q_norm (q_lora_rank=1024)
ttl_rmsnorm_M32_h512  = None    # kv_norm, attn.compressor.norm (head_dim=512)
ttl_rmsnorm_M32_h128  = None    # indexer.compressor.norm (index_head_dim=128)

ttl_mhc_norm_fn_ksplit_K512_Kp8 = None   # decode (D=16384 -> K_tiles=512)
ttl_mhc_split_mixes_M1 = None             # decode (num_tokens_pad=32 -> 1 tile)
ttl_mhc_sinkhorn_N1_R20 = None            # decode (num_tokens=1, sinkhorn=20)
ttl_mhc_apply_mix_h_N1_h128 = None        # decode (num_tokens=1, h_tiles=128)
ttl_mhc_post_N1_h128 = None               # decode (num_tokens=1, h_tiles=128)

# DeviceCompressor slot-shift (overlap=True, ratio=4, ratio_pad=32). One
# kernel call per state buffer collapses 4 ttnn dispatches into 1; with 4
# buffers per emit step that's 4 dispatches vs the old 16-dispatch loop.
ttl_compressor_slot_shift_B1_pad32_d512 = None  # attention compressor (head_dim=512)
ttl_compressor_slot_shift_B1_pad32_d128 = None  # indexer compressor (index_head_dim=128)

# Fused slice/concat + softmax + weighted-sum + RMSNorm kernel for the
# Compressor emit branch (overlap=True). One dispatch replaces ~8 ttnn ops.
ttl_compressor_softmax_sum_norm_pad32_d512 = None  # attention compressor
ttl_compressor_softmax_sum_norm_pad32_d128 = None  # indexer compressor

# Block-wise act_quant round-trip used in the indexer kv path (head_dim - rope).
# 1 dispatch replaces 9 ttnn ops in `_device_act_quant_block`.
ttl_act_quant_block_M32_N448_B64 = None  # head_dim=512 - rope_head_dim=64


def prebuild_ttl_decode_kernels(args: "ModelArgs"):
    """Eagerly build every tt-lang kernel the decode path uses and bind them
    to module globals. Called once after model construction so the first
    decode step doesn't pay 7+ JIT compiles up front.

    Prefill kernels (variable num_tokens) are not pre-built here; they go
    through the `_get_*` cache helpers, which still get the cross-layer
    sharing benefit (one compile for all 86 layers per Mt value)."""
    global ttl_rmsnorm_M32_h4096, ttl_rmsnorm_M32_h1024
    global ttl_rmsnorm_M32_h512, ttl_rmsnorm_M32_h128
    global ttl_mhc_norm_fn_ksplit_K512_Kp8, ttl_mhc_split_mixes_M1
    global ttl_mhc_sinkhorn_N1_R20
    global ttl_mhc_apply_mix_h_N1_h128, ttl_mhc_post_N1_h128
    global ttl_compressor_slot_shift_B1_pad32_d512
    global ttl_compressor_slot_shift_B1_pad32_d128
    global ttl_compressor_softmax_sum_norm_pad32_d512
    global ttl_compressor_softmax_sum_norm_pad32_d128
    global ttl_act_quant_block_M32_N448_B64

    eps = args.norm_eps
    ttl_rmsnorm_M32_h4096 = _get_ttl_rmsnorm_kernel(
        1, args.dim // _RMS_TILE, eps, 1.0 / args.dim)
    ttl_rmsnorm_M32_h1024 = _get_ttl_rmsnorm_kernel(
        1, args.q_lora_rank // _RMS_TILE, eps, 1.0 / args.q_lora_rank)
    ttl_rmsnorm_M32_h512 = _get_ttl_rmsnorm_kernel(
        1, args.head_dim // _RMS_TILE, eps, 1.0 / args.head_dim)
    ttl_rmsnorm_M32_h128 = _get_ttl_rmsnorm_kernel(
        1, args.index_head_dim // _RMS_TILE, eps, 1.0 / args.index_head_dim)

    D = args.hc_mult * args.dim
    K_tiles = D // _MHC_TILE
    h_tiles = args.dim // _MHC_TILE
    ttl_mhc_norm_fn_ksplit_K512_Kp8 = _get_ttl_mhc_norm_fn_ksplit_kernel(
        K_tiles, _MHC_NORM_FN_KP, eps, 1.0 / D)
    ttl_mhc_split_mixes_M1 = _get_ttl_mhc_split_mixes_kernel(1)
    ttl_mhc_sinkhorn_N1_R20 = _get_ttl_mhc_sinkhorn_kernel(
        1, args.hc_sinkhorn_iters, args.hc_eps)
    ttl_mhc_apply_mix_h_N1_h128 = _get_ttl_mhc_apply_mix_h_kernel(1, h_tiles)
    ttl_mhc_post_N1_h128 = _get_ttl_mhc_post_kernel(1, h_tiles)

    ttl_compressor_slot_shift_B1_pad32_d512 = _get_ttl_compressor_slot_shift_kernel(
        1, _RMS_TILE, args.head_dim)
    ttl_compressor_slot_shift_B1_pad32_d128 = _get_ttl_compressor_slot_shift_kernel(
        1, _RMS_TILE, args.index_head_dim)

    # Compressor emit branch (overlap=True ↔ compress_ratio=4). The kernel
    # softmaxes over all 32 ratio_pad rows; padding rows pull -inf from the
    # score buffer (preserved by slot-shift) and 0 from the kv buffer.
    ttl_compressor_softmax_sum_norm_pad32_d512 = (
        _get_ttl_compressor_softmax_sum_norm_kernel(
            4, _RMS_TILE, args.head_dim, eps))
    ttl_compressor_softmax_sum_norm_pad32_d128 = (
        _get_ttl_compressor_softmax_sum_norm_kernel(
            4, _RMS_TILE, args.index_head_dim, eps))

    ttl_act_quant_block_M32_N448_B64 = _get_ttl_act_quant_block_kernel(
        _RMS_TILE, args.head_dim - args.rope_head_dim, 64)


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
        # Pre-allocated destination for the post-norm slice that feeds the
        # downstream attn / ffn forward. ttnn.slice accepts output_tensor=, so
        # _block_forward can drop a per-call alloc here.
        self._norm_slice_tt = ttnn.from_torch(
            torch.zeros(num_tokens, self.hidden, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        # Pre-allocated slice scratch for hc_pre_device / hc_post_device's
        # a_sliced ([1, mhc*hidden] fp32, tile-aligned). The other slices in
        # this chain (comb_sliced, pre_sliced, post_sliced) target sub-tile
        # widths (mhc=4 or mhc*mhc=16), which slice's preallocated-output
        # path rejects (padded_shape mismatch), so they keep per-call allocs.
        self._a_sliced_scratch_tt = ttnn.from_torch(
            torch.zeros(num_tokens, self.hc_mult * self.hidden,
                        dtype=torch.float32),
            device=self.mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self._upload_mapper = ttnn.ReplicateTensorToMesh(self.mesh)

    def _norm_fn_kernel(self, num_out_tiles: int):
        K_tiles = self.D // _MHC_TILE
        if num_out_tiles == 1 and K_tiles % _MHC_NORM_FN_KP == 0:
            if ttl_mhc_norm_fn_ksplit_K512_Kp8 is not None and K_tiles == 512:
                return ttl_mhc_norm_fn_ksplit_K512_Kp8
            return _get_ttl_mhc_norm_fn_ksplit_kernel(
                K_tiles, _MHC_NORM_FN_KP, self.norm_eps, 1.0 / self.D)
        return _get_ttl_mhc_norm_fn_kernel(
            num_out_tiles, K_tiles, self.norm_eps, 1.0 / self.D)

    def _split_mixes_kernel(self, num_out_tiles: int):
        if ttl_mhc_split_mixes_M1 is not None and num_out_tiles == 1:
            return ttl_mhc_split_mixes_M1
        return _get_ttl_mhc_split_mixes_kernel(num_out_tiles)

    def _sinkhorn_kernel(self, num_slices: int):
        if (ttl_mhc_sinkhorn_N1_R20 is not None
                and num_slices == 1 and self.sinkhorn_iters == 20):
            return ttl_mhc_sinkhorn_N1_R20
        return _get_ttl_mhc_sinkhorn_kernel(
            num_slices, self.sinkhorn_iters, self.hc_eps)

    def _apply_mix_kernel(self, num_tokens: int):
        h_tiles = self.hidden // _MHC_TILE
        if num_tokens == 1:
            if ttl_mhc_apply_mix_h_N1_h128 is not None and h_tiles == 128:
                return ttl_mhc_apply_mix_h_N1_h128
            return _get_ttl_mhc_apply_mix_h_kernel(num_tokens, h_tiles)
        return _get_ttl_mhc_apply_mix_kernel(num_tokens, h_tiles)

    def _post_kernel(self, num_tokens: int):
        h_tiles = self.hidden // _MHC_TILE
        if (ttl_mhc_post_N1_h128 is not None
                and num_tokens == 1 and h_tiles == 128):
            return ttl_mhc_post_N1_h128
        return _get_ttl_mhc_post_kernel(num_tokens, h_tiles)

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

    def hc_pre_device(self, num_tokens: int, num_tokens_pad: int,
                      a_tt=None):
        """Pure-device hc_pre body. If a_tt is None, reads from
        self._a_upload_tt. Otherwise uses the provided device tensor
        (must be [num_tokens_pad, mhc*hidden] fp32 TILE_LAYOUT). Runs
        norm_fn -> split_mixes -> sinkhorn -> apply_mix and returns the
        apply_mix output device tensor [num_tokens * TILE, hidden].
        Stashes a_tt/post_tt/comb_sk_out_tt for the matching hc_post."""
        ttnn = self._ttnn
        mhc = self.hc_mult
        hidden = self.hidden
        if a_tt is None:
            a_tt = self._a_upload_tt
        mixes_tt = self._mixes_tt

        self._norm_fn_kernel(num_tokens_pad // _MHC_TILE)(
            a_tt, self.fn_tt, self.scaler_tt, mixes_tt)

        pre_tt = self._pre_tt
        post_tt = self._post_tt
        comb_tt = self._comb_tt
        self._split_mixes_kernel(num_tokens_pad // _MHC_TILE)(
            mixes_tt, self.split_scale_tt, self.split_base_tt,
            self.split_pre_mask_tt, self.split_pre_eps_tt,
            self.split_post_mult_mask_tt, self.split_comb_mask_tt,
            pre_tt, post_tt, comb_tt,
        )

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

        pre_sliced = ttnn.slice(pre_tt, [0, 0], [num_tokens, mhc])
        pre_3d = ttnn.reshape(pre_sliced, [num_tokens, mhc, 1])
        mix_padded = ttnn.pad(
            pre_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, _MHC_TILE - 1)],
            value=0.0,
        )
        mix_tt = ttnn.reshape(mix_padded, [num_tokens * _MHC_TILE, _MHC_TILE])

        ttnn.slice(a_tt, [0, 0], [num_tokens, mhc * hidden],
                   output_tensor=self._a_sliced_scratch_tt)
        a_3d = ttnn.reshape(self._a_sliced_scratch_tt, [num_tokens, mhc, hidden])
        x_padded = ttnn.pad(
            a_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, 0)],
            value=0.0,
        )
        x_tt = ttnn.reshape(x_padded, [num_tokens * _MHC_TILE, hidden])
        out_tt = self._apply_mix_out_tt
        self._apply_mix_kernel(num_tokens)(
            x_tt, mix_tt, self.scaler_tt, out_tt)

        self._stash_a_tt = a_tt
        self._stash_post_tt = post_tt
        self._stash_comb_sk_tt = comb_sk_out_tt
        return out_tt

    def hc_pre_with_upload(self, x: torch.Tensor):
        """Upload x and run hc_pre_device. Returns the device tensor
        [num_tokens * TILE, hidden] fp32 directly. Use when chaining on
        device (no download)."""
        ttnn = self._ttnn
        B, S, mhc, hidden = x.shape
        if mhc != self.hc_mult or hidden != self.hidden:
            raise ValueError(
                f"x shape mismatch: got mhc={mhc}, hidden={hidden}; "
                f"expected mhc={self.hc_mult}, hidden={self.hidden}")
        num_tokens = B * S
        num_tokens_pad = -(-num_tokens // _MHC_TILE) * _MHC_TILE
        a_packed = _mhc_pack_residual(x, num_tokens_pad)
        host_mesh = ttnn.from_torch(
            a_packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._a_upload_tt)
        return self.hc_pre_device(num_tokens, num_tokens_pad)

    def hc_pre(self, x: torch.Tensor):
        """[B, S, mhc, hidden] -> y [B, S, hidden].

        Wrapper: uploads x, runs hc_pre_device, downloads y. Use
        hc_pre_device directly when the caller already has a device tensor
        (block_forward_device chain).
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

        a_packed = _mhc_pack_residual(x, num_tokens_pad)
        host_mesh = ttnn.from_torch(
            a_packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._a_upload_tt)

        out_tt = self.hc_pre_device(num_tokens, num_tokens_pad)

        out_packed = _readback_replicated_2d(
            ttnn, out_tt, self.mesh, self.mesh_shape)
        y_flat = _mhc_unpack_apply_mix_out(out_packed, num_tokens, hidden)
        return y_flat.view(B, S, hidden).to(out_dtype)

    def hc_post_device(self, num_tokens: int, x_tt=None):
        """Pure-device hc_post body. If x_tt is None, reads from
        self._x_upload_tt. Otherwise uses the provided device tensor
        (must be [num_tokens * TILE, hidden] fp32 TILE_LAYOUT with rows
        0..mhc-1 of each block holding the x copy). The matching
        hc_pre_device must have stashed a/post/comb_sk tensors. Runs
        the post kernel and returns the output device tensor
        [num_tokens * TILE, hidden]. Clears the stash afterwards. No host
        transfers."""
        ttnn = self._ttnn
        mhc = self.hc_mult
        hidden = self.hidden

        a_tt_stashed = self._stash_a_tt
        post_tt_stashed = self._stash_post_tt
        comb_sk_tt_stashed = self._stash_comb_sk_tt
        if a_tt_stashed is None or post_tt_stashed is None or comb_sk_tt_stashed is None:
            raise RuntimeError("hc_post_device called without matching hc_pre_device stash")
        self._stash_a_tt = None
        self._stash_post_tt = None
        self._stash_comb_sk_tt = None

        ttnn.slice(a_tt_stashed, [0, 0], [num_tokens, mhc * hidden],
                   output_tensor=self._a_sliced_scratch_tt)
        a_3d = ttnn.reshape(self._a_sliced_scratch_tt, [num_tokens, mhc, hidden])
        res_padded = ttnn.pad(
            a_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, 0)],
            value=0.0,
        )
        res_tt = ttnn.reshape(res_padded, [num_tokens * _MHC_TILE, hidden])

        post_sliced = ttnn.slice(
            post_tt_stashed, [0, mhc], [num_tokens, 2 * mhc])
        post_3d = ttnn.reshape(post_sliced, [num_tokens, mhc, 1])
        post_padded = ttnn.pad(
            post_3d,
            padding=[(0, 0), (0, _MHC_TILE - mhc), (0, _MHC_TILE - 1)],
            value=0.0,
        )
        post_tt = ttnn.reshape(post_padded, [num_tokens * _MHC_TILE, _MHC_TILE])

        comb_3d = ttnn.reshape(
            comb_sk_tt_stashed, [num_tokens, _MHC_TILE, _MHC_TILE])
        comb_T_3d = ttnn.transpose(comb_3d, -2, -1)
        comb_tt = ttnn.reshape(
            comb_T_3d, [num_tokens * _MHC_TILE, _MHC_TILE])

        if x_tt is None:
            x_tt = self._x_upload_tt
        out_tt = self._post_out_tt
        self._post_kernel(num_tokens)(
            x_tt, res_tt, comb_tt, post_tt, out_tt)
        return out_tt

    def hc_post(self, x: torch.Tensor):
        """x [B, S, hidden] -> [B, S, mhc, hidden].

        Wrapper: uploads x, runs hc_post_device, downloads y. Use
        hc_post_device directly when the caller already has a device tensor.
        """
        ttnn = self._ttnn
        B, S, hidden = x.shape
        mhc = self.hc_mult
        out_dtype = x.dtype
        num_tokens = B * S

        x_2d = x.reshape(num_tokens, hidden)
        x_packed = _mhc_pack_x_bc(x_2d, mhc, num_tokens)
        host_mesh = ttnn.from_torch(
            x_packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)

        out_tt = self.hc_post_device(num_tokens)

        out_packed = _readback_replicated_2d(
            ttnn, out_tt, self.mesh, self.mesh_shape)
        y_flat = _mhc_unpack_post_out(out_packed, num_tokens, mhc, hidden)
        return y_flat.view(B, S, mhc, hidden).to(out_dtype)

    def hc_post_with_upload_device(self, x: torch.Tensor):
        """Upload x (MoE output) and run hc_post_device. Returns the device
        tensor [num_tokens * TILE, hidden] fp32 directly (no download). Use
        when chaining hc_post → next layer's hc_pre on device."""
        ttnn = self._ttnn
        B, S, hidden = x.shape
        if hidden != self.hidden:
            raise ValueError(
                f"x hidden {hidden} != expected {self.hidden}")
        num_tokens = B * S
        x_2d = x.reshape(num_tokens, hidden)
        x_packed = _mhc_pack_x_bc(x_2d, self.hc_mult, num_tokens)
        host_mesh = ttnn.from_torch(
            x_packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)
        return self.hc_post_device(num_tokens)


class DeviceSharedExpert(nn.Module):
    """V4-Flash shared-expert SwiGLU sharded TP=N across the full mesh.

    w1/w3 are col-parallel on inter_dim (each chip holds [dim, inter_dim/N]),
    w2 is row-parallel on inter_dim (each chip holds [inter_dim/N, dim]).
    Pipeline:

        y1 = x @ w1^T              # [1, inter/N] sharded on each chip
        y3 = x @ w3^T              # [1, inter/N] sharded on each chip
        clamp y1, y3                # elementwise on local shards
        mid = silu(y1) * y3        # elementwise on local shards
        partial = mid @ w2^T       # [1, dim] partial sum on each chip
        out = all_reduce(partial)  # [1, dim] replicated

    Kills the 32x DRAM duplication of the full weights (~62 GB across 43
    layers) and gives ~32x compute reduction on the per-layer SwiGLU body.
    The clamp interacts only with local activations (it bounds y1/y3 before
    the silu*y3 product, all of which is elementwise on local shards), so
    the partial sums fed into all_reduce are mathematically identical to
    the original replicated path.
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

        num_chips = self.mesh_shape[0] * self.mesh_shape[1]
        if inter_dim % num_chips != 0:
            raise ValueError(
                f"DeviceSharedExpert requires inter_dim={inter_dim} divisible by "
                f"num_chips={num_chips}"
            )
        self._num_chips = num_chips
        self.inter_per_chip = inter_dim // num_chips

        common = dict(
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # w1, w3: stored as [dim, inter_dim] after transpose. Shard last dim
        # (inter_dim) across all chips → each chip holds [dim, inter/N].
        col_shard = ttnn.ShardTensorToMesh(mesh, dim=-1)
        # w2: stored as [inter_dim, dim] after transpose. Shard first dim
        # (inter_dim) across all chips → each chip holds [inter/N, dim].
        row_shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        self.w1_tt = ttnn.as_tensor(
            _weight_to_bf16(cpu_w1).transpose(0, 1).contiguous(),
            mesh_mapper=col_shard, **common)
        self.w3_tt = ttnn.as_tensor(
            _weight_to_bf16(cpu_w3).transpose(0, 1).contiguous(),
            mesh_mapper=col_shard, **common)
        self.w2_tt = ttnn.as_tensor(
            _weight_to_bf16(cpu_w2).transpose(0, 1).contiguous(),
            mesh_mapper=row_shard, **common)
        self._alloc_decode_tensors()

    def _alloc_decode_tensors(self):
        """Pre-allocate per-step buffers for the M=1 SwiGLU pipeline. y1/y3/
        silu/mid hold [1, inter_per_chip] on each chip (shard layout); the
        partial matmul output is replicated-shaped [1, dim] but each chip
        holds a partial sum until all_reduce."""
        ttnn = self._ttnn
        common = dict(
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        col_shard = ttnn.ShardTensorToMesh(self.mesh, dim=-1)
        replicate = ttnn.ReplicateTensorToMesh(self.mesh)
        self._x_upload_tt = ttnn.from_torch(
            torch.zeros(1, self.dim, dtype=torch.bfloat16),
            mesh_mapper=replicate, **common)
        self._y1_tt = ttnn.from_torch(
            torch.zeros(1, self.inter_dim, dtype=torch.bfloat16),
            mesh_mapper=col_shard, **common)
        self._y3_tt = ttnn.from_torch(
            torch.zeros(1, self.inter_dim, dtype=torch.bfloat16),
            mesh_mapper=col_shard, **common)
        self._silu_tt = ttnn.from_torch(
            torch.zeros(1, self.inter_dim, dtype=torch.bfloat16),
            mesh_mapper=col_shard, **common)
        self._mid_tt = ttnn.from_torch(
            torch.zeros(1, self.inter_dim, dtype=torch.bfloat16),
            mesh_mapper=col_shard, **common)
        self._partial_tt = ttnn.from_torch(
            torch.zeros(1, self.dim, dtype=torch.bfloat16),
            mesh_mapper=replicate, **common)
        self._upload_mapper = replicate
        # Trace state: captured on first forward_device call. ttnn.all_reduce
        # allocates its own output buffer (no output_tensor= support); the
        # captured handle is stored in _ar_out_tt and reused via execute_trace.
        self._trace_id = None
        self._ar_out_tt = None

    def _compute_body(self):
        """Pure-device SwiGLU body that reads from the stable _x_upload_tt
        buffer. Caller (forward_device) is responsible for staging the input
        via ttnn.copy before the trace replays this body."""
        ttnn = self._ttnn
        ttnn.matmul(self._x_upload_tt, self.w1_tt,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._y1_tt)
        ttnn.matmul(self._x_upload_tt, self.w3_tt,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._y3_tt)
        if self.swiglu_limit > 0:
            ttnn.clamp(self._y1_tt, max=self.swiglu_limit, output_tensor=self._y1_tt)
            ttnn.clamp(self._y3_tt, min=-self.swiglu_limit, max=self.swiglu_limit,
                       output_tensor=self._y3_tt)
        ttnn.silu(self._y1_tt, output_tensor=self._silu_tt)
        ttnn.multiply(self._silu_tt, self._y3_tt, output_tensor=self._mid_tt)
        ttnn.matmul(self._mid_tt, self.w2_tt,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    optional_output_tensor=self._partial_tt)
        return ttnn.all_reduce(self._partial_tt)

    def forward_device(self, x_tt):
        """Pure-device shared-expert SwiGLU. x_tt: [M=1, dim] bf16 device
        tensor (replicated). Returns out_tt [1, dim] replicated (post
        all_reduce). No host transfers.

        First call captures a trace; subsequent calls stage x_tt into the
        stable _x_upload_tt buffer via device-to-device copy and execute the
        trace with blocking=True. Per-layer instances each capture their
        own trace.
        """
        ttnn = self._ttnn
        ttnn.copy(x_tt, self._x_upload_tt)
        if self._trace_id is None:
            # Warmup outside trace so any lazy device allocations settle.
            self._compute_body()
            ttnn.synchronize_device(self.mesh)
            self._trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._ar_out_tt = self._compute_body()
            ttnn.end_trace_capture(self.mesh, self._trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh)
        else:
            ttnn.execute_trace(self.mesh, self._trace_id, cq_id=0, blocking=True)
        return self._ar_out_tt

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [M=1, dim] -> [M=1, dim]. weights matches Expert.forward
        signature; shared expert is always called with weights=None.

        Wrapper: uploads x, runs forward_device, downloads y. Use
        forward_device when chaining on device.
        """
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
        out_tt = self.forward_device(self._x_upload_tt)

        y = _readback_replicated_2d(
            ttnn, out_tt, self.mesh, self.mesh_shape)[:M]
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
        out = _readback_replicated_2d(
            ttnn, out_tt, self.mesh, self.mesh_shape)[:B]
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
        # Double-buffer scratch for the slot-shift kernel (overlap=True only).
        # Each step writes `out = P @ in`, then we swap in/out so the rotated
        # state becomes the active buffer. Allocated alongside the state.
        self.kv_state_front_out_tt = None
        self.kv_state_back_out_tt = None
        self.score_state_front_out_tt = None
        self.score_state_back_out_tt = None
        self.shift_P_tt = None
        # Constants for the fused softmax+sum+RMSNorm kernel (overlap=True only).
        self.cssn_mask_front_tt = None
        self.cssn_mask_back_tt = None
        self.cssn_mask_pad_tt = None
        self.cssn_out_tt = None
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
        # 2D view shares memory with the 4D buffer; lets the slot_shift +
        # cssn kernels read/write without per-call ttnn.reshape (those are
        # trace-hostile). 4D view is still used by kv_cache.update_cache_.
        self.kv_state_front_2d_tt = ttnn.reshape(
            self.kv_state_front_tt, [ratio_pad, d])
        self.score_state_front_2d_tt = ttnn.reshape(
            self.score_state_front_tt, [ratio_pad, d])
        if self.overlap:
            self.kv_state_back_tt = ttnn.as_tensor(
                kv_init_back.view(B, 1, ratio_pad, d).contiguous(), **rep)
            self.score_state_back_tt = ttnn.as_tensor(
                score_init_back.view(B, 1, ratio_pad, d).contiguous(), **rep)
            self.kv_state_back_2d_tt = ttnn.reshape(
                self.kv_state_back_tt, [ratio_pad, d])
            self.score_state_back_2d_tt = ttnn.reshape(
                self.score_state_back_tt, [ratio_pad, d])

            # Slot-shift double buffers: zeros for kv (data-side), -inf for
            # score (matches the score-state padding semantic). Pads beyond
            # the valid 2*ratio rows are treated as identity by the shift
            # matrix, so init values for those rows are irrelevant.
            zero_pad = torch.zeros(B, ratio_pad, d, dtype=torch.bfloat16)
            ninf_pad = torch.full_like(zero_pad, float("-inf"))
            self.kv_state_front_out_tt = ttnn.as_tensor(
                zero_pad.view(B, 1, ratio_pad, d).contiguous(), **rep)
            self.kv_state_back_out_tt = ttnn.as_tensor(
                zero_pad.view(B, 1, ratio_pad, d).contiguous(), **rep)
            self.score_state_front_out_tt = ttnn.as_tensor(
                ninf_pad.view(B, 1, ratio_pad, d).contiguous(), **rep)
            self.score_state_back_out_tt = ttnn.as_tensor(
                ninf_pad.view(B, 1, ratio_pad, d).contiguous(), **rep)
            self.kv_state_front_out_2d_tt = ttnn.reshape(
                self.kv_state_front_out_tt, [ratio_pad, d])
            self.kv_state_back_out_2d_tt = ttnn.reshape(
                self.kv_state_back_out_tt, [ratio_pad, d])
            self.score_state_front_out_2d_tt = ttnn.reshape(
                self.score_state_front_out_tt, [ratio_pad, d])
            self.score_state_back_out_2d_tt = ttnn.reshape(
                self.score_state_back_out_tt, [ratio_pad, d])
            P_cpu = _compressor_shift_matrix(self.compress_ratio, ratio_pad)
            self.shift_P_tt = ttnn.as_tensor(P_cpu.contiguous(), **rep)

            mf_cpu, mb_cpu, mp_cpu = _compressor_softmax_sum_norm_masks(
                self.compress_ratio)
            self.cssn_mask_front_tt = ttnn.as_tensor(mf_cpu.contiguous(), **rep)
            self.cssn_mask_back_tt = ttnn.as_tensor(mb_cpu.contiguous(), **rep)
            self.cssn_mask_pad_tt = ttnn.as_tensor(mp_cpu.contiguous(), **rep)
            self.cssn_out_tt = ttnn.as_tensor(
                torch.zeros(_RMS_TILE, d, dtype=torch.bfloat16).contiguous(), **rep)

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

        if self.overlap:
            # Fused slice/concat + softmax + weighted-sum + RMSNorm via
            # tt-lang kernel. Padding rows of score_state_*_tt hold -inf
            # (preserved by the slot-shift kernel), so a 32-row softmax
            # behaves as a 2*ratio-row softmax. Output row 0 is the valid
            # normalized kv_sum.
            if d == 512:
                cssn = ttl_compressor_softmax_sum_norm_pad32_d512
            elif d == 128:
                cssn = ttl_compressor_softmax_sum_norm_pad32_d128
            else:
                raise ValueError(
                    f"no compressor_softmax_sum_norm kernel built for d={d}")
            if cssn is None:
                raise RuntimeError(
                    "compressor_softmax_sum_norm kernel not pre-built; "
                    "call prebuild_ttl_decode_kernels(args) first")
            cssn(
                self.kv_state_front_2d_tt,
                self.kv_state_back_2d_tt,
                self.score_state_front_2d_tt,
                self.score_state_back_2d_tt,
                self.cssn_mask_front_tt,
                self.cssn_mask_back_tt,
                self.cssn_mask_pad_tt,
                self.norm_dev.gamma_tt,
                self.norm_dev.sc_tt,
                self.cssn_out_tt,
            )
            kv_2d = ttnn.slice(self.cssn_out_tt, [0, 0], [B, d])
            kv_normed = ttnn.reshape(kv_2d, [B, 1, d])
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

        # Slot-shift via tt-lang kernel (compressor_slot_shift). One kernel
        # call per buffer (4 dispatches) replaces the old 16-dispatch loop
        # of slice + update_cache_for_token_. Compute is `out = P @ in`, then
        # swap in/out so the rotated state is the next step's input.
        if self.overlap:
            if d == 512:
                slot_shift = ttl_compressor_slot_shift_B1_pad32_d512
            elif d == 128:
                slot_shift = ttl_compressor_slot_shift_B1_pad32_d128
            else:
                raise ValueError(f"no compressor_slot_shift kernel built for d={d}")
            if slot_shift is None:
                raise RuntimeError(
                    "compressor_slot_shift kernel not pre-built; "
                    "call prebuild_ttl_decode_kernels(args) first")
            for base in ("kv_state_front", "kv_state_back",
                         "score_state_front", "score_state_back"):
                in_4d = f"{base}_tt"
                out_4d = f"{base}_out_tt"
                in_2d = f"{base}_2d_tt"
                out_2d = f"{base}_out_2d_tt"
                slot_shift(
                    getattr(self, in_2d),
                    self.shift_P_tt,
                    getattr(self, out_2d),
                )
                # Swap both 4D (used by kv_cache.update_cache_) and 2D
                # (used by slot_shift / cssn) attrs together so they stay
                # aliased to the same underlying buffers.
                buf_in_4d = getattr(self, in_4d)
                buf_out_4d = getattr(self, out_4d)
                buf_in_2d = getattr(self, in_2d)
                buf_out_2d = getattr(self, out_2d)
                setattr(self, in_4d, buf_out_4d)
                setattr(self, out_4d, buf_in_4d)
                setattr(self, in_2d, buf_out_2d)
                setattr(self, out_2d, buf_in_2d)

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
        # Window topk index lookup table: [1, 2*win, win] int32, replicated.
        # Per-step we slice row s = _window_topk_row_for_pos(start_pos, win)
        # to get [B=1, 1, win], reproducing get_window_topk_idxs without any
        # host transfer. See _build_window_topk_table for the layout.
        if B != 1:
            raise ValueError(
                f"DeviceAttention only supports B=1, got B={B}; "
                "_win_idxs_table_tt is sized for [1, 2*win, win]."
            )
        win_table = _build_window_topk_table(self.window_size).view(
            1, 2 * self.window_size, self.window_size
        )
        self._win_idxs_table_tt = ttnn.from_torch(
            win_table,
            device=self.mesh, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        # Compress topk index ramp: precomputed [win, win+1, ..., win+max_T-1]
        # as int32 TILE on device, where max_T = max_seq_len // compress_ratio.
        # Slice to [0:T_active] each step to get the compress slot indices
        # without any host arithmetic. Only allocated when compress_ratio is
        # set (layers without a compressor never reach the slice).
        if self.compress_ratio:
            max_T = self.max_seq_len // self.compress_ratio
            ramp = (
                torch.arange(0, max_T, dtype=torch.int32) + self.window_size
            ).view(1, 1, max_T).expand(B, 1, max_T).contiguous()
            self._compress_idxs_ramp_tt = ttnn.from_torch(
                ramp, device=self.mesh, dtype=ttnn.int32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._upload_mapper,
            )
        else:
            self._compress_idxs_ramp_tt = None

        # Pre-allocated slice destinations for the rotary split/recombine
        # pattern. Each forward_device call slices q/kv/o into nope+rope
        # halves before/after rotary; ttnn.slice accepts output_tensor=, so
        # these scratches absorb 6 per-call allocs.
        H, D = self.n_heads, self.head_dim
        rd = self.rope_head_dim
        S = 1
        self._q_nope_scratch_tt = ttnn.from_torch(
            torch.zeros(B, S, H, D - rd, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        self._q_rope_scratch_tt = ttnn.from_torch(
            torch.zeros(B, S, H, rd, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        self._kv_nope_scratch_tt = ttnn.from_torch(
            torch.zeros(B, S, D - rd, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        self._kv_rope_scratch_tt = ttnn.from_torch(
            torch.zeros(B, S, rd, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        self._o_nope_scratch_tt = ttnn.from_torch(
            torch.zeros(B, S, H, D - rd, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        self._o_rope_scratch_tt = ttnn.from_torch(
            torch.zeros(B, S, H, rd, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        # Scratch for the act_quant_block tt-lang kernel: persistent output
        # buffer (kernel writes into it; we then slice row 0). The scaler is
        # a [TILE, TILE] tile of ones used as the block-reduce sentinel.
        self._act_quant_out_tt = ttnn.from_torch(
            torch.zeros(_RMS_TILE, D - rd, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._upload_mapper,
        )
        self._act_quant_sc_tt = ttnn.from_torch(
            torch.ones(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16),
            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
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
        """Wrapper: uploads x, runs forward_device, downloads y. Use
        forward_device directly when chaining on device."""
        ttnn = self._ttnn
        B, S, _ = x.shape

        host_mesh = ttnn.from_torch(
            x.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_mesh, self._x_upload_tt)

        out_tt = self.forward_device(self._x_upload_tt, start_pos)

        out = _readback_replicated_2d(
            ttnn, out_tt, self.mesh, self.mesh_shape)[:B]
        return out.to(x.dtype)

    def forward_device(self, x_tt, start_pos: int):
        """Pure-device attention body. x_tt is the pre-uploaded
        [B, 1, dim] device tensor; returns out_tt [B, 1, dim] device
        tensor. Caller is responsible for upload/download.

        Requires offload_compressor_indexer: attn._device_compressor and
        (when compress_ratio is set) attn._device_indexer must be bound.
        """
        ttnn = self._ttnn
        attn = self.attn
        B = int(x_tt.shape[0])
        S = 1
        H, D = self.n_heads, self.head_dim
        rd = self.rope_head_dim
        win = self.window_size

        device_comp = getattr(attn, "_device_compressor", None)
        device_indexer = getattr(attn, "_device_indexer", None)
        if self.compress_ratio and device_comp is None:
            raise RuntimeError(
                "DeviceAttention.forward_device requires "
                "attn._device_compressor when compress_ratio is set; "
                "run offload_compressor_indexer."
            )

        # Q path: wq_a -> q_norm -> wq_b -> per-head rsqrt-norm -> rotary.
        with _phase("attn.q"):
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
            ttnn.slice(q_tt, [0, 0, 0, 0], [B, S, H, D - rd],
                       output_tensor=self._q_nope_scratch_tt)
            ttnn.slice(q_tt, [0, 0, 0, D - rd], [B, S, H, D],
                       output_tensor=self._q_rope_scratch_tt)
            q_rope = _device_apply_rotary_interleaved(
                ttnn, self._q_rope_scratch_tt, cos_q, sin_q, inverse=False)
            q_tt = ttnn.concat([self._q_nope_scratch_tt, q_rope], dim=-1)

        # KV path: wkv -> kv_norm -> rotary.
        with _phase("attn.kv"):
            kv_tt = attn.wkv.forward_device(x_tt)                 # [B, S, D]
            kv_2d = ttnn.reshape(kv_tt, [B * S, D])
            kv_2d = self._rmsnorm_device(self.kv_norm_dev, kv_2d, B * S)
            kv_tt = ttnn.reshape(kv_2d, [B, S, D])
            cos_kv, sin_kv = self._rotary_tables(start_pos, S, q_dims=3)
            ttnn.slice(kv_tt, [0, 0, 0], [B, S, D - rd],
                       output_tensor=self._kv_nope_scratch_tt)
            ttnn.slice(kv_tt, [0, 0, D - rd], [B, S, D],
                       output_tensor=self._kv_rope_scratch_tt)
            kv_rope = _device_apply_rotary_interleaved(
                ttnn, self._kv_rope_scratch_tt, cos_kv, sin_kv, inverse=False)
            kv_tt = ttnn.concat([self._kv_nope_scratch_tt, kv_rope], dim=-1)

            # Device-side act_quant on the nope region (bf16 round-trip; no fp8
            # cast under the bf16-everywhere precision policy). One tt-lang
            # kernel dispatch replaces the 9-op chain in `_device_act_quant_block`.
            # The kernel only reads tile (0, 0..NB); a logical-[B*S, N] input in
            # TILE_LAYOUT shares the [1 tile-row, N/TILE] grid with [TILE, N], so
            # no host-side pad is required.
            if ttl_act_quant_block_M32_N448_B64 is None:
                raise RuntimeError(
                    "act_quant_block kernel not pre-built; "
                    "call prebuild_ttl_decode_kernels(args) first")
            nope_2d = ttnn.reshape(
                ttnn.slice(kv_tt, [0, 0, 0], [B, S, D - rd]),
                [B * S, D - rd],
            )
            ttl_act_quant_block_M32_N448_B64(
                nope_2d,
                self._act_quant_sc_tt,
                self._act_quant_out_tt,
            )
            kv_nope_q = ttnn.reshape(
                ttnn.slice(self._act_quant_out_tt, [0, 0], [B * S, D - rd]),
                [B, S, D - rd],
            )
            kv_rope_only = ttnn.slice(kv_tt, [0, 0, D - rd], [B, S, D])
            kv_tt = ttnn.concat([kv_nope_q, kv_rope_only], dim=-1)

        # Build the topk index tensor entirely on device.
        # Window slot indices are sliced from the precomputed
        # _win_idxs_table_tt lookup table (no host transfer per step).
        # Compress indices, when applicable:
        #  * device_indexer present: device-side score reduce -> ttnn.topk.
        #  * device_indexer absent (compress_ratio in {2, 128, ...}): slice
        #    the precomputed [win, win+1, ..., win+max_T-1] ramp on device.
        with _phase("attn.topk"):
            win_row = _window_topk_row_for_pos(start_pos, win)
            topk_idxs_dev = ttnn.slice(
                self._win_idxs_table_tt,
                [0, win_row, 0], [B, win_row + 1, win],
            )
            topk_idxs_dev_K = win
            if self.compress_ratio:
                T_active = (start_pos + 1) // self.compress_ratio
                if T_active > 0:
                    if device_indexer is not None:
                        with _phase("attn.topk.indexer"):
                            score_tt = device_indexer.forward_device_score(
                                x_tt, qr_tt, B, start_pos
                            )
                            k = min(device_indexer.index_topk, T_active)
                            score_valid_tt = ttnn.slice(
                                score_tt, [0, 0, 0], [B, S, T_active])
                            _, cmp_idxs_tt = ttnn.topk(
                                score_valid_tt, k=k, dim=-1,
                                largest=True, sorted=True)
                            cmp_idxs_int = ttnn.typecast(cmp_idxs_tt, dtype=ttnn.int32)
                            cmp_idxs_int = ttnn.add(cmp_idxs_int, win)
                    else:
                        cmp_idxs_int = ttnn.slice(
                            self._compress_idxs_ramp_tt,
                            [0, 0, 0], [B, S, T_active],
                        )
                        k = T_active
                    topk_idxs_dev = ttnn.concat(
                        [topk_idxs_dev, cmp_idxs_int], dim=-1
                    )
                    topk_idxs_dev_K = win + k

        # KV cache update: write window slot from kv_tt; the device compressor
        # (when compress_ratio is set) writes compress slot at win + comp_idx
        # into the same buffer.
        with _phase("attn.kv_update"):
            kv_4d = ttnn.reshape(kv_tt, [B, 1, 1, D])
            ttnn.kv_cache.update_cache_for_token_(
                self.kv_cache_tt, kv_4d, start_pos % win, 0
            )
        if self.compress_ratio:
            with _phase("attn.compress"):
                device_comp.forward_device(x_tt, B, start_pos)

        with _phase("attn.sparse"):
            kv_full_tt = ttnn.reshape(self.kv_cache_tt, [self.kv_cache_size_pad, D])
            K = topk_idxs_dev_K
            idxs_tt, valid_tt = self.sparse_attn_dev._idxs_int_tile_to_idxs_and_mask(
                topk_idxs_dev, B, S, K
            )
            o_tt = self.sparse_attn_dev.forward_device(
                q_tt, kv_full_tt, idxs_tt, valid_tt, S, K
            )
            # o_tt: [B, S, H, D].

        with _phase("attn.o"):
            # Inverse rotary on o[..., -rd:].
            cos_o, sin_o = self._rotary_tables(start_pos, S, q_dims=4)
            ttnn.slice(o_tt, [0, 0, 0, 0], [B, S, H, D - rd],
                       output_tensor=self._o_nope_scratch_tt)
            ttnn.slice(o_tt, [0, 0, 0, D - rd], [B, S, H, D],
                       output_tensor=self._o_rope_scratch_tt)
            o_rope = _device_apply_rotary_interleaved(
                ttnn, self._o_rope_scratch_tt, cos_o, sin_o, inverse=True)
            o_tt = ttnn.concat([self._o_nope_scratch_tt, o_rope], dim=-1)

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
        return out_tt

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


def _open_mesh(shape=(4, 8), trace_region_size: int = 100_000_000):
    """Open a Galaxy (TG) mesh device. Default shape (4, 8) matches
    SYSTEM_NAME_TO_MESH_SHAPE["TG"] in tt-metal/models/demos/deepseek_v3.

    FABRIC_1D is correct here even on a 2D mesh: the model only does 1D
    tensor parallelism (sharded along the 8-col axis, replicated along the
    4-row axis). deepseek_v3's Galaxy demo uses the same setting (see
    demo/demo.py:292)."""
    import ttnn
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    return ttnn.open_mesh_device(
        ttnn.MeshShape(*shape), trace_region_size=trace_region_size
    )


def _readback_replicated_2d(ttnn, tensor_tt, mesh, mesh_shape):
    """Read a fully-replicated device tensor back to host as chip (0,0)'s view.

    Replicated tensors hold identical data on every chip, but
    ConcatMesh2dToTensor with `dims=(1, 0)` materialises them as
    `[M1 * d0, M0 * d1, ...rest]` on a `(M0, M1)` mesh. On a (1, 4) mesh
    the result happens to collapse to `[4 * d0, d1]` so a single `[:M]`
    slice is enough. On (4, 8) the second axis grows too, so we have to
    slice both. This helper centralises the slicing so every call site
    is mesh-shape agnostic."""
    composer = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape, dims=(1, 0))
    raw = ttnn.to_torch(tensor_tt, mesh_composer=composer)
    shape = tuple(tensor_tt.shape)
    slicer = tuple(slice(0, s) for s in shape)
    return raw[slicer]


# Inlined from tt-lang-kernels/fp4_gemm.py. Used by the cached routed-expert
# path (Model.offload_moe_routed_experts -> MoE._forward_device_routed_cached).
# The cache stores fp4 e2m1 nibbles bit-cast as bfp4_b (lossless via the bfp4
# lattice trick); the algebraic remap below recovers fp4 magnitudes on device.
FP4_BLOCK_K = 32


def _remap_bfp4_lattice_to_fp4_mags(ttnn, b_tt):
    """f(b) = 2b + 2(relu(b-1) - relu(-b-1)) + 4(relu(b-1.5) - relu(-b-1.5))

    Maps the bfp4 lattice {0, ±0.25, ..., ±1.75} (what gets stored when we
    feed our lattice bf16 values to ttnn.from_torch(dtype=bfloat4_b)) back to
    fp4 e2m1 magnitudes {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Verified by hand
    for all 16 lattice values."""
    two_b = ttnn.multiply(b_tt, 2.0)
    pos1 = ttnn.relu(ttnn.subtract(b_tt, 1.0))
    neg1 = ttnn.relu(ttnn.subtract(ttnn.neg(b_tt), 1.0))
    step1 = ttnn.multiply(ttnn.subtract(pos1, neg1), 2.0)
    pos15 = ttnn.relu(ttnn.subtract(b_tt, 1.5))
    neg15 = ttnn.relu(ttnn.subtract(ttnn.neg(b_tt), 1.5))
    step15 = ttnn.multiply(ttnn.subtract(pos15, neg15), 4.0)
    return ttnn.add(ttnn.add(two_b, step1), step15)


def _dequant_to_native_bfp4(ttnn, w_bfp4_lattice_tt, scale_compact_tt,
                             *, bf16_scratch):
    """One-shot offload-time dequant: bfp4-lattice + compact-scale ->
    native bfp4_b weight tensor whose intrinsic per-tile exponent captures
    the actual weight magnitudes.

    Steps (all on device):
      1) typecast bfp4 -> bf16 (lossless; lattice values are bf16-exact)
      2) algebraic remap: bfp4 lattice -> fp4 e2m1 magnitudes
      3) repeat_interleave compact scale Kb -> K, multiply by remapped weights
      4) typecast bf16 -> bfp4_b (re-quantize with native scale)

    Run once per (layer, weight) at startup. The returned bfp4_b weight is
    drop-in for `ttnn.matmul(x_bf16, w_bfp4_b)` -- no per-call dequant.
    """
    ttnn.typecast(w_bfp4_lattice_tt, ttnn.bfloat16, output_tensor=bf16_scratch)
    w_remap = _remap_bfp4_lattice_to_fp4_mags(ttnn, bf16_scratch)
    scale_expanded = ttnn.repeat_interleave(
        scale_compact_tt, repeats=FP4_BLOCK_K, dim=-2)
    w_scaled = ttnn.multiply(w_remap, scale_expanded)
    ttnn.deallocate(w_remap)
    ttnn.deallocate(scale_expanded)
    w_native = ttnn.typecast(w_scaled, ttnn.bfloat4_b)
    ttnn.deallocate(w_scaled)
    return w_native


def _make_chip_local_ids_tt(ttnn, mesh, mesh_shape, n_experts: int):
    """[rows, cols, per_chip] int32 sharded over (rows, cols).

    chip(r, c) sees [per_chip] = [r*cols*per_chip + c*per_chip + i for i].
    Used by Path D selection mask: an indices == chip_local_ids broadcast
    comparison gives a per-chip [per_chip, topk] match matrix on each chip.
    """
    rows, cols = mesh_shape
    if n_experts % (rows * cols) != 0:
        raise ValueError(
            f"n_experts={n_experts} not divisible by mesh size {rows*cols}")
    per_chip = n_experts // (rows * cols)
    ids = torch.arange(n_experts, dtype=torch.int32).view(rows, cols, per_chip).contiguous()
    return ttnn.from_torch(
        ids,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (rows, cols), dims=(0, 1)),
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
        """Move hc_combiner + final RMSNorm + lm_head matmul to a 1xN mesh.

        Must be called after load_weights(). Shards the lm_head column-parallel
        across vocab; replicates the tiny final-norm weight, hc_fn/scale/base
        on each chip. ParallelHead.forward is rebound to take the residual
        a_tt device tensor directly (no CPU round-trip)."""
        head = self.transformer.head
        final_norm = self.transformer.norm
        transformer = self.transformer
        self._device_lm_head = DeviceLMHead(
            mesh, head.weight.data,
            norm_weight=final_norm.weight.data,
            norm_eps=final_norm.eps,
            hc_fn=transformer.hc_head_fn.data,
            hc_scale=transformer.hc_head_scale.data,
            hc_base=transformer.hc_head_base.data,
            hc_eps=transformer.hc_eps,
            hc_mult=transformer.hc_mult,
        )
        dlh = self._device_lm_head

        def _device_head_forward(self_head, a_tt, num_tokens):
            with _phase("head.norm_and_logits"):
                return dlh.forward_a_tt(a_tt, num_tokens)
        head.forward = _device_head_forward.__get__(head, type(head))
        final_norm.weight.data = torch.empty(0)  # device owns it
        head.weight.data = torch.empty(0)
        transformer.hc_head_fn.data = torch.empty(0)
        transformer.hc_head_scale.data = torch.empty(0)
        transformer.hc_head_base.data = torch.empty(0)

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
        """Move token embedding to the mesh, dim-sharded along the col axis.

        Replaces Transformer.embed.forward with a per-chip ttnn.embedding call
        on a hidden-axis-sharded weight, followed by an all_gather along
        cluster_axis=1. Per-chip table size is `[vocab, hidden/n_cols]`
        (~125MB on Galaxy vs ~1GB replicated). Output is replicated across the
        col axis after the gather, matching the previous calling contract.
        """
        import ttnn
        embed = self.transformer.embed
        w = _weight_to_bf16(embed.weight).contiguous()
        mesh_shape = tuple(mesh.shape)
        n_cols = mesh_shape[1]
        vocab, hidden = w.shape
        if hidden % n_cols != 0:
            raise ValueError(
                f"embedding hidden={hidden} not divisible by mesh cols={n_cols}"
            )
        w_tt = ttnn.as_tensor(
            w,
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh, mesh_shape=mesh_shape, dims=(None, -1)),
        )

        # Single shared per-step input_ids upload buffer. Lives on the
        # transformer so Transformer.forward and any hash-gate MoE layer
        # read from the same tensor (one host->device transfer per step).
        self.transformer._input_ids_tt = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32),
            device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.transformer._input_ids_upload_mapper = ttnn.ReplicateTensorToMesh(mesh)

        def _device_embed(self_embed, ids_tt):
            """Return a ttnn device tensor [B, S, hidden] bf16 TILE_LAYOUT,
            replicated across the mesh. ids_tt: [B, S] uint32 device tensor
            (pre-uploaded by the caller). No host transfers."""
            embed_local = ttnn.embedding(ids_tt, w_tt, layout=ttnn.TILE_LAYOUT)
            return ttnn.all_gather(embed_local, dim=-1, cluster_axis=1,
                                   num_links=1)

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

        Per layer (ordering invariant: offload_attn_full and
        offload_indexer_linears must run first):
          - attn._device_compressor: DeviceCompressor for attn.compressor.
            Bound to attn._device_attention.kv_cache_tt at first decode.
          - attn._device_indexer (when ratio=4): DeviceIndexer wrapping
            attn.indexer with its own DeviceCompressor (rotate=True).
        Layers with compress_ratio not in {0, 4} have a compressor but no
        indexer; the indexer wiring is skipped for those layers.
        """
        if not hasattr(self, "_device_attn_full") or not self._device_attn_full:
            raise RuntimeError(
                "offload_compressor_indexer must run after offload_attn_full"
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
                    "offload_compressor_indexer must run after "
                    "offload_indexer_linears"
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

    def offload_moe_routed_experts(self, mesh):
        """Attach cached bfp4_b routed-expert weights to every non-hash MoE
        layer, plus the per-chip expert-ID lookup used by Path D selection.

        Cache layout (produced by scripts/preprocess_routed_experts.py):
          {cache_dir}/layer_{L:03d}/{w1,w1_scale,w2,w2_scale,w3,w3_scale}.tensorbin

        Each .tensorbin holds a [rows, cols, per_chip, K, N] tensor sharded
        with ShardTensor2dMesh(dims=(0, 1)) for the (rows, cols) mesh, so
        ttnn.load_tensor returns a 5D device tensor that's already laid out
        for grouped-expert matmul.

        Hash-gate layers (layer_id < args.n_hash_layers) are skipped: the
        preprocess does not cover them, so they keep the host fallback in
        MoE.forward_device.

        Cache directory comes from $DS_ROUTED_EXPERT_CACHE; defaults to
        /home/ubuntu/hf/state_dict_bfp4_routed.
        """
        import ttnn
        cache_dir = os.environ.get(
            "DS_ROUTED_EXPERT_CACHE", "/home/ubuntu/hf/state_dict_bfp4_routed")
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            raise FileNotFoundError(
                f"routed-expert cache dir not found: {cache_dir}. Run "
                f"scripts/preprocess_routed_experts.py to populate it, or "
                f"set $DS_ROUTED_EXPERT_CACHE to its location.")
        mesh_shape = tuple(mesh.shape)
        rows, cols = mesh_shape
        n_chips = rows * cols
        self._device_routed_experts = []
        for layer in self.transformer.layers:
            ffn = layer.ffn
            if not isinstance(ffn, MoE):
                continue
            n_per_chip = ffn.n_routed_experts // n_chips
            if ffn.n_routed_experts % n_chips != 0:
                raise ValueError(
                    f"layer {layer.layer_id}: n_routed_experts="
                    f"{ffn.n_routed_experts} not divisible by mesh size "
                    f"{n_chips}")
            layer_dir = cache_dir / f"layer_{layer.layer_id:03d}"
            paths = {
                name: layer_dir / f"{name}.tensorbin"
                for name in ("w1", "w1_scale", "w2", "w2_scale",
                             "w3", "w3_scale")
            }
            for name, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(
                        f"layer {layer.layer_id}: missing {path}")
            # Cache was dumped as rank 5 [rows, cols, per_chip, K, N]; the
            # loader's per-chip view depends on TTNN's sharded-load convention
            # (may be rank 5 with sharded dims kept as 1, or rank 3 with
            # sharded dims dropped). Reshape to rank 4 [1, per_chip, K, N]
            # so downstream ops (which max out at rank 4) are happy.
            dim = ffn.dim
            # Expert.w1.weight is [inter, dim] (out, in); shape[0] is inter.
            # Read it before the post-load cleanup loop zeroes the CPU weights.
            inter = ffn.experts[0].w1.weight.shape[0]
            kb_w1 = dim // FP4_BLOCK_K
            kb_w2 = inter // FP4_BLOCK_K

            def _load_w(ttnn_path, K, N):
                t = ttnn.load_tensor(str(ttnn_path), device=mesh)
                return ttnn.reshape(t, [1, n_per_chip, K, N])

            def _load_s(ttnn_path, Kb, N):
                t = ttnn.load_tensor(str(ttnn_path), device=mesh)
                return ttnn.reshape(t, [1, n_per_chip, Kb, N])

            # Per-(layer, weight) reusable bf16 scratch for the offload-time
            # dequant of bfp4-lattice -> bf16 -> native bfp4_b. Scratch is the
            # full sharded weight shape; freed after this layer's three weights
            # are converted.
            w13_scratch = ttnn.from_torch(
                torch.zeros([1, n_per_chip, dim, inter], dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            w2_scratch = ttnn.from_torch(
                torch.zeros([1, n_per_chip, inter, dim], dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

            def _convert(name_w, name_s, K, N, Kb, scratch):
                w_lat = _load_w(paths[name_w], K, N)
                s_compact = _load_s(paths[name_s], Kb, N)
                w_native = _dequant_to_native_bfp4(
                    ttnn, w_lat, s_compact, bf16_scratch=scratch)
                ttnn.deallocate(w_lat)
                ttnn.deallocate(s_compact)
                return w_native

            ffn._w1_tt = _convert("w1", "w1_scale", dim, inter, kb_w1, w13_scratch)
            ffn._w3_tt = _convert("w3", "w3_scale", dim, inter, kb_w1, w13_scratch)
            ffn._w2_tt = _convert("w2", "w2_scale", inter, dim, kb_w2, w2_scratch)
            ttnn.deallocate(w13_scratch)
            ttnn.deallocate(w2_scratch)

            ffn._chip_local_ids_tt = _make_chip_local_ids_tt(
                ttnn, mesh, mesh_shape, ffn.n_routed_experts)
            # Pre-shape chip_ids to [1, per_chip, 1, 1]: matches the per_chip
            # axis of the grouped MLP output (dim=1), so the per-call selection
            # mask comes out aligned without a reshape.
            ffn._chip_ids_4d_tt = ttnn.reshape(
                ffn._chip_local_ids_tt, [1, n_per_chip, 1, 1])
            ffn._n_per_chip = n_per_chip

            # Per-MoE pre-allocated matmul output scratches. Tiny
            # (~32-64KB per chip per scratch); replicated, written each call.
            def _alloc_y_scratch(N):
                return ttnn.from_torch(
                    torch.zeros([1, n_per_chip, 1, N], dtype=torch.bfloat16),
                    device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            ffn._fp4_y1_scratch_tt = _alloc_y_scratch(inter)
            ffn._fp4_y3_scratch_tt = _alloc_y_scratch(inter)
            ffn._fp4_y2_scratch_tt = _alloc_y_scratch(dim)

            for expert in ffn.experts:
                expert.w1.weight.data = torch.empty(0)
                expert.w2.weight.data = torch.empty(0)
                expert.w3.weight.data = torch.empty(0)
                if hasattr(expert.w1, "scale"):
                    expert.w1.scale.data = torch.empty(0)
                    expert.w2.scale.data = torch.empty(0)
                    expert.w3.scale.data = torch.empty(0)
            self._device_routed_experts.append(ffn)

    def offload_moe_gate(self, mesh):
        """Move every MoE gate to the 1xN mesh (hash + non-hash).

        Gate is tiny (n_experts x dim bf16, ~2MB) — replicated on every chip.
        Hash gates additionally hold a [vocab, topk] tid2eid lookup (~3MB)
        for input_ids → expert-ids on device.
        """
        self._device_moe_gates = []
        for layer in self.transformer.layers:
            ffn = layer.ffn
            if not isinstance(ffn, MoE):
                continue
            gate = ffn.gate
            if gate.hash:
                dg = HashDeviceMoEGate(
                    mesh, gate.weight, gate.tid2eid,
                    topk=gate.topk, route_scale=gate.route_scale,
                    score_func=gate.score_func,
                    n_routed_experts=ffn.n_routed_experts,
                )
                gate.tid2eid.data = torch.empty(0)  # device owns it
                gate._device_gate = dg
            else:
                dg = DeviceMoEGate(
                    mesh, gate.weight, gate.bias,
                    topk=gate.topk, route_scale=gate.route_scale,
                    score_func=gate.score_func,
                )

                def _device_gate_forward(self_gate, x, input_ids=None, _dg=dg):
                    return _dg(x)

                gate.forward = _device_gate_forward.__get__(gate, type(gate))
                gate._device_gate = dg
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
                        f"offload_attn_full must run after "
                        f"offload_attn_{w_attr}"
                    )
            if not hasattr(self, "_device_sparse_attn"):
                raise RuntimeError(
                    "offload_attn_full must run after offload_sparse_attn"
                )
            if not hasattr(self, "_device_rms_norms"):
                raise RuntimeError(
                    "offload_attn_full must run after offload_rms_norms "
                    "(q_norm / kv_norm)"
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
            layer._device_mhc_attn = mhc_attn
            layer._device_mhc_ffn = mhc_ffn
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
    def generate(self, tokens: list[int], max_tokens: int = 32,
                 temperature: float = 1.0, warmup_tokens: int = 0):
        """Generate up to max_tokens. Phase counters are reset after the
        first `warmup_tokens` decode steps so steady-state timings don't
        include first-call JIT compilation / trace capture."""
        with _phase("prefill"):
            logits = self.prefill(tokens)
        pos = len(tokens)
        for i in range(max_tokens):
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
            if warmup_tokens > 0 and i + 1 == warmup_tokens:
                _PHASE_ACCUM.clear()
                _PHASE_COUNTS.clear()
                print(f"[phase] warmup complete after {warmup_tokens} tokens; "
                      f"phase counters reset")


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
    parser.add_argument("--warmup-tokens", type=int, default=0,
                        help="Reset phase counters after this many decode steps "
                             "so steady-state timings exclude first-call JIT / "
                             "trace capture overhead.")
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

    if args.weights_only:
        print("[done] --weights-only set; stopping before mesh open")
        return

    print("[phase] opening 4x8 mesh ...")
    t0 = time.time()
    mesh = _open_mesh()
    print(f"[phase] mesh opened in {time.time() - t0:.1f}s")

    import ttnn
    _set_phase_sync(lambda: ttnn.synchronize_device(mesh))

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
        ("moe_routed_experts", model.offload_moe_routed_experts),
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

    print("[phase] prebuilding tt-lang decode kernels ...")
    t0 = time.time()
    prebuild_ttl_decode_kernels(model.args)
    print(f"[phase] prebuilt {len(_TTL_KERNEL_CACHE)} tt-lang kernels in "
          f"{time.time() - t0:.1f}s")

    n_alloc_modules = sum(1 for _ in model._iter_device_modules())
    print(f"[phase] {n_alloc_modules} device modules registered for "
          f"Model.allocate_decode_tensors")

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
    for tok in model.generate(tokens, max_tokens=args.max_tokens,
                              temperature=args.temperature,
                              warmup_tokens=args.warmup_tokens):
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
