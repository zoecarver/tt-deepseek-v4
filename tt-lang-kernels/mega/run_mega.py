"""Mega-kernel inference for DeepSeek-V4-Flash.

This script orchestrates the per-step decode using the fused tt-lang
mega-kernels in `tt-lang-kernels/mega/test_*.py`. It is the production
counterpart to `../inference.py` — same model, same boilerplate (Model
class, weight loader, mesh open, KV cache layout, trace capture), but the
hot path is a flat sequence of `make_*_kernel(...)` callables instead of
the legacy DeviceMHC/DeviceAttention/DeviceMoE Python orchestration.

Design rules:

1. **Kernels are imported, never inlined.** Every `ttl.operation` lives in
   a `test_*.py` file next to its standalone PCC test. This file is the
   thin orchestrator. If you find yourself writing tt-lang code here, stop
   — move it into a `test_*.py` and import it.

2. **Boilerplate (weight loading, mesh, KV cache, traces) is reused from
   ../inference.py.** We import `Model`, `ModelArgs`, `ParallelEmbedding`,
   `Block`, etc. and subclass `Model` so we get HF safetensors loading,
   tokenizer, and trace state for free.

3. **L1 placement for residuals.** The Quiet Box / Galaxy has ~1.5MB L1
   per core × ~110 cores per chip = ~165 MB usable per chip, ~5 GB across
   the 32-chip Galaxy. The hot residual stream `a_tt`
   (32 × 16384 bf16 = 1 MB total) and the per-layer interconnect tensors
   (`x_tt`, `q_tt`, `kv_tt`, `wq_a_partial`, etc.) all fit easily. We
   allocate them with `ttnn.L1_MEMORY_CONFIG` so kernel<->kernel handoff
   inside one decode step never round-trips through DRAM. KV caches and
   weights stay in DRAM (too big).

4. **One kernel per zone.** The mega kernels already span CCL boundaries.
   Step decode = a flat `for layer in layers: lk_a; lk_b; lk_c; lk_d1;
   lk_dsparse; lk_e; lk_f` loop. Anything that isn't a kernel call is
   pre-step host upload (start_pos, input_ids) or a CCL.

What still talks to the legacy `Model`:

- `Model.load_weights(...)` from ../inference.py — populates the
  state_dict from HF safetensors. We read tensors out of that state_dict
  to feed the kernel factories' constant args (gamma, hc_fn, hc_scale,
  hc_base, gate_bias, cos/sin tables, weight tensors).

- `Model.from_hf(...)` — constructs the `ModelArgs` from
  `inference/config.json` so we share the exact decode-config knobs
  (n_layers, dim, hc_mult, etc.).

- The KV cache layout (sequence-sharded on the col axis), trace state
  (no-emit / emit branches), pre_stage_decode (per-step host uploads)
  are all from `../inference.py`'s Transformer + DeviceAttention. The
  mega step calls into those for the decode-step machinery surrounding
  the kernels.

What does NOT come from ../inference.py:

- The hot-path `_block_forward` body. We replace that wholesale with a
  mega-kernel sequence (`MegaModel._block_forward_mega`).
- DeviceMHC / DeviceRMSNorm / DeviceColLinear dispatches in the inner
  decode body. The mega kernels absorb all of them.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Optional

# `inference` lives one dir up locally; on the remote sandbox the file is
# colocated under /tmp. Mirror `_refs.py`'s sys.path bootstrap so we work
# in both environments without env-var ceremony.
_HERE = pathlib.Path(__file__).resolve().parent
for _candidate in (_HERE, *_HERE.parents):
    if (_candidate / "inference.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import torch  # noqa: E402
import ttnn   # noqa: E402

# ---------------------------------------------------------------------------
# Kernel factories: one per mega-zone. Each returns a callable that takes
# per-step ttnn tensors and writes its output into a pre-allocated buffer.
# Constant CPU tensors (gamma, cos/sin tables, hc_fn, gate_bias, etc.) are
# baked into the factory closure at construction time.
# ---------------------------------------------------------------------------
from test_l0 import make_l0_kernel                 # noqa: E402
from test_lk_a import make_lk_a_kernel             # noqa: E402
from test_lk_b import make_lk_b_kernel             # noqa: E402
from test_lk_c import make_lk_c_kernel             # noqa: E402
from test_lk_d1 import make_lk_d1_kernel           # noqa: E402
from test_lk_d_idx_q import make_lk_d_idx_q_kernel # noqa: E402
from test_lk_d_idx_cmp import make_lk_d_idx_cmp_kernel  # noqa: E402
from test_lk_d_idx_emit import make_lk_d_idx_emit_kernel  # noqa: E402
from test_lk_d_idx_score import make_lk_d_idx_score_kernel  # noqa: E402
from test_lk_d_topk import make_lk_d_topk_kernel   # noqa: E402
from test_lk_d_comp import make_lk_d_comp_kernel   # noqa: E402
from test_lk_dsparse import make_lk_dsparse_kernel # noqa: E402
from test_lk_e import make_lk_e_kernel             # noqa: E402
from test_lk_f import make_lk_f_kernel             # noqa: E402
from test_final import (                            # noqa: E402
    make_final_kernel,
    _pack_hc_fn_t_scaled, _pack_hc_base_tile,
)

# ---------------------------------------------------------------------------
# Boilerplate from the legacy single-file inference. We pull out only what
# the mega path actually needs: model wiring, weight loader, mesh open,
# tokenizer, ModelArgs.
# ---------------------------------------------------------------------------
from inference import (                            # noqa: E402
    # Model subclassed for from_hf + HF safetensors loader. No legacy
    # offload_* / step_decode is called: every override lives below.
    Model as _BaseModel,
    ModelArgs,
    Transformer,
    Block,
    MoE,
    DeviceCompressor,
    DeviceIndexer,
    DeviceColLinear,
    DeviceLMHead,
    DeviceMHC,
    DeviceRMSNorm,
    _mhc_post_to_a_tt,
    _build_window_topk_table,
    _window_topk_row_for_pos,
    _open_mesh,
    _close_mesh,
    _phase, _phase_report, _phase_postwarm,
    _phase_snapshot_at_trace_warm,
    _PHASE_ACCUM, _PHASE_COUNTS,
    _PHASE_PREWARM_ACCUM, _PHASE_PREWARM_COUNTS,
    _MHC_TILE,
    _sylvester_hadamard,
)

# Module-level constants used by the per-step buffer / Final-head packing
# helpers. The mega kernels pin these exactly (DIM, MHC, etc. are baked into
# `test_lk_*.py`); we mirror them here so allocation matches the kernel
# expected dtype/shape.
TILE = 32
DIM = 4096           # hidden dim
MHC = 4              # hc_mult
D = MHC * DIM        # 16384
NORM_EPS = 1e-6
HC_EPS = 1e-6
VOCAB = 129280
N_HEADS = 64

# Indexer / compressor pipeline (ratio=4 layers).
INDEX_HEAD_DIM = 128
INDEX_N_HEADS = 64
INDEX_RATIO = 4
INDEX_TOPK_BUCKET = 128  # smallest bucket the topk kernel was built for
INDEX_T_PAD = 128        # indexer kv_cache slot count after padding


# ---------------------------------------------------------------------------
# L1 helpers. Kernel<->kernel handoff buffers live in L1; weights and KV
# caches stay in DRAM. Sharding follows the existing inference pattern
# (`ReplicateTensorToMesh` for replicated, `ShardTensor2dMesh` for col-shard).
# ---------------------------------------------------------------------------

L1 = "l1"
DRAM = "dram"


def _mem_cfg(ttnn, kind: str):
    if kind == L1:
        return ttnn.L1_MEMORY_CONFIG
    if kind == DRAM:
        return ttnn.DRAM_MEMORY_CONFIG
    raise ValueError(f"unknown memory kind {kind!r}")


def alloc_replicated(mesh, shape, dtype, *, mem=L1, fill=None):
    """Allocate a replicated ttnn tensor, defaulting to L1.

    Hot-path interconnect tensors (residual `a_tt`, per-layer Q/KV pieces,
    wq_a_partial, etc.) all fit comfortably in L1 once sharded across the
    mesh — keep them out of DRAM so kernel<->kernel handoff is L1-local.
    """
    if fill is None:
        fill = torch.zeros(*shape, dtype=torch.bfloat16)
    return ttnn.from_torch(
        fill.contiguous(),
        device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
        memory_config=_mem_cfg(ttnn, mem),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def alloc_col_shard(mesh, shape, dtype, *, dim: int, mem=L1):
    """Allocate a tensor col-sharded along `dim`. Default L1.

    Used for per-layer linear outputs that are N-sharded on the col axis
    (wq_a_partial, wkv_partial, wo_b_partial). Pre-allgather they're tiny
    per chip; post-allgather they're replicated and we copy to a separate
    replicated L1 buffer for downstream consumers.
    """
    fill = torch.zeros(*shape, dtype=torch.bfloat16)
    mesh_shape = tuple(mesh.shape)
    # ShardTensor2dMesh `dims=(rows_dim, cols_dim)` — None means replicate
    # on that axis. We col-shard so dims=(None, dim).
    mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape, dims=(None, dim))
    return ttnn.from_torch(
        fill.contiguous(), device=mesh, dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_mem_cfg(ttnn, mem),
        mesh_mapper=mapper,
    )


# ---------------------------------------------------------------------------
# Per-layer kernel bundle. Each layer in the model gets one of these,
# carrying the closures that own this layer's compiled tt-lang programs
# and constant tensors. Built once at offload time; called every decode
# step.
# ---------------------------------------------------------------------------


class LayerKernels:
    """Holds the bound mega-kernel callables for one transformer layer.

    Each field is a callable returned by the corresponding `make_*_kernel`
    factory in `mega/test_*.py`. Factories are called once per layer at
    offload time, so each callable closes over that layer's CPU constants
    (gamma vectors, hc_fn/scale/base, etc.) and pre-uploaded device weight
    tensors.

    None entries indicate the kernel is not applicable to this layer
    (e.g. `lk_d_idx_*` kernels only on indexer layers, `lk_d_comp` only
    on compressor layers).
    """

    __slots__ = (
        "layer_idx",
        "lk_a", "lk_b", "lk_c", "lk_d1",
        "lk_d_idx_q", "lk_d_idx_cmp", "lk_d_idx_emit",
        "lk_d_idx_score", "lk_d_topk",
        "lk_d_comp",
        "lk_dsparse", "lk_e", "lk_f",
        "mhc_ffn", "attn_norm",
        "weights",
    )

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.lk_a = None
        self.lk_b = None
        self.lk_c = None
        self.lk_d1 = None
        self.lk_d_idx_q = None
        self.lk_d_idx_cmp = None
        self.lk_d_idx_emit = None
        self.lk_d_idx_score = None
        self.lk_d_topk = None
        self.lk_d_comp = None
        self.lk_dsparse = None
        self.lk_e = None
        self.lk_f = None
        # Legacy DeviceMHC for ffn-side hc_post bridge. The mega kernels
        # (Lk-E + Lk-F) compute the FFN partial but do not apply
        # hc_post_ffn; the residual stream needs the MHC-weighted combine
        # `out = x*post_mix + comb^T @ residual` to match the reference.
        # Until a proper Lk-G kernel exists, we run hc_pre_ffn (small)
        # before the FFN body to populate the stash and hc_post_ffn after
        # to assemble the next layer's a_tt.
        self.mhc_ffn = None
        # Legacy DeviceRMSNorm for the attn-side bridge (Lk-A bakes gamma
        # into wq_a so it never materializes the post-norm tile, but Lk-C
        # needs it). Using the kernel-based norm (matches legacy exactly)
        # rather than `ttnn.rms_norm`, which gave numerically different
        # output during bringup.
        self.attn_norm = None
        # Per-layer device weight tensors uploaded once at offload time.
        # Kept here (not on the closures) so we can also free them on
        # tear-down without poking inside the kernel state.
        self.weights: dict = {}


def _replicate_weight(mesh, t: torch.Tensor, dtype):
    """Upload a 2D weight as fully replicated (DRAM). For incremental
    bringup; the future_max_utilization plan replaces these with 2D
    row-K + col-N shard via `ShardTensor2dMesh`."""
    return ttnn.as_tensor(
        t.contiguous().to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.contiguous(),
        device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _build_layer_kernels(mesh, layer: Block, layer_idx: int,
                         hc_attn_fn: torch.Tensor,
                         hc_attn_scale: torch.Tensor,
                         hc_attn_base: torch.Tensor,
                         hc_ffn_fn: torch.Tensor,
                         hc_ffn_scale: torch.Tensor,
                         hc_ffn_base: torch.Tensor,
                         cos_full_cpu: torch.Tensor,
                         sin_full_cpu: torch.Tensor,
                         cos_compressor_cpu: Optional[torch.Tensor],
                         sin_compressor_cpu: Optional[torch.Tensor],
                         args: ModelArgs) -> LayerKernels:
    """Compile every mega-kernel callable that this layer needs.

    Each layer holds its own copy of compiled tt-lang programs because
    the constant CPU tensors (hc_fn, gamma) differ per layer. The kernel
    cache in tt-lang dedupes identical programs by structure, so 43
    layers don't pay 43× compile cost — just 43× factory closure setup.
    """
    lk = LayerKernels(layer_idx)
    attn = layer.attn

    # --- Lk-A: hc_pre_attn + attn_norm + wq_a -----------------------------
    lk.lk_a = make_lk_a_kernel(
        mesh,
        hc_fn_cpu=hc_attn_fn,
        hc_scale_cpu=hc_attn_scale,
        hc_base_cpu=hc_attn_base,
        gamma_cpu=layer.attn_norm.weight.data,
    )
    # Lk-A expects wq_a in [DIM=in, Q_LORA_RANK=out] orientation with
    # gamma pre-baked into the K (DIM) axis: Wg[k, n] = gamma[k] * W[k, n].
    # nn.Linear stores it as [out, in] so we transpose first.
    _wq_a_T = attn.wq_a.weight.data.transpose(0, 1).contiguous()
    _gamma_attn = layer.attn_norm.weight.data
    _wq_a_baked = (_gamma_attn.float()[:, None] * _wq_a_T.float()) \
        .to(torch.bfloat16).contiguous()
    lk.weights["wq_a"] = ttnn.as_tensor(
        _wq_a_baked, device=mesh, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    # Lk-A bakes gamma into wq_a, so the bf16 post-attn-norm tile that Lk-C
    # consumes is never materialized inside the kernel. The orchestrator
    # runs a small ttnn.rms_norm device op on Lk-A's published pre-norm
    # output using this gamma to produce Lk-C's `x_for_wkv` bridge.
    lk.weights["attn_norm_gamma"] = ttnn.as_tensor(
        layer.attn_norm.weight.data.contiguous().to(torch.bfloat16),
        device=mesh, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # --- Lk-B: q_norm + wq_b (gamma baked into wq_b on host) --------------
    # Factory expects wq_b in [Q_LORA_RANK=in, N=out] orientation; transpose
    # from nn.Linear's [out, in].
    lk.lk_b = make_lk_b_kernel(
        mesh,
        gamma_cpu=attn.q_norm.weight.data,
        wq_b_cpu=attn.wq_b.weight.data.transpose(0, 1).contiguous(),
    )

    # --- Lk-C: q_rsqrt_norm + q rotary + wkv ------------------------------
    # wkv expected as [DIM=in, HEAD_DIM=out]; transpose from nn.Linear's [out, in].
    lk.lk_c = make_lk_c_kernel(mesh, cos_full_cpu, sin_full_cpu)
    _wkv_T = attn.wkv.weight.data.transpose(0, 1).contiguous()
    lk.weights["wkv"] = _replicate_weight(mesh, _wkv_T, ttnn.bfloat16)

    # --- Lk-D1: kv_norm + kv rotary + act_quant_block ---------------------
    lk.lk_d1 = make_lk_d1_kernel(
        mesh,
        gamma_cpu=attn.kv_norm.weight.data,
        cos_full_cpu=cos_full_cpu,
        sin_full_cpu=sin_full_cpu,
    )

    # --- Lk-D-idx-*: only on indexer layers (compress_ratio == 4) --------
    is_indexer = (
        getattr(attn, "compressor", None) is not None
        and attn.compressor.compress_ratio == 4
        and getattr(attn, "indexer", None) is not None
    )
    if is_indexer:
        ix = attn.indexer
        # Upload replicated weight tensors used by the mega kernels. The
        # legacy DeviceColLinear path col-shards these on N which would
        # collide with the kernel's internal SUMMA tiling, so we keep our
        # own replicated copies.
        wq_b_io = ix.wq_b.weight.data.to(torch.bfloat16).transpose(0, 1).contiguous()
        lk.weights["indexer_wq_b"] = ttnn.as_tensor(
            wq_b_io,
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # cat([wkv, wgate], dim=-1) for the fused indexer-cmp matmul.
        ix_wkv_io = ix.compressor.wkv.weight.data.to(torch.bfloat16).transpose(0, 1)
        ix_wgate_io = ix.compressor.wgate.weight.data.to(torch.bfloat16).transpose(0, 1)
        ix_wkv_gate_cpu = torch.cat([ix_wkv_io, ix_wgate_io], dim=-1).contiguous()
        lk.weights["indexer_wkv_gate"] = ttnn.as_tensor(
            ix_wkv_gate_cpu,
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # weights_proj baked with softmax_scale * (H ** -0.5) so the kernel
        # matmul folds in the score scale.
        scale = float(ix.softmax_scale) * (INDEX_N_HEADS ** -0.5)
        wproj_io = ix.weights_proj.weight.data.to(torch.bfloat16).transpose(0, 1)
        lk.weights["indexer_wproj_scaled"] = ttnn.as_tensor(
            (wproj_io * scale).contiguous(),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Walsh-Hadamard rotation matrix for q.
        H_idx = (_sylvester_hadamard(INDEX_HEAD_DIM) *
                 (INDEX_HEAD_DIM ** -0.5)).to(torch.bfloat16)
        lk.weights["indexer_H"] = ttnn.as_tensor(
            H_idx.contiguous(),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        # Shared height-sharded L1 memcfg for paged_update_cache calls.
        # Indexer state buffers and indexer kv_cache all have last dim
        # INDEX_HEAD_DIM = 128.
        sharded_input_memcfg_idx = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                (32, INDEX_HEAD_DIM),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        lk.lk_d_idx_q = make_lk_d_idx_q_kernel(mesh, cos_full_cpu, sin_full_cpu)
        lk.lk_d_idx_cmp = make_lk_d_idx_cmp_kernel(
            mesh, sharded_input_memcfg=sharded_input_memcfg_idx)
        lk.lk_d_idx_emit = make_lk_d_idx_emit_kernel(
            mesh,
            cos_compressor_cpu=cos_compressor_cpu,
            sin_compressor_cpu=sin_compressor_cpu,
            sharded_input_memcfg=sharded_input_memcfg_idx,
        )
        lk.lk_d_idx_score = make_lk_d_idx_score_kernel(mesh)
        lk.lk_d_topk = make_lk_d_topk_kernel(mesh)

    # --- Lk-D-comp: attn-side compressor (ratio=4 layers only) -----------
    # The Lk-D-comp kernel hardcodes RATIO=4, HEAD_DIM=512. ratio=128 layers
    # need a different kernel; for bringup we leave them with stale state
    # (their slots aren't reached by the window-only topk fallback either).
    if getattr(attn, "compressor", None) is not None and attn.compress_ratio == 4:
        comp_head_dim = attn.compressor.head_dim   # 512
        # wkv / wgate replicated; passed separately.
        wkv_io = attn.compressor.wkv.weight.data.to(torch.bfloat16).transpose(0, 1).contiguous()
        wgate_io = attn.compressor.wgate.weight.data.to(torch.bfloat16).transpose(0, 1).contiguous()
        lk.weights["comp_wkv"] = ttnn.as_tensor(
            wkv_io, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        lk.weights["comp_wgate"] = ttnn.as_tensor(
            wgate_io, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        sharded_input_memcfg_comp = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                (32, comp_head_dim),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        lk.lk_d_comp = make_lk_d_comp_kernel(
            mesh,
            cos_compressor_cpu=cos_compressor_cpu,
            sin_compressor_cpu=sin_compressor_cpu,
            sharded_input_memcfg=sharded_input_memcfg_comp,
        )

    # --- Lk-Dsparse: full sparse_attn body + wo_a + wo_b -----------------
    softmax_scale = float(attn.softmax_scale)
    # Lk-Dsparse height-shards the [TILE, HEAD_DIM] kv input on a single
    # L1 core for paged_update_cache. Mirrors the test's canonical shard.
    sharded_memcfg_dsparse = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            (32, args.head_dim),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    lk.lk_dsparse = make_lk_dsparse_kernel(
        mesh,
        cos_full_cpu=cos_full_cpu,
        sin_full_cpu=sin_full_cpu,
        sharded_input_memcfg=sharded_memcfg_dsparse,
        softmax_scale=softmax_scale,
    )
    # wo_a: nn.Linear weight is [n_groups*o_lora_rank, in_per_group].
    # Lk-Dsparse expects [n_groups, in_per_group, o_lora_rank] (3D, block-diag).
    _n_groups = attn.n_local_groups
    _o_lora_rank = attn.o_lora_rank
    _wo_a_cpu = attn.wo_a.weight.data
    _out_total, _in_per_group = _wo_a_cpu.shape
    if _out_total != _n_groups * _o_lora_rank:
        raise ValueError(
            f"wo_a shape {tuple(_wo_a_cpu.shape)} != expected "
            f"({_n_groups * _o_lora_rank}, in_per_group)")
    _wo_a_3d = _wo_a_cpu.to(torch.bfloat16).view(
        _n_groups, _o_lora_rank, _in_per_group).transpose(1, 2).contiguous()
    lk.weights["wo_a"] = ttnn.as_tensor(
        _wo_a_3d, device=mesh, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    # wo_b: nn.Linear weight is [DIM, n_groups*o_lora_rank]. Lk-Dsparse
    # expects [n_groups*o_lora_rank=in, DIM=out]; transpose.
    _wo_b_T = attn.wo_b.weight.data.transpose(0, 1).contiguous()
    lk.weights["wo_b"] = _replicate_weight(mesh, _wo_b_T, ttnn.bfloat16)

    # sink_padded: [N_HEADS, TILE] bf16. Col 0 = per-head attn_sink, cols
    # 1..TILE-1 = -1e9 sentinel. Lk-Dsparse's softmax_with_sink reads col 0
    # to concat the sink as the BIG-th column of `scores_masked_padded`.
    sink_padded_cpu = torch.full(
        (N_HEADS, TILE), -1.0e9, dtype=torch.bfloat16)
    sink_padded_cpu[:, 0] = attn.attn_sink.data.to(torch.bfloat16)
    lk.weights["sink_padded"] = ttnn.as_tensor(
        sink_padded_cpu.contiguous(),
        device=mesh, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sink_4d_cpu = attn.attn_sink.data.to(torch.bfloat16).view(
        1, 1, N_HEADS, 1).contiguous()
    lk.weights["sink_4d"] = ttnn.as_tensor(
        sink_4d_cpu, device=mesh, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # --- Lk-E: hc_post_attn + ffn_norm + shared expert + (optional) gate -
    if isinstance(layer.ffn, MoE):
        ffn = layer.ffn
        shared = ffn.shared_experts
        gate_bias_cpu = ffn.gate.bias.data if ffn.gate.bias is not None else torch.zeros(args.n_routed_experts)
    else:
        shared = layer.ffn          # plain SwiGLU FFN; no MoE
        gate_bias_cpu = None

    lk.lk_e = make_lk_e_kernel(
        mesh,
        hc_attn_fn_cpu=hc_attn_fn,
        hc_attn_scale_cpu=hc_attn_scale,
        hc_attn_base_cpu=hc_attn_base,
        hc_ffn_fn_cpu=hc_ffn_fn,
        hc_ffn_scale_cpu=hc_ffn_scale,
        hc_ffn_base_cpu=hc_ffn_base,
        ffn_norm_gamma_cpu=layer.ffn_norm.weight.data,
    )
    # Lk-E SUMMA matmuls expect [K=in, N=out]; nn.Linear stores [out, in].
    lk.weights["w1_shared"] = _replicate_weight(
        mesh, shared.w1.weight.data.transpose(0, 1).contiguous(), ttnn.bfloat16)
    lk.weights["w2_shared"] = _replicate_weight(
        mesh, shared.w2.weight.data.transpose(0, 1).contiguous(), ttnn.bfloat16)
    lk.weights["w3_shared"] = _replicate_weight(
        mesh, shared.w3.weight.data.transpose(0, 1).contiguous(), ttnn.bfloat16)

    # --- Lk-F: gate post + routed experts (MoE layers only) --------------
    if isinstance(layer.ffn, MoE):
        ffn = layer.ffn
        lk.lk_f = make_lk_f_kernel(mesh, gate_bias_cpu=gate_bias_cpu)
        # gate matmul expects [DIM=in, N_ROUTED=out]; transpose nn.Linear weight.
        lk.weights["gate_w"] = _replicate_weight(
            mesh, ffn.gate.weight.data.transpose(0, 1).contiguous(), ttnn.bfloat16)
        # gate_bias is baked into the lk_f closure above (factory pads it
        # to [TILE, N_ROUTED] internally), but the callable still takes a
        # gate_bias_tt arg for parity with the test harness — pass the
        # same replicated bias.
        lk.weights["gate_bias"] = _replicate_weight(
            mesh, gate_bias_cpu.unsqueeze(0).contiguous(), ttnn.bfloat16)
        # TODO: mega — routed expert weights w1/w2/w3 (256 experts ×
        # 2048×4096 each ≈ 8 GB total) must be sharded per-chip with
        # PER_CHIP=8 experts/chip on the (4, 8) mesh. Wire via
        # `_offload_routed_experts` from the legacy code path.

    # --- mhc_ffn: legacy DeviceMHC for hc_post_ffn bridge ----------------
    # Lk-E only does hc_post_attn → hc_pre_ffn → ffn_norm → shared expert.
    # The full hc_post_ffn (`out = x*post_mix + comb^T @ residual`) is not
    # in any mega kernel today; without it the FFN partial gets uniformly
    # broadcast across MHC heads instead of the proper weighted combine
    # and decode produces gibberish. Bringup workaround: run hc_pre_ffn +
    # hc_post_ffn via the legacy DeviceMHC. TODO: mega — fold into Lk-F or
    # add a dedicated Lk-G kernel.
    lk.mhc_ffn = DeviceMHC(
        mesh=mesh,
        hc_fn=hc_ffn_fn, hc_scale=hc_ffn_scale, hc_base=hc_ffn_base,
        hc_mult=args.hc_mult, hc_eps=args.hc_eps,
        sinkhorn_iters=args.hc_sinkhorn_iters, norm_eps=args.norm_eps,
    )

    # Legacy DeviceRMSNorm for the attn-norm bridge (Lk-A bakes gamma
    # into wq_a, so the post-norm tile must be reconstructed for Lk-C).
    lk.attn_norm = DeviceRMSNorm(
        mesh=mesh, cpu_gamma=layer.attn_norm.weight.data,
        eps=args.norm_eps,
    )

    return lk


# ---------------------------------------------------------------------------
# Per-step output buffer pool. The mega step uses a small set of stable
# device tensors as kernel<->kernel handoff slots; allocating them once
# (in L1) is a precondition for trace capture.
# ---------------------------------------------------------------------------


class StepBuffers:
    """Pre-allocated per-step ttnn tensors shared across layers.

    Every kernel writes its output into one of these buffers; the next
    kernel reads from it. Allocating once + reusing means the trace body
    is pure compute and CCL — no allocations at replay.

    Sizes (V4-Flash, M=1 decode):
      a_tt:           [Mpad=32, D=16384] bf16, 1 MB total — residual stream
      x_tt:           [1, 1, dim=4096] bf16, 8 KB — post-attn-norm
      wq_a_partial:   [1, 1, q_lora_rank=1024] bf16, 2 KB
      q_full:         [1, 1, n_heads*head_dim=32768] bf16, 64 KB
      q_rotated:      [1, 1, n_heads, head_dim] bf16, 64 KB
      kv_partial:     [1, 1, head_dim=512] bf16, 1 KB
      kv_rotated:     [1, 1, head_dim] bf16, 1 KB
      wo_b_out:       [1, 1, dim] bf16, 8 KB
      moe_out:        [1, 1, dim] bf16, 8 KB

    Per-chip footprint: ~150 KB for the lot. Fits in L1 trivially.
    """

    def __init__(self, mesh, args: ModelArgs):
        D_total = args.hc_mult * args.dim          # 16384
        Mpad = 32                                  # MHC_TILE
        nh_hd = args.n_heads * args.head_dim       # 32768
        # Residual stream lives in fp32 to match the existing hc_pre/hc_post
        # contract — demoting to bf16 is a follow-up (see test_lk_a TODO).
        self.a_tt = alloc_replicated(
            mesh, (Mpad, D_total), ttnn.float32, mem=L1,
            fill=torch.zeros(Mpad, D_total, dtype=torch.float32),
        )
        # bf16 view of the post-MoE residual the Final head consumes.
        # The block loop emits fp32 a_tt; we typecast once before Final.
        self.a_tt_bf16 = alloc_replicated(
            mesh, (Mpad, D_total), ttnn.bfloat16, mem=L1,
            fill=torch.zeros(Mpad, D_total, dtype=torch.bfloat16),
        )
        # Embedding output [B, S, DIM] bf16 — used by L0 only.
        self.embed_out = alloc_replicated(
            mesh, (1, 1, args.dim), ttnn.bfloat16, mem=L1)
        # Lk-A / L0 publish the bf16 norm tile here so Lk-C's wkv input is
        # not re-derived from the fp32 residual.
        # Lk-A path: rms_in (pre-norm); orchestrator applies rms_norm + gamma.
        # L0 path: rms_out (post-norm); used directly.
        self.x_pre_norm = alloc_replicated(
            mesh, (Mpad, args.dim), ttnn.bfloat16, mem=L1)
        self.x_post_norm = alloc_replicated(
            mesh, (Mpad, args.dim), ttnn.bfloat16, mem=L1)
        # Lk-C / Lk-D-idx-cmp / Lk-D-comp consume x_for_wkv as [B, S, DIM] =
        # [1, 1, DIM]. The producers (Lk-A bridge / L0) write [Mpad, DIM]; we
        # slice + reshape into this [1, 1, DIM] view buffer per layer.
        self.x_for_wkv_3d = alloc_replicated(
            mesh, (1, 1, args.dim), ttnn.bfloat16, mem=L1)
        self.wq_a_out = alloc_replicated(
            mesh, (1, 1, args.q_lora_rank), ttnn.bfloat16, mem=L1)
        self.q_full = alloc_replicated(
            mesh, (1, 1, nh_hd), ttnn.bfloat16, mem=L1)
        self.q_rotated = alloc_replicated(
            mesh, (1, 1, args.n_heads, args.head_dim), ttnn.bfloat16, mem=L1)
        self.kv_partial = alloc_replicated(
            mesh, (1, 1, args.head_dim), ttnn.bfloat16, mem=L1)
        self.kv_rotated = alloc_replicated(
            mesh, (1, 1, args.head_dim), ttnn.bfloat16, mem=L1)
        self.attn_out = alloc_replicated(
            mesh, (1, 1, args.dim), ttnn.bfloat16, mem=L1)
        # Shared expert partial output from Lk-E: [1, NUM_TOKENS, DIM] bf16.
        # This is added to the routed-expert sum (Lk-F) into the next
        # residual via ttnn.add inside `_block_forward_mega`.
        self.shared_partial = alloc_replicated(
            mesh, (1, 1, args.dim), ttnn.bfloat16, mem=L1)
        # Post-ffn-norm slice exposed by Lk-E. Lk-F's gate/routed-FFN
        # consumes the SAME post-norm input the shared FFN does, not the
        # shared FFN's output. Lk-E writes its internal norm_slice tile
        # (TILE rows, only row 0 carries the live token) here.
        self.norm_slice = alloc_replicated(
            mesh, (1, args.dim), ttnn.bfloat16, mem=L1)
        self.moe_out = alloc_replicated(
            mesh, (1, 1, args.dim), ttnn.bfloat16, mem=L1)
        # The next-layer residual `a_tt_next` is the output of hc_post in
        # Lk-E. We ping-pong between two `a_tt` buffers so the kernel
        # never reads/writes the same address.
        self.a_tt_next = alloc_replicated(
            mesh, (Mpad, D_total), ttnn.float32, mem=L1,
            fill=torch.zeros(Mpad, D_total, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# MegaModel — extends the legacy Model so we get HF weight loading,
# tokenizer, and trace state for free, then replaces the inner
# `_block_forward` with a mega-kernel sequence.
# ---------------------------------------------------------------------------


class MegaModel(_BaseModel):
    """`Model` that runs the decode hot path through mega kernels.

    Construction mirrors the legacy `Model`:
        m = MegaModel.from_hf(repo, max_seq_len=...)
        m.load_weights(...)
        m.offload_to_mesh(mesh)              # NEW — replaces N offload_*
        for tok in m.generate(prompt_ids, max_tokens=N): ...

    The single `offload_to_mesh` builds every per-layer kernel bundle and
    every step-buffer in one pass — there's no progressive offload to
    bisect against any more, since the mega kernels are the offload.
    """

    def __init__(self, args: ModelArgs, tokenizer, ckpt_dir: str):
        super().__init__(args, tokenizer, ckpt_dir)
        self._mesh = None
        self._layer_kernels: list[LayerKernels] = []
        self._step_buffers: Optional[StepBuffers] = None
        self._final_kernel = None
        self._l0_kernel = None
        # Cos/sin tables (CPU) — built once from ModelArgs.
        self._cos_full_cpu: Optional[torch.Tensor] = None
        self._sin_full_cpu: Optional[torch.Tensor] = None
        self._cos_compressor_cpu: Optional[torch.Tensor] = None
        self._sin_compressor_cpu: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def offload_to_mesh(self, mesh):
        """Build every per-layer mega kernel + per-step buffer on the
        given Galaxy mesh.

        Galaxy-only. No DeviceMHC / DeviceAttention / DeviceMoE — every
        weight that the mega kernels need is uploaded directly inside
        `_build_layer_kernels`; the KV cache and per-step host-upload
        buffers are allocated below.
        """
        self._mesh = mesh
        args = self.args

        # Per-step host-upload buffers (input_ids, start_pos, kv_slot). The
        # mega kernels read these directly; pre_stage writes them before
        # `execute_trace`. `offload_embedding` (called below) also uploads
        # `_input_ids_tt` / `_start_pos_tt` onto `self.transformer`; we
        # alias them here too so the local `_pre_stage` finds them.
        with _phase("setup.embed"):
            # Replaces transformer.embed.forward with a device gather +
            # all_gather, and uploads the embed weight col-sharded on the
            # mesh col axis. Also creates self.transformer._input_ids_tt
            # and ._start_pos_tt.
            self.offload_embedding(mesh)
            self._input_ids_tt = self.transformer._input_ids_tt
            self._start_pos_tt = self.transformer._start_pos_tt
            self._input_ids_upload_mapper = (
                self.transformer._input_ids_upload_mapper)

        # Sequence-sharded KV cache per the future_max_utilization brief.
        # Each `(row, col)` chip holds T_local = T_pad / 8 slots of the
        # head_dim row. Per-layer slot/owner-mask tensors live alongside
        # so the kernels can issue local writes only on the owning col.
        with _phase("setup.kv_cache"):
            self._alloc_kv_caches(mesh)

        with _phase("setup.cos_sin"):
            self._cos_full_cpu, self._sin_full_cpu = self._build_rotary_tables(
                base=args.rope_theta,
                head_dim=args.rope_head_dim,
                seqlen=args.max_seq_len,
            )
            if args.compress_rope_theta is not None:
                self._cos_compressor_cpu, self._sin_compressor_cpu = (
                    self._build_rotary_tables(
                        base=args.compress_rope_theta,
                        head_dim=args.rope_head_dim,
                        seqlen=args.max_seq_len,
                    )
                )

        with _phase("setup.l0"):
            # Layer 0's L0 mega kernel needs the embedding output; build it
            # using layer 0's hc_attn constants since L0 = embed_prep + Lk-A
            # body.
            l0_layer = self.transformer.layers[0]
            # `Block` exposes hc_attn_* on `self.transformer` — they are
            # per-decode constants of the MHC body.
            hc_attn_fn = l0_layer.hc_attn_fn.data
            hc_attn_scale = l0_layer.hc_attn_scale.data
            hc_attn_base = l0_layer.hc_attn_base.data
            self._l0_kernel = make_l0_kernel(
                mesh,
                hc_fn_cpu=hc_attn_fn,
                hc_scale_cpu=hc_attn_scale,
                hc_base_cpu=hc_attn_base,
                gamma_cpu=l0_layer.attn_norm.weight.data,
            )
            # L0 applies attn_norm gamma internally (rmsnorm has gamma_tt),
            # so it MUST receive an unbaked wq_a. Lk-A's `lk.weights["wq_a"]`
            # has gamma pre-baked into the K axis (see _build_layer_kernels);
            # using it here would compute γ² ⊙ rmsnorm(x) @ wq_a and corrupt
            # the layer-0 residual that flows through every subsequent block.
            _l0_wq_a_T = l0_layer.attn.wq_a.weight.data.transpose(0, 1) \
                .contiguous().to(torch.bfloat16)
            self._l0_wq_a_tt = ttnn.as_tensor(
                _l0_wq_a_T, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        with _phase("setup.layers"):
            for i, layer in enumerate(self.transformer.layers):
                lk = _build_layer_kernels(
                    mesh, layer, i,
                    hc_attn_fn=layer.hc_attn_fn.data,
                    hc_attn_scale=layer.hc_attn_scale.data,
                    hc_attn_base=layer.hc_attn_base.data,
                    hc_ffn_fn=layer.hc_ffn_fn.data,
                    hc_ffn_scale=layer.hc_ffn_scale.data,
                    hc_ffn_base=layer.hc_ffn_base.data,
                    cos_full_cpu=self._cos_full_cpu,
                    sin_full_cpu=self._sin_full_cpu,
                    cos_compressor_cpu=self._cos_compressor_cpu,
                    sin_compressor_cpu=self._sin_compressor_cpu,
                    args=args,
                )
                self._layer_kernels.append(lk)
                if i % 8 == 0:
                    print(f"[setup.layers] built layer {i}/{len(self.transformer.layers)}")

        with _phase("setup.indexer_compressor"):
            self._build_indexer_compressor_state(mesh)

        with _phase("setup.final"):
            self._final_kernel = make_final_kernel(mesh)
            self._build_final_constants(mesh)
            # Legacy DeviceLMHead alternate path: column-parallel sharded
            # lm_head + per-chip topk. Used in `_step_body` instead of the
            # mega Final kernel until the gibberish in the mega lmhead
            # fusion is root-caused.
            tr = self.transformer
            self._device_lm_head = DeviceLMHead(
                mesh, tr.head.weight.data,
                norm_weight=tr.norm.weight.data,
                norm_eps=tr.norm.eps,
                hc_fn=tr.hc_head_fn.data,
                hc_scale=tr.hc_head_scale.data,
                hc_base=tr.hc_head_base.data,
                hc_eps=tr.hc_eps,
                hc_mult=tr.hc_mult,
            )

        with _phase("setup.routed_experts"):
            # Reuses the base Model's bfp4_b cache loader (preprocessed
            # routed-expert weights from $DS_ROUTED_EXPERT_CACHE). After
            # this call each MoE layer's `ffn._w1_tt`, `_w3_tt`, `_w2_tt`,
            # and `_chip_ids_4d_tt` are populated.
            try:
                self.offload_moe_routed_experts(mesh)
                self._wire_routed_expert_weights()
            except FileNotFoundError as exc:
                # No bfp4 cache: continue without routed experts. Lk-F is
                # guarded in `_block_forward_mega` ("w13_routed" in
                # weights), so MoE layers gracefully fall back to
                # shared-experts-only.
                print(f"[setup.routed_experts] WARN: {exc}; skipping Lk-F")

        with _phase("setup.moe_gate"):
            # Hash-gate layers (layer_id < n_hash_layers) use a vocab→expert
            # lookup table on input_ids instead of topk(scores). Lk-F doesn't
            # support that path; for those layers we fall back to the legacy
            # gate + _forward_device_routed_cached pipeline, which needs
            # ffn.gate._device_gate populated.
            self.offload_moe_gate(mesh)
            # _forward_device_routed_cached pulls `ttnn` off
            # ffn.shared_experts._ttnn (legacy uses DeviceSharedExpert there).
            # Mega keeps the host-side Expert and runs shared in Lk-E, so
            # plant the module on the host Expert for the hash-layer fallback.
            for layer in self.transformer.layers:
                if isinstance(layer.ffn, MoE):
                    layer.ffn.shared_experts._ttnn = ttnn

        with _phase("setup.buffers"):
            self._step_buffers = StepBuffers(mesh, args)

        # Lazy: one warmup decode pulls every kernel through compile +
        # populates each closure's per-step scratch dict. Do that before
        # trace capture (handled by the existing Model.step_decode trace
        # warmup — we share its 16-step warmup window).
        print(f"[setup] mega kernels ready: {len(self._layer_kernels)} layers, "
              f"L0 + Final + per-step buffers (residual a_tt in L1)")

    @staticmethod
    def _build_rotary_tables(base: float, head_dim: int, seqlen: int):
        """Build replicated cos/sin row tables for rotary embedding. The
        per-position [rd/2] half is what the mega rotary kernels expect."""
        rd_half = head_dim // 2
        idx = torch.arange(rd_half, dtype=torch.float32)
        freqs = 1.0 / (base ** (idx / rd_half))
        t = torch.arange(seqlen, dtype=torch.float32)
        ang = torch.outer(t, freqs)            # [seqlen, rd/2]
        return ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16)

    # -----------------------------------------------------------------
    # Hot path
    # -----------------------------------------------------------------

    def _block_forward_mega(self, layer_idx: int, lk: LayerKernels,
                            a_tt, a_tt_next, start_pos_tt, input_ids_tt,
                            sb: StepBuffers):
        """One transformer block via mega kernels.

        Bringup workaround: `ttnn.synchronize_device` between
        `lk_d_idx_cmp` and `lk_d_idx_score`. Without it, paged_update_cache
        state on core (0, 0) conflicts with the score kernel's PipeNet on
        the same core and the device hangs. Track as a tt-lang issue.
        """
        if layer_idx == 0:
            pass
        else:
            with _phase("lk_a"):
                lk.lk_a(a_tt, lk.weights["wq_a"], sb.wq_a_out,
                        x_pre_norm_bf16_out=sb.x_pre_norm)
            with _phase("lk_a_bridge"):
                _xn = lk.attn_norm.forward_device(sb.x_pre_norm, num_rows=1)
                ttnn.copy(_xn, sb.x_post_norm)
        with _phase("x_post_norm_to_3d"):
            _row = ttnn.slice(sb.x_post_norm, [0, 0], [1, self.args.dim])
            _row3 = ttnn.reshape(_row, [1, 1, self.args.dim])
            ttnn.copy(_row3, sb.x_for_wkv_3d)
        x_for_wkv = sb.x_for_wkv_3d

        with _phase("lk_b"):
            lk.lk_b(sb.wq_a_out, sb.q_full)

        with _phase("lk_c"):
            lk.lk_c(sb.q_full, x_for_wkv, None, None, start_pos_tt,
                    lk.weights["wkv"], sb.q_rotated, sb.kv_partial)

        with _phase("lk_d1"):
            lk.lk_d1(sb.kv_partial, None, None, start_pos_tt, sb.kv_rotated)

        if lk.lk_d_idx_q is not None:
            with _phase("lk_d_idx_q"):
                lk.lk_d_idx_q(
                    sb.wq_a_out,
                    lk.weights["attn_norm_gamma"],
                    lk.weights["attn_norm_gamma"],
                    start_pos_tt,
                    lk.weights["indexer_wq_b"],
                    lk.weights["indexer_H"],
                    self._q_idx_tt,
                )
            with _phase("lk_d_idx_cmp"):
                lk.lk_d_idx_cmp(
                    x_for_wkv,
                    lk.weights["indexer_wkv_gate"],
                    lk.weights["idx_ape_padded"],
                    start_pos_tt,
                    lk.weights["idx_state_slot"],
                    lk.weights["idx_kv_state_front_4d"],
                    lk.weights["idx_kv_state_back_4d"],
                    lk.weights["idx_score_state_front_4d"],
                    lk.weights["idx_score_state_back_4d"],
                    self._idx_cmp_kv_tt, self._idx_cmp_score_tt,
                )
            if (self._emit_now if hasattr(self, "_emit_now") else False):
                with _phase("lk_d_idx_emit"):
                    lk.lk_d_idx_emit(
                        lk.weights["idx_kv_state_front_2d"],
                        lk.weights["idx_kv_state_back_2d"],
                        lk.weights["idx_score_state_front_2d"],
                        lk.weights["idx_score_state_back_2d"],
                        lk.weights["idx_cssn_mask_front"],
                        lk.weights["idx_cssn_mask_back"],
                        lk.weights["idx_cssn_mask_pad"],
                        lk.weights["idx_norm_gamma"],
                        lk.weights["idx_norm_scaler"],
                        lk.weights["attn_norm_gamma"],
                        lk.weights["attn_norm_gamma"],
                        start_pos_tt,
                        lk.weights["idx_H"],
                        lk.weights["idx_kv_cache"],
                        lk.weights["idx_emit_slot"],
                        lk.weights["idx_shift_P"],
                        lk.weights["idx_kv_state_front_scratch"],
                        lk.weights["idx_kv_state_back_scratch"],
                        lk.weights["idx_score_state_front_scratch"],
                        lk.weights["idx_score_state_back_scratch"],
                        self._idx_kv_normed_sink_tt,
                    )
            ttnn.synchronize_device(self._mesh)
            with _phase("lk_d_idx_score"):
                lk.lk_d_idx_score(
                    x_for_wkv,
                    lk.weights["indexer_wproj_scaled"],
                    self._q_idx_tt,
                    lk.weights["idx_kv_cache"],
                    self._idx_score_tt,
                )
            with _phase("lk_d_topk"):
                lk.lk_d_topk(
                    self._idx_score_tt,
                    self._topk_ramp_int_tt,
                    self._t_active_tt,
                    self._cmp_idxs_tt,
                )
        if lk.lk_d_comp is not None and (
                self._emit_now if hasattr(self, "_emit_now") else False):
            with _phase("lk_d_comp"):
                lk.lk_d_comp(
                    x_for_wkv,
                    lk.weights["comp_wkv"], lk.weights["comp_wgate"],
                    lk.weights["comp_ape_padded"],
                    lk.weights["attn_norm_gamma"],
                    lk.weights["attn_norm_gamma"],
                    start_pos_tt,
                    lk.weights["comp_state_slot"],
                    lk.weights["comp_emit_slot"],
                    lk.weights["comp_kv_state_front_2d"],
                    lk.weights["comp_kv_state_back_2d"],
                    lk.weights["comp_score_state_front_2d"],
                    lk.weights["comp_score_state_back_2d"],
                    lk.weights["comp_kv_state_front_4d"],
                    lk.weights["comp_kv_state_back_4d"],
                    lk.weights["comp_score_state_front_4d"],
                    lk.weights["comp_score_state_back_4d"],
                    lk.weights["comp_cssn_mask_front"],
                    lk.weights["comp_cssn_mask_back"],
                    lk.weights["comp_cssn_mask_pad"],
                    lk.weights["comp_norm_gamma"],
                    lk.weights["comp_norm_scaler"],
                    lk.weights["comp_kv_cache"],
                    lk.weights["comp_shift_P"],
                    lk.weights["comp_kv_state_front_scratch"],
                    lk.weights["comp_kv_state_back_scratch"],
                    lk.weights["comp_score_state_front_scratch"],
                    lk.weights["comp_score_state_back_scratch"],
                    lk.weights["comp_cssn_out"],
                    self._comp_kv_normed_sink_tt,
                )
        # Per-step window topk indices: row-pick start_pos from the precomputed
        # [max_seq_len, win] bf16 table (-1 sentinels for unfilled cache slots),
        # typecast back to int32 → [1, 1, win] for lk_dsparse. Without
        # sentinels, attention mass spreads across zero-init cache slots.
        win = self.args.window_size
        with _phase("win_idxs"):
            _win_bf16 = ttnn.embedding(
                start_pos_tt, self._win_idxs_padded_tt,
                layout=ttnn.TILE_LAYOUT,
            )
            _win_i32 = ttnn.typecast(_win_bf16, dtype=ttnn.int32)
            topk_idxs_tt = ttnn.reshape(_win_i32, [1, 1, win])
        with _phase("lk_dsparse"):
            kv_cache_tt = self._kv_cache_tt[layer_idx]
            kv_slot_tt = self._kv_slot_tt[layer_idx]
            sink_4d_tt = lk.weights["sink_4d"]
            lk.lk_dsparse(
                sb.q_rotated, sb.kv_rotated, kv_cache_tt, kv_slot_tt,
                topk_idxs_tt, sink_4d_tt,
                None, None, start_pos_tt,
                lk.weights["wo_a"], lk.weights["wo_b"], sb.attn_out,
            )

        with _phase("lk_e"):
            lk.lk_e(
                sb.attn_out, a_tt,
                lk.weights["w1_shared"],
                lk.weights["w2_shared"],
                lk.weights["w3_shared"],
                sb.shared_partial, a_tt_next,
                norm_slice_out=sb.norm_slice,
            )

        is_hash_layer = layer_idx < self.args.n_hash_layers
        if is_hash_layer and isinstance(self.transformer.layers[layer_idx].ffn, MoE):
            with _phase("lk_f_hash"):
                # Hash-gate layers can't use Lk-F's topk-based gate; the
                # indices come from a vocab→expert lookup on input_ids.
                # Fall back to the legacy gate + cached-routed path, which
                # internally all_reduces routed across the mesh and adds
                # shared_out (we feed our Lk-E shared partial as that).
                ffn = self.transformer.layers[layer_idx].ffn
                gate_dev = ffn.gate._device_gate
                weights_tt, indices_tt = gate_dev.forward_device(
                    sb.norm_slice, input_ids_tt)
                shared_2d = ttnn.reshape(
                    sb.shared_partial, [1, self.args.dim])
                combined_2d = ffn._forward_device_routed_cached(
                    sb.norm_slice, weights_tt, indices_tt, shared_2d)
                combined_3d = ttnn.reshape(
                    combined_2d, [1, 1, self.args.dim])
                ttnn.copy(combined_3d, sb.shared_partial)
        elif lk.lk_f is not None and "w13_routed" in lk.weights:
            with _phase("lk_f"):
                # Lk-F gate + routed FFN takes the SAME post-norm input as
                # the shared FFN inside Lk-E, not the shared FFN's output.
                routed_partial = lk.lk_f(
                    sb.norm_slice,
                    lk.weights["gate_w"], lk.weights["gate_bias"],
                    lk.weights["w13_routed"],
                    lk.weights["w2_routed"],
                    lk.weights["chip_ids_4d"],
                )
                # Each chip holds PER_CHIP=8 of the 256 routed experts; sum
                # across the full mesh to assemble the global routed output.
                routed_full = ttnn.all_reduce(routed_partial)
                routed_3d = ttnn.reshape(routed_full, [1, 1, self.args.dim])
                ttnn.add(sb.shared_partial, routed_3d,
                         output_tensor=sb.shared_partial)

        # hc_post_ffn bridge: legacy DeviceMHC. Lk-E only does hc_post_attn
        # + hc_pre_ffn + shared expert; the FFN-side hc_post is not in any
        # mega kernel. We re-run hc_pre_ffn to populate the stash (needed
        # by hc_post_device) and apply the MHC-weighted combine here.
        with _phase("hc_post_ffn"):
            mhc_ffn = lk.mhc_ffn
            num_tokens = 1
            num_tokens_pad = _MHC_TILE
            mhc = self.args.hc_mult
            hidden = self.args.dim
            # Stash population. Output of hc_pre_ffn is discarded — Lk-E
            # already produced the same intermediate to feed its own
            # ffn_norm + shared expert.
            mhc_ffn.hc_pre_device(
                num_tokens, num_tokens_pad, a_tt=a_tt_next)
            # Build [TILE, DIM] fp32 x_tt for hc_post: broadcast the FFN
            # partial across `mhc` head bins and pad to TILE rows.
            shared_4d = ttnn.reshape(
                sb.shared_partial, [num_tokens, 1, hidden])
            shared_repeated = ttnn.repeat(
                shared_4d, ttnn.Shape([1, mhc, 1]))
            shared_padded = ttnn.pad(
                shared_repeated,
                padding=[(0, 0), (0, _MHC_TILE - mhc), (0, 0)],
                value=0.0)
            x_post_input = ttnn.reshape(
                shared_padded, [num_tokens * _MHC_TILE, hidden])
            ttnn.typecast(x_post_input, dtype=ttnn.float32,
                          output_tensor=mhc_ffn._x_upload_tt)
            ffn_post_out_tt = mhc_ffn.hc_post_device(num_tokens)
            next_a_input_tt = _mhc_post_to_a_tt(
                ttnn, ffn_post_out_tt, num_tokens, num_tokens_pad,
                mhc, hidden)
            ttnn.copy(next_a_input_tt, a_tt_next)
        return a_tt_next

    def _block_forward_mega_full(self, layer_idx: int, lk: LayerKernels,
                            a_tt, a_tt_next, start_pos_tt, input_ids_tt,
                            sb: StepBuffers):
        """One transformer block via mega kernels.

        Reads residual `a_tt` (the L1 fp32 residual stream); writes
        next-layer residual into `a_tt_next`. All intermediates between
        kernels are L1-resident `sb.*` buffers.

        For layer 0 we reuse L0's outputs (`sb.wq_a_out`, `sb.x_post_norm`)
        and skip Lk-A. For layer i > 0 we run Lk-A and apply a device-side
        rms_norm with gamma to bridge to Lk-C (Lk-A bakes gamma into wq_a
        so its post-norm tile is never materialized).

        CCL boundaries (see `briefs/future_max_utilization.md`) are not
        yet wired — kernels run replicated for bringup. Future work
        replaces the implicit replication with explicit reduce_scatter +
        all_gather across the row/col mesh axes.
        """
        # --- Lk-A (skip for layer 0; L0 already published wq_a_out + x_post_norm) ---
        if layer_idx == 0:
            pass  # L0 wrote sb.x_post_norm
        else:
            with _phase("lk_a"):
                lk.lk_a(a_tt, lk.weights["wq_a"], sb.wq_a_out,
                        x_pre_norm_bf16_out=sb.x_pre_norm)
            with _phase("lk_a_bridge"):
                _xn = lk.attn_norm.forward_device(sb.x_pre_norm, num_rows=1)
                ttnn.copy(_xn, sb.x_post_norm)
        # Producers (Lk-A bridge / L0) emit [Mpad=32, DIM]; downstream
        # consumers (Lk-C, Lk-D-idx-cmp, Lk-D-comp, Lk-Dsparse) want
        # [1, 1, DIM]. Slice + reshape into the bridge buffer.
        with _phase("x_post_norm_to_3d"):
            _row = ttnn.slice(sb.x_post_norm, [0, 0], [1, self.args.dim])
            _row3 = ttnn.reshape(_row, [1, 1, self.args.dim])
            ttnn.copy(_row3, sb.x_for_wkv_3d)
        x_for_wkv = sb.x_for_wkv_3d

        # --- Lk-B: q_norm + wq_b -----------------------------------------
        with _phase("lk_b"):
            lk.lk_b(sb.wq_a_out, sb.q_full)

        # --- Lk-C: q rotary + wkv ----------------------------------------
        # x_for_wkv is [1, 1, DIM] bf16; Lk-C reshapes to [B*S, DIM] and
        # pads to [TILE, DIM] internally.
        with _phase("lk_c"):
            lk.lk_c(sb.q_full, x_for_wkv, None, None, start_pos_tt,
                    lk.weights["wkv"], sb.q_rotated, sb.kv_partial)

        # --- Lk-D1 -------------------------------------------------------
        with _phase("lk_d1"):
            lk.lk_d1(sb.kv_partial, None, None, start_pos_tt, sb.kv_rotated)

        # --- Lk-D-idx-* / Lk-D-comp (indexer + compressor) ---------------
        # For ratio=4 layers we run the full indexer pipeline (q rotated +
        # cmp + emit-on-emit-pos + score) and the attn-side compressor
        # (Lk-D-comp on emit-pos). The Lk-D-topk output produces 64
        # compressed-token indices but Lk-Dsparse was tested for K=128
        # window-only; for the bringup smoke run we still feed Lk-Dsparse
        # the window-ramp topk and treat the indexer output as a shadow
        # path (state buffers are exercised, score is computed, but the
        # selection is not consumed downstream). The K-mismatch fix is
        # tracked separately.
        if lk.lk_d_idx_q is not None:
            # Wq_a output is [1, 1, q_lora_rank=1024] — that's the qr we
            # feed Lk-D-idx-q. cos_full/sin_full args are baked into the
            # factory closure; pass any pre-allocated tensor as a stand-in
            # (the kernel ignores them).
            with _phase("lk_d_idx_q"):
                lk.lk_d_idx_q(
                    sb.wq_a_out,
                    lk.weights["attn_norm_gamma"],   # cos_full slot — unused
                    lk.weights["attn_norm_gamma"],   # sin_full slot — unused
                    start_pos_tt,
                    lk.weights["indexer_wq_b"],
                    lk.weights["indexer_H"],
                    self._q_idx_tt,
                )
            # The post-attn-norm tile (x_for_wkv) is what feeds the
            # indexer's compressor wkv/wgate matmul.
            with _phase("lk_d_idx_cmp"):
                lk.lk_d_idx_cmp(
                    x_for_wkv,
                    lk.weights["indexer_wkv_gate"],
                    lk.weights["idx_ape_padded"],
                    start_pos_tt,
                    lk.weights["idx_state_slot"],
                    lk.weights["idx_kv_state_front_4d"],
                    lk.weights["idx_kv_state_back_4d"],
                    lk.weights["idx_score_state_front_4d"],
                    lk.weights["idx_score_state_back_4d"],
                    self._idx_cmp_kv_tt, self._idx_cmp_score_tt,
                )
            # Lk-D-idx-emit only on positions where (start_pos+1) % ratio==0.
            # We branch on the host bool (the trace machinery captures
            # separate emit / no-emit traces).
            if (self._emit_now if hasattr(self, "_emit_now") else False):
                with _phase("lk_d_idx_emit"):
                    lk.lk_d_idx_emit(
                        lk.weights["idx_kv_state_front_2d"],
                        lk.weights["idx_kv_state_back_2d"],
                        lk.weights["idx_score_state_front_2d"],
                        lk.weights["idx_score_state_back_2d"],
                        lk.weights["idx_cssn_mask_front"],
                        lk.weights["idx_cssn_mask_back"],
                        lk.weights["idx_cssn_mask_pad"],
                        lk.weights["idx_norm_gamma"],
                        lk.weights["idx_norm_scaler"],
                        lk.weights["attn_norm_gamma"],   # cos_compressor — unused
                        lk.weights["attn_norm_gamma"],   # sin_compressor — unused
                        start_pos_tt,
                        lk.weights["idx_H"],
                        lk.weights["idx_kv_cache"],
                        lk.weights["idx_emit_slot"],
                        lk.weights["idx_shift_P"],
                        lk.weights["idx_kv_state_front_scratch"],
                        lk.weights["idx_kv_state_back_scratch"],
                        lk.weights["idx_score_state_front_scratch"],
                        lk.weights["idx_score_state_back_scratch"],
                        self._idx_kv_normed_sink_tt,
                    )
            with _phase("lk_d_idx_score"):
                lk.lk_d_idx_score(
                    x_for_wkv,
                    lk.weights["indexer_wproj_scaled"],
                    self._q_idx_tt,
                    lk.weights["idx_kv_cache"],
                    self._idx_score_tt,
                )
            with _phase("lk_d_topk"):
                lk.lk_d_topk(
                    self._idx_score_tt,
                    self._topk_ramp_int_tt,
                    self._t_active_tt,
                    self._cmp_idxs_tt,
                )
        if lk.lk_d_comp is not None and (
                self._emit_now if hasattr(self, "_emit_now") else False):
            with _phase("lk_d_comp"):
                lk.lk_d_comp(
                    x_for_wkv,
                    lk.weights["comp_wkv"], lk.weights["comp_wgate"],
                    lk.weights["comp_ape_padded"],
                    lk.weights["attn_norm_gamma"],   # cos_compressor — unused
                    lk.weights["attn_norm_gamma"],   # sin_compressor — unused
                    start_pos_tt,
                    lk.weights["comp_state_slot"],
                    lk.weights["comp_emit_slot"],
                    lk.weights["comp_kv_state_front_2d"],
                    lk.weights["comp_kv_state_back_2d"],
                    lk.weights["comp_score_state_front_2d"],
                    lk.weights["comp_score_state_back_2d"],
                    lk.weights["comp_kv_state_front_4d"],
                    lk.weights["comp_kv_state_back_4d"],
                    lk.weights["comp_score_state_front_4d"],
                    lk.weights["comp_score_state_back_4d"],
                    lk.weights["comp_cssn_mask_front"],
                    lk.weights["comp_cssn_mask_back"],
                    lk.weights["comp_cssn_mask_pad"],
                    lk.weights["comp_norm_gamma"],
                    lk.weights["comp_norm_scaler"],
                    lk.weights["comp_kv_cache"],
                    lk.weights["comp_shift_P"],
                    lk.weights["comp_kv_state_front_scratch"],
                    lk.weights["comp_kv_state_back_scratch"],
                    lk.weights["comp_score_state_front_scratch"],
                    lk.weights["comp_score_state_back_scratch"],
                    lk.weights["comp_cssn_out"],
                    self._comp_kv_normed_sink_tt,
                )
        # Per-step window topk indices: row-pick start_pos from the precomputed
        # [max_seq_len, win] bf16 table (-1 sentinels for unfilled cache slots),
        # typecast back to int32 → [1, 1, win] for lk_dsparse. (K mismatch with
        # Lk-D-topk's K_FIXED=64 output is tracked separately.)
        win = self.args.window_size
        with _phase("win_idxs"):
            _win_bf16 = ttnn.embedding(
                start_pos_tt, self._win_idxs_padded_tt,
                layout=ttnn.TILE_LAYOUT,
            )
            _win_i32 = ttnn.typecast(_win_bf16, dtype=ttnn.int32)
            topk_idxs_tt = ttnn.reshape(_win_i32, [1, 1, win])

        # --- Lk-Dsparse: sparse attention → wo_a → wo_b ------------------
        with _phase("lk_dsparse"):
            kv_cache_tt = self._kv_cache_tt[layer_idx]
            kv_slot_tt = self._kv_slot_tt[layer_idx]
            sink_4d_tt = lk.weights["sink_4d"]
            lk.lk_dsparse(
                sb.q_rotated, sb.kv_rotated, kv_cache_tt, kv_slot_tt,
                topk_idxs_tt, sink_4d_tt,
                None, None, start_pos_tt,
                lk.weights["wo_a"], lk.weights["wo_b"], sb.attn_out,
            )

        # --- Lk-E: hc_post_attn + ffn_norm + shared expert ---------------
        # Lk-E writes the next residual into a_tt_next and the shared
        # expert partial into shared_partial. Call signature:
        #   lk_e(attn_out, prev_a_tt, w1, w2, w3, shared_partial_out, next_a_out)
        with _phase("lk_e"):
            lk.lk_e(
                sb.attn_out, a_tt,
                lk.weights["w1_shared"],
                lk.weights["w2_shared"],
                lk.weights["w3_shared"],
                sb.shared_partial, a_tt_next,
                norm_slice_out=sb.norm_slice,
            )

        # --- Lk-F: gate post + routed experts (MoE layers only) ----------
        if lk.lk_f is not None and "w13_routed" in lk.weights:
            # M6 prerequisite: routed-expert weight upload. Skip the call
            # until weights["w13_routed"] / weights["w2_routed"] /
            # weights["chip_ids_4d"] are populated by the routed-expert
            # offload phase.
            with _phase("lk_f"):
                # Lk-F gate + routed FFN takes the SAME post-norm input as
                # the shared FFN inside Lk-E, not the shared FFN's output.
                routed_partial = lk.lk_f(
                    sb.norm_slice,
                    lk.weights["gate_w"], lk.weights["gate_bias"],
                    lk.weights["w13_routed"],
                    lk.weights["w2_routed"],
                    lk.weights["chip_ids_4d"],
                )
                # Each chip holds PER_CHIP=8 of the 256 routed experts; sum
                # across the full mesh to assemble the global routed output.
                routed_full = ttnn.all_reduce(routed_partial)
                routed_3d = ttnn.reshape(routed_full, [1, 1, self.args.dim])
                ttnn.add(sb.shared_partial, routed_3d,
                         output_tensor=sb.shared_partial)

        # Fold the shared+routed partial back into a_tt_next. Lk-E's
        # `next_a_out` already contains the post-attn residual stream
        # (hc_post_attn output); the FFN partial gets added on top, broadcast
        # across the MHC tiling. The reference path does this via a 4D
        # broadcast add into the [Mpad, MHC*DIM] residual.
        with _phase("residual_add"):
            shared_4d = ttnn.reshape(
                sb.shared_partial, [1, 1, self.args.dim])
            shared_repeated = ttnn.repeat(
                shared_4d, ttnn.Shape([1, self.args.hc_mult, 1]))
            shared_2d = ttnn.reshape(
                shared_repeated,
                [1, self.args.hc_mult * self.args.dim])
            shared_padded = ttnn.pad(
                shared_2d,
                padding=[(0, _MHC_TILE - 1), (0, 0)], value=0.0)
            shared_fp32 = ttnn.typecast(shared_padded, dtype=ttnn.float32)
            ttnn.add(a_tt_next, shared_fp32, output_tensor=a_tt_next)

        return a_tt_next

    # -----------------------------------------------------------------
    # KV cache + global per-step buffers
    # -----------------------------------------------------------------

    def _wire_routed_expert_weights(self):
        """After `offload_moe_routed_experts`, attach the per-MoE
        ffn._w1/_w2/_w3_tt and ffn._chip_ids_4d_tt onto each
        `LayerKernels.weights` so `_block_forward_mega` can pass them to
        Lk-F. Builds the concat'd w13 = ttnn.concat([_w1, _w3], dim=-1)
        on device once at offload time (Lk-F's matmul wants a single
        weight tensor on the N axis).
        """
        for layer_idx, lk in enumerate(self._layer_kernels):
            ffn = self.transformer.layers[layer_idx].ffn
            if not isinstance(ffn, MoE) or not hasattr(ffn, "_w1_tt"):
                continue
            lk.weights["w13_routed"] = ttnn.concat(
                [ffn._w1_tt, ffn._w3_tt], dim=-1)
            lk.weights["w2_routed"] = ffn._w2_tt
            lk.weights["chip_ids_4d"] = ffn._chip_ids_4d_tt

    def _build_indexer_compressor_state(self, mesh):
        """For every ratio==4 layer, instantiate a DeviceCompressor for the
        attn-side compressor and a DeviceIndexer (with its own inner
        DeviceCompressor) for the indexer. Both publish device-resident
        state buffers (kv_state_*, score_state_*, ape_padded, cssn masks,
        shift_P, _state_slot, _emit_slot, h_tt) that the mega kernels read
        directly. We don't go through `offload_attn_full` /
        `offload_compressor_indexer` — those build their own kv_cache and
        DeviceColLinear weights that don't match the mega kernels'
        expectations. We only need the device wrappers as constant
        holders; they are never `forward_device`-called from the mega
        path. The DeviceColLinear instances exist so DeviceCompressor's
        constructor doesn't reject the args.

        Per layer we also allocate:
          * `indexer_kv_cache_tt`: replicated [1, 1, INDEX_T_PAD, D] bf16
            buffer the indexer compressor's emit step writes into and
            Lk-D-idx-score reads from.
          * `t_active_tt`: persistent int32 device tensor uploaded each
            step before topk. Sized [B=1, 1, BUCKET=128] for the
            mask-build kernel's [TILE, BUCKET] output.
          * `topk_idxs_tt`: int32 [1, 1, K_FIXED=64] holding the
            per-step compressed-token indices for Lk-Dsparse.
        """
        for layer_idx, layer in enumerate(self.transformer.layers):
            attn = layer.attn
            if attn.compress_ratio != 4:
                continue
            lk = self._layer_kernels[layer_idx]
            # Wire freqs_cis/kv_cache lazily (mirrors offload_compressor_indexer).
            if attn.compressor.freqs_cis is None:
                attn.compressor.freqs_cis = attn.freqs_cis
            if attn.compressor.kv_cache is None:
                attn.compressor.kv_cache = attn.kv_cache[:, attn.window_size:]
            if attn.indexer.freqs_cis is None:
                attn.indexer.freqs_cis = attn.freqs_cis
            if attn.indexer.compressor.freqs_cis is None:
                attn.indexer.compressor.freqs_cis = attn.indexer.freqs_cis
            if attn.indexer.compressor.kv_cache is None:
                attn.indexer.compressor.kv_cache = attn.indexer.kv_cache

            # --- attn-side compressor -------------------------------------
            wkv_dev = DeviceColLinear(mesh, attn.compressor.wkv.weight)
            wgate_dev = DeviceColLinear(mesh, attn.compressor.wgate.weight)
            norm_dev = DeviceRMSNorm(
                mesh, attn.compressor.norm.weight, attn.compressor.norm.eps)
            dc = DeviceCompressor(
                mesh=mesh, comp=attn.compressor,
                wkv_dev=wkv_dev, wgate_dev=wgate_dev, norm_dev=norm_dev,
            )
            # Bind to the seq-sharded mega kv_cache. The Lk-D-comp emit-slot
            # writes against this buffer for parity; topk currently only
            # reads window slots so the compressed slots are not yet read
            # downstream. (Future: replicated kv_cache or a Lk-Dsparse path
            # that reads compressed slots from the indexer's cache.)
            dc.bind_kv_cache_tt(
                self._kv_cache_tt[layer_idx], slot_offset=attn.window_size)
            dc._alloc_decode_tensors(B=1)
            attn._device_compressor = dc

            # --- indexer's inner compressor (rotate=True) -----------------
            ix = attn.indexer
            ix_wkv_dev = DeviceColLinear(mesh, ix.compressor.wkv.weight)
            ix_wgate_dev = DeviceColLinear(mesh, ix.compressor.wgate.weight)
            ix_norm_dev = DeviceRMSNorm(
                mesh, ix.compressor.norm.weight, ix.compressor.norm.eps)
            ix_dc = DeviceCompressor(
                mesh=mesh, comp=ix.compressor,
                wkv_dev=ix_wkv_dev, wgate_dev=ix_wgate_dev,
                norm_dev=ix_norm_dev,
            )
            ix_dc._alloc_decode_tensors(B=1)
            # Indexer's own kv_cache: replicated [1, 1, INDEX_T_PAD, D].
            # Allocate fresh; the legacy DeviceCompressor with
            # `_owns_kv_cache=True` will allocate one but sized from
            # comp.kv_cache.shape (max_seq_len // 4 slots). We override
            # to a fixed INDEX_T_PAD for the mega kernels.
            indexer_kv_cache_cpu = torch.zeros(
                1, 1, INDEX_T_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
            indexer_kv_cache_tt = ttnn.as_tensor(
                indexer_kv_cache_cpu.contiguous(),
                device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ix_dc.kv_cache_tt = indexer_kv_cache_tt
            ix_dc.slot_offset = 0
            ix_dc._owns_kv_cache = True

            wq_b_dev = DeviceColLinear(mesh, ix.wq_b.weight)
            wproj_dev = DeviceColLinear(mesh, ix.weights_proj.weight)
            di = DeviceIndexer(
                mesh=mesh, indexer=ix, dc=ix_dc,
                wq_b_dev=wq_b_dev, weights_proj_dev=wproj_dev,
            )
            attn._device_indexer = di

            # --- per-layer state references onto lk.weights ---------------
            # Indexer compressor (rotate=True) state.
            lk.weights["idx_kv_state_front_2d"] = ix_dc.kv_state_front_2d_tt
            lk.weights["idx_kv_state_back_2d"] = ix_dc.kv_state_back_2d_tt
            lk.weights["idx_score_state_front_2d"] = ix_dc.score_state_front_2d_tt
            lk.weights["idx_score_state_back_2d"] = ix_dc.score_state_back_2d_tt
            lk.weights["idx_kv_state_front_4d"] = ix_dc.kv_state_front_tt
            lk.weights["idx_kv_state_back_4d"] = ix_dc.kv_state_back_tt
            lk.weights["idx_score_state_front_4d"] = ix_dc.score_state_front_tt
            lk.weights["idx_score_state_back_4d"] = ix_dc.score_state_back_tt
            lk.weights["idx_kv_state_front_scratch"] = ix_dc.kv_state_front_out_tt
            lk.weights["idx_kv_state_back_scratch"] = ix_dc.kv_state_back_out_tt
            lk.weights["idx_score_state_front_scratch"] = ix_dc.score_state_front_out_tt
            lk.weights["idx_score_state_back_scratch"] = ix_dc.score_state_back_out_tt
            lk.weights["idx_cssn_mask_front"] = ix_dc.cssn_mask_front_tt
            lk.weights["idx_cssn_mask_back"] = ix_dc.cssn_mask_back_tt
            lk.weights["idx_cssn_mask_pad"] = ix_dc.cssn_mask_pad_tt
            lk.weights["idx_cssn_out"] = ix_dc.cssn_out_tt
            lk.weights["idx_ape_padded"] = ix_dc.ape_padded_tt
            lk.weights["idx_shift_P"] = ix_dc.shift_P_tt
            lk.weights["idx_norm_gamma"] = ix_dc.norm_dev.gamma_tt
            lk.weights["idx_norm_scaler"] = ix_dc.norm_dev.sc_tt
            lk.weights["idx_state_slot"] = ix_dc._state_slot_tt
            lk.weights["idx_emit_slot"] = ix_dc._emit_slot_tt
            lk.weights["idx_kv_cache"] = ix_dc.kv_cache_tt

            # Attn-side compressor (rotate=False, head_dim=512) state.
            lk.weights["comp_kv_state_front_2d"] = dc.kv_state_front_2d_tt
            lk.weights["comp_kv_state_back_2d"] = dc.kv_state_back_2d_tt
            lk.weights["comp_score_state_front_2d"] = dc.score_state_front_2d_tt
            lk.weights["comp_score_state_back_2d"] = dc.score_state_back_2d_tt
            lk.weights["comp_kv_state_front_4d"] = dc.kv_state_front_tt
            lk.weights["comp_kv_state_back_4d"] = dc.kv_state_back_tt
            lk.weights["comp_score_state_front_4d"] = dc.score_state_front_tt
            lk.weights["comp_score_state_back_4d"] = dc.score_state_back_tt
            lk.weights["comp_kv_state_front_scratch"] = dc.kv_state_front_out_tt
            lk.weights["comp_kv_state_back_scratch"] = dc.kv_state_back_out_tt
            lk.weights["comp_score_state_front_scratch"] = dc.score_state_front_out_tt
            lk.weights["comp_score_state_back_scratch"] = dc.score_state_back_out_tt
            lk.weights["comp_cssn_mask_front"] = dc.cssn_mask_front_tt
            lk.weights["comp_cssn_mask_back"] = dc.cssn_mask_back_tt
            lk.weights["comp_cssn_mask_pad"] = dc.cssn_mask_pad_tt
            lk.weights["comp_cssn_out"] = dc.cssn_out_tt
            lk.weights["comp_ape_padded"] = dc.ape_padded_tt
            lk.weights["comp_shift_P"] = dc.shift_P_tt
            lk.weights["comp_norm_gamma"] = dc.norm_dev.gamma_tt
            lk.weights["comp_norm_scaler"] = dc.norm_dev.sc_tt
            lk.weights["comp_state_slot"] = dc._state_slot_tt
            lk.weights["comp_emit_slot"] = dc._emit_slot_tt
            lk.weights["comp_kv_cache"] = dc.kv_cache_tt
            # Hadamard table for the indexer kv (in Lk-D-idx-emit body).
            lk.weights["idx_H"] = ix_dc.h_tt

        # Per-step int32 buffer for Lk-D-topk's t_active mask. Replicated.
        self._t_active_tt = ttnn.from_torch(
            torch.zeros(1, 1, INDEX_TOPK_BUCKET, dtype=torch.int32),
            device=mesh, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Persistent int32 ramp [0..BUCKET-1] for Lk-D-topk.
        ramp_int = (torch.arange(INDEX_TOPK_BUCKET, dtype=torch.int32)
                    .view(1, 1, INDEX_TOPK_BUCKET).contiguous())
        self._topk_ramp_int_tt = ttnn.from_torch(
            ramp_int, device=mesh, dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Per-step output of Lk-D-topk: [1, 1, K_FIXED=64] int32 indices
        # used as the topk for Lk-Dsparse on indexer layers. Allocated
        # once, written each step.
        self._cmp_idxs_tt = ttnn.from_torch(
            torch.zeros(1, 1, 64, dtype=torch.int32),
            device=mesh, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Per-step indexer score output: [1, 1, T_PAD=128] bf16.
        self._idx_score_tt = ttnn.from_torch(
            torch.zeros(1, 1, INDEX_T_PAD, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Per-step indexer cmp partial outputs (kv_3d, score_3d).
        self._idx_cmp_kv_tt = ttnn.from_torch(
            torch.zeros(1, 1, 2 * INDEX_HEAD_DIM, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        self._idx_cmp_score_tt = ttnn.from_torch(
            torch.zeros(1, 1, 2 * INDEX_HEAD_DIM, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Per-step q_idx [1, 1, INDEX_N_HEADS, INDEX_HEAD_DIM] bf16.
        self._q_idx_tt = ttnn.from_torch(
            torch.zeros(1, 1, INDEX_N_HEADS, INDEX_HEAD_DIM,
                        dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Per-step Lk-D-comp output (replicated kv_normed scratch); only
        # used as a sink — the kv_cache write is the side effect we want.
        # Attn-side compressor head_dim = head_dim from any ratio==4 layer
        # (uniform across the model); fall back to 512 for V4-Flash.
        comp_head_dim = self.args.head_dim
        self._comp_kv_normed_sink_tt = ttnn.from_torch(
            torch.zeros(1, 1, comp_head_dim, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Lk-D-idx-emit kv_normed sink (smaller — INDEX_HEAD_DIM=128).
        self._idx_kv_normed_sink_tt = ttnn.from_torch(
            torch.zeros(1, 1, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def _build_final_constants(self, mesh):
        """Pack and upload the Final head's host constants once.

        Mirrors the per-test setup in `test_final.main`:
          hc_fn_t_scaled = _pack_hc_fn_t_scaled(hc_head_fn, hc_head_scale)
          hc_base_tile   = _pack_hc_base_tile(hc_head_base)
          scaler_bf16    = ones [TILE, TILE]
          w_baked        = (norm.weight[:, None] * head.weight).bf16

        These are decode-step constants of the lm-head and never change at
        runtime, so we upload once at offload time and re-use across every
        step / trace replay.
        """
        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        hc_fn = self.transformer.hc_head_fn.data.float()
        hc_scale = self.transformer.hc_head_scale.data.float()
        hc_base = self.transformer.hc_head_base.data.float().view(1, MHC)
        norm_gamma = self.transformer.norm.weight.data.float()
        # nn.Linear stores head.weight as [VOCAB=out, DIM=in]; the fused
        # rms+ksplit lmhead expects [DIM=in, VOCAB=out] with gamma baked
        # into the K (DIM) axis.
        w_lmhead = self.transformer.head.weight.data.float().transpose(0, 1).contiguous()

        hc_fn_t_scaled = _pack_hc_fn_t_scaled(hc_fn, hc_scale)
        hc_base_tile = _pack_hc_base_tile(hc_base)
        scaler_bf16 = torch.ones(TILE, TILE, dtype=torch.bfloat16)
        w_baked = (norm_gamma[:, None] * w_lmhead).to(torch.bfloat16)

        self._final_hc_fn_t_scaled_tt = ttnn.as_tensor(
            hc_fn_t_scaled.contiguous(), dtype=ttnn.bfloat16, **rep)
        self._final_hc_base_tile_tt = ttnn.as_tensor(
            hc_base_tile.contiguous(), dtype=ttnn.bfloat16, **rep)
        self._final_scaler_tt = ttnn.as_tensor(
            scaler_bf16.contiguous(), dtype=ttnn.bfloat16, **rep)
        self._final_w_baked_tt = ttnn.as_tensor(
            w_baked.contiguous(), dtype=ttnn.bfloat16, **rep)

    def _alloc_kv_caches(self, mesh):
        """KV caches sized to Lk-Dsparse's hardcoded KV_CACHE_SIZE_PAD=128.

        Lk-Dsparse and Lk-D-comp hardcode the cache to 128 slots
        (window-only attention). For bringup we allocate replicated
        [1, 1, 128, head_dim]. Future work re-introduces seq-sharding
        once the kernel can operate on a sharded view.
        """
        args = self.args
        KV_CACHE_SIZE_PAD = 128
        T_local = KV_CACHE_SIZE_PAD

        self._kv_cache_tt = []
        self._kv_slot_tt = []
        for _ in range(args.n_layers):
            cache = ttnn.from_torch(
                torch.zeros(1, 1, T_local, args.head_dim, dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            slot = ttnn.from_torch(
                torch.zeros(1, dtype=torch.int32),
                device=mesh, dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            self._kv_cache_tt.append(cache)
            self._kv_slot_tt.append(slot)

        # Softmax scaler tile (bf16 ones) shared by every Lk-Dsparse call.
        self._sm_scaler_tt = ttnn.from_torch(
            torch.ones(32, 32, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # Window topk lookup table: [max_seq_len, win] bf16, replicated.
        # Mirrors legacy DeviceAttention._win_idxs_padded_tt: each row is
        # the window-topk index vector for a specific start_pos, with -1
        # sentinels for slots that have not been filled yet (forward-pad
        # at p < window_size, wraparound at p >= window_size). Without
        # sentinels, attention drowns in zero-init cache slots → gibberish.
        # Per-step pick: ttnn.embedding(start_pos_tt, table) → typecast int32.
        win = args.window_size
        if win > 256:
            raise ValueError(
                f"window_size {win} > 256: bf16 cannot represent all integer "
                "index values exactly; switch the lookup table to int32 + a "
                "different gather op when this fires."
            )
        full_table = _build_window_topk_table(win)
        win_idxs_padded = full_table[
            torch.tensor(
                [_window_topk_row_for_pos(p, win)
                 for p in range(args.max_seq_len)],
                dtype=torch.long,
            )
        ].to(torch.bfloat16).contiguous()
        self._win_idxs_padded_tt = ttnn.from_torch(
            win_idxs_padded,
            device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    # -----------------------------------------------------------------
    # Per-step host upload + step body
    # -----------------------------------------------------------------

    def _pre_stage(self, token_id: int, pos: int):
        """Per-step host->device uploads. Writes input_ids, start_pos, and
        per-layer KV slot indices into pre-allocated buffers. Must run
        BEFORE `execute_trace` since the trace body has no host I/O."""
        ids = ttnn.from_torch(
            torch.tensor([[token_id]], dtype=torch.int32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._input_ids_upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(ids, self._input_ids_tt)
        sp = ttnn.from_torch(
            torch.tensor([[pos]], dtype=torch.int32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._input_ids_upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(sp, self._start_pos_tt)
        # KV slot per layer: window slot for the attn cache.
        win = self.args.window_size
        slot_int = pos % win
        slot_h = ttnn.from_torch(
            torch.tensor([slot_int], dtype=torch.int32),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._input_ids_upload_mapper,
        )
        for layer_slot in self._kv_slot_tt:
            ttnn.copy_host_to_device_tensor(slot_h, layer_slot)
        # Indexer / compressor per-layer slot uploads (state_slot, emit_slot).
        # DeviceCompressor.pre_stage_decode handles the full computation.
        for layer in self.transformer.layers:
            attn = layer.attn
            if attn.compress_ratio != 4:
                continue
            attn._device_compressor.pre_stage_decode(start_pos=pos)
            attn._device_indexer.dc.pre_stage_decode(start_pos=pos)
        # Indexer t_active mask: number of compressed-token slots active
        # at the current position. T_active = (pos + 1) // INDEX_RATIO,
        # but only positions reachable up to INDEX_T_PAD are valid.
        t_active_int = (pos + 1) // INDEX_RATIO
        t_active_int = min(max(t_active_int, 0), INDEX_T_PAD)
        # The mask-build kernel reads a [1, 1, BUCKET=128] vector where
        # each entry holds the same t_active scalar (broadcast inside the
        # kernel). Stage as a fully-populated row.
        t_active_row = torch.full(
            (1, 1, INDEX_TOPK_BUCKET), t_active_int, dtype=torch.int32)
        t_active_h = ttnn.from_torch(
            t_active_row, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._input_ids_upload_mapper,
        )
        ttnn.copy_host_to_device_tensor(t_active_h, self._t_active_tt)

    def _step_body(self):
        """Pure-device per-step body: embed → L0 (first hc_pre → wq_a) →
        block loop → Final. Returns the per-chip top-1 token tensor.

        No host I/O inside this function: every per-step value comes from
        the pre-uploaded `_input_ids_tt` / `_start_pos_tt` / `_kv_slot_tt`.
        Safe to wrap in `ttnn.begin_trace_capture` / `end_trace_capture`.
        """
        sb = self._step_buffers
        a_tt = sb.a_tt
        a_tt_next = sb.a_tt_next

        # 1. Token embedding: input_ids -> [B, S, DIM] bf16, replicated.
        with _phase("embed"):
            embed_tt = self.transformer.embed.forward(self._input_ids_tt)

        # 2. L0: embed_prep + hc_pre(layer 0) + attn_norm + wq_a. Writes
        # the residual stream into a_tt, layer 0's wq_a partial into
        # wq_a_out, and the post-norm bf16 tile into x_post_norm (used as
        # Lk-C input for layer 0; layers 1+ derive x_post_norm via Lk-A's
        # x_pre_norm + a device rms_norm).
        with _phase("l0"):
            l0_lk = self._layer_kernels[0]
            self._l0_kernel(
                embed_tt, self._l0_wq_a_tt,
                a_tt, sb.wq_a_out,
                x_post_norm_bf16_out=sb.x_post_norm,
            )

        # Optional layer-bisect: env DEBUG_NUM_LAYERS=N runs only the
        # first N transformer blocks (N=0 skips all blocks). Useful for
        # finding the first broken layer during bringup.
        _debug_env = os.environ.get("DEBUG_NUM_LAYERS")
        _debug_layers = (
            self._layer_kernels[:int(_debug_env)]
            if _debug_env is not None and _debug_env != ""
            else self._layer_kernels)

        # 3. Block loop.
        for layer_idx, lk in enumerate(_debug_layers):
            a_tt, a_tt_next = (
                self._block_forward_mega(
                    layer_idx, lk, a_tt, a_tt_next,
                    self._start_pos_tt, self._input_ids_tt, sb),
                a_tt,                       # ping-pong residual buffers
            )

        # 4. Final head via legacy DeviceLMHead: hc_combiner + final RMSNorm
        # + column-parallel lm_head matmul + per-chip top-1. Returns
        # (idxs_tt, vals_tt); _argmax_pull globalizes them. The mega-fused
        # Final kernel produces gibberish output, so we use the blessed
        # ttnn op chain here while that's debugged.
        with _phase("final"):
            top_idx, top_val = self._device_lm_head.forward_argmax_device(
                a_tt, num_tokens=1)
        return top_val, top_idx

    # -----------------------------------------------------------------
    # Public step / prefill / generate (override the legacy paths)
    # -----------------------------------------------------------------

    @torch.inference_mode()
    def step_decode(self, token_id: int, pos: int) -> int:
        is_emit = (pos + 1) % 4 == 0
        # `_emit_now` is read by `_block_forward_mega` as a host-side bool
        # so the indexer/compressor emit-branch is included or excluded
        # from the captured trace per (no_emit / emit) variant.
        self._emit_now = is_emit
        self._pre_stage(token_id, pos)

        if self._trace_warmup_remaining > 0:
            self._trace_warmup_remaining -= 1
            top_val, top_idx = self._step_body()
            return self._argmax_pull(top_val, top_idx)

        target = self._trace_emit if is_emit else self._trace_no_emit
        if target is None:
            kind = "emit" if is_emit else "no-emit"
            print(f"[trace] capturing {kind} decode body at pos={pos} ...")
            t0 = time.time()
            trace_id = ttnn.begin_trace_capture(self._mesh, cq_id=0)
            top_val, top_idx = self._step_body()
            ttnn.end_trace_capture(self._mesh, trace_id, cq_id=0)
            target = (trace_id, top_val, top_idx)
            if is_emit:
                self._trace_emit = target
            else:
                self._trace_no_emit = target
            print(f"[trace] captured {kind} trace in {time.time() - t0:.1f}s")
            if (self._trace_no_emit is not None
                    and self._trace_emit is not None
                    and not self._traces_warm):
                _phase_snapshot_at_trace_warm()
                self._traces_warm = True
            return self._argmax_pull(top_val, top_idx)

        trace_id, top_val, top_idx = target
        ttnn.execute_trace(self._mesh, trace_id, cq_id=0, blocking=True)
        return self._argmax_pull(top_val, top_idx)

    def _argmax_pull(self, top_val, top_idx) -> int:
        """Pull per-chip top-1 (column-parallel sharded vocab) and return the
        global argmax. Delegates to `DeviceLMHead.argmax_pull` so the chip
        index → vocab offset stays consistent with the lm_head sharding."""
        return self._device_lm_head.argmax_pull(top_idx, top_val)

    @torch.inference_mode()
    def prefill(self, tokens: list[int]) -> int:
        """Greedy prefill via repeated `step_decode`. The trace machinery
        handles the warmup window. Prints the per-step predicted next-token
        so divergence in prefill (vs decode) is visible."""
        nxt = -1
        for i, tok in enumerate(tokens):
            nxt = self.step_decode(tok, i)
            try:
                tok_str = self.tokenizer.decode([nxt])
            except Exception:
                tok_str = "?"
            print(f"[prefill] pos={i} input_tok={tok} -> next={nxt} {tok_str!r}")
        return nxt

    @torch.inference_mode()
    def generate(self, tokens: list[int], max_tokens: int = 32,
                 warmup_tokens: int = 0):
        with _phase("prefill"):
            nxt = self.prefill(tokens)
        pos = len(tokens)
        for i in range(max_tokens):
            yield nxt
            with _phase("decode_step"):
                nxt = self.step_decode(nxt, pos)
            pos += 1
            if warmup_tokens > 0 and i + 1 == warmup_tokens:
                _PHASE_ACCUM.clear()
                _PHASE_COUNTS.clear()
                print(f"[phase] warmup complete after {warmup_tokens} tokens")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Mega-kernel inference for DeepSeek-V4-Flash")
    parser.add_argument("--repo", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--weights-cache",
                        default=os.environ.get(
                            "DS_WEIGHTS_CACHE",
                            "/tmp/deepseek_v4_flash_cache/state_dict.pt"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(min(32, os.cpu_count() or 8))

    print(f"[phase] building model (max_seq_len={args.max_seq_len}) ...")
    t0 = time.time()
    model = MegaModel.from_hf(args.repo, max_seq_len=args.max_seq_len)
    print(f"[phase] model instantiated in {time.time() - t0:.1f}s")

    print("[phase] loading weights ...")
    t0 = time.time()
    model.load_weights(cache_path=args.weights_cache or None)
    print(f"[phase] weights loaded in {time.time() - t0:.1f}s")

    print("[phase] opening 4x8 mesh ...")
    t0 = time.time()
    # Mega kernels (esp. Lk-B fused rms+ksplit) have ~98KB programs that
    # don't fit in the default ~70KB kernel-config buffer. Trade L1 to
    # grow it, per mega/README.md.
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    _default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(4, 8),
        trace_region_size=200_000_000,
        worker_l1_size=_default_l1 - 128 * 1024,
    )
    print(f"[phase] mesh opened in {time.time() - t0:.1f}s")

    print("[phase] offloading to mesh via mega kernels ...")
    t0 = time.time()
    model.offload_to_mesh(mesh)
    print(f"[phase] offload complete in {time.time() - t0:.1f}s")

    print("[phase] encoding prompt ...")
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().rstrip("\n")
    tokens = model.tokenizer.encode(prompt)
    print(f"[phase] prompt = {prompt!r}, {len(tokens)} tokens")

    print("[phase] generating ...")
    out_tokens = []
    for tok in model.generate(tokens, max_tokens=args.max_tokens,
                              warmup_tokens=args.warmup_tokens):
        out_tokens.append(tok)
        piece = model.tokenizer.decode([tok])
        print(piece, end="", flush=True)
    print()
    print(f"[debug] token ids: {out_tokens}")
    print(f"[debug] full decode: {model.tokenizer.decode(out_tokens)!r}")

    if model._traces_warm:
        post_accum, post_counts = _phase_postwarm()
        ds_total = post_accum.get("decode_step", 0.0)
        ds_count = post_counts.get("decode_step", 0)
        if ds_count > 0 and ds_total > 0:
            print(f"[phase] trace replay: {ds_count} tokens in "
                  f"{ds_total:.2f}s ({ds_count / ds_total:.2f} tok/s)")

    if model._traces_warm:
        print("\n[timing] pre-warm phases:")
        print(_phase_report(_PHASE_PREWARM_ACCUM, _PHASE_PREWARM_COUNTS))
        post_accum, post_counts = _phase_postwarm()
        print("\n[timing] post-warm phases (trace replay only):")
        print(_phase_report(post_accum, post_counts))
    else:
        print("\n[timing] per-phase breakdown:")
        print(_phase_report())

    _close_mesh(mesh)


if __name__ == "__main__":
    main()
