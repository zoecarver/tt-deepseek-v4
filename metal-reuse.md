# tt-metal kernels / utilities reusable for DeepSeek-V4-Flash

Inventory of existing DeepSeek work in `../tt-metal` that can be pulled into V4 bring-up. Grouped by implementation, then by op.

Three DeepSeek implementations exist:
- **V3** — `models/demos/deepseek_v3/` — high-level Python pipeline, module-based (convert_weights → model_config → forward). Runs on any mesh (TG, DUAL, QUAD, T3K, N300, N150).
- **V3-B1 ("blitz")** — `models/demos/deepseek_v3_b1/` — Blackhole-specific, single-core micro-ops + unified-kernel framework. Decode path mature; prefill incomplete.
- **V3-D-P** — `models/demos/deepseek_v3_d_p/` — placeholder for disaggregated prefill.

**"Blaze":** there is no `blaze` DeepSeek dir on `main`. A `remotes/origin/blaze-metal` branch exists with GPU-style prefill MoE and CCL optimizations targeting Blackhole. Worth checking out separately if we need prefill-MoE on multi-chip.

**Quiet Box target:** 4 Blackhole chips — closest match in the matrix is the `TG (4x8)` mesh shape from V3, and the V3-B1 micro-ops (Blackhole-only). Pick V3 path for portability, lift V3-B1 micro-ops in later for perf.

---

## MLA (Multi-Latent Attention) — critical; V4 also uses MLA

### MLA1D
- **What**: Row-distributed MLA (qk_nope, qk_rope, kv_lora projections) with block-quantized weights and async all-gather / reduce-scatter for KV distribution.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/mla/mla1d.py` (class `MLA1D`)
- **How to use**:
  ```python
  from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
  weight_cfg = MLA1D.convert_weights(hf_config, state_dicts, out_path, mesh_device)
  model_cfg  = MLA1D.prefill_model_config(hf_config, mesh_device)   # or decode_model_config
  state      = MLA1D.create_state(hf_config, mesh_device)
  run_cfg    = run_config(model_cfg, weight_cfg, state)
  out = MLA1D.forward_prefill(x, run_cfg)                           # or forward_decode
  ```
- **HW**: any mesh (TG, DUAL, QUAD, T3K tested).
- **Notes**: block-quantized (`weight_block_size` from hf_config); consumes YaRN RoPE from `rope.py`.

### MLA2D
- **What**: 2D-mesh MLA; mostly delegates to MLA1D.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/mla/mla2d.py`
- **HW**: Galaxy (TG+).

### Flash MLA (Blackhole single-core decode)
- **What**: Single-core decode attention w/ paged KV cache, optimized for Blackhole NOC0. Hard-coded 8-S-block × 8-core grid.
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` (`FlashMLAOptimalGridNOC0`)
- **How to use**: `op.execute_flash_mla_decode(...)`
- **HW**: **Blackhole only** (strict NOC0 tile layout).
- **Notes**: single-tile per core; soft-token mode; DRAM-banked paged KV.

---

## RoPE (YaRN variant — V3/V4)

### RotarySetup
- **What**: YaRN-scaled RoPE with pre-computed cos/sin, batch-sharded + interleaved layouts.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/rope.py` (class `RotarySetup`, `get_cos_sin_matrix`)
- **How to use**:
  ```python
  from models.demos.deepseek_v3.tt.rope import RotarySetup, get_cos_sin_matrix
  cos, sin = get_cos_sin_matrix(hf_config)
  rope = RotarySetup(device, batch_per_row, hf_config)
  rot_idxs = rope.get_rot_idxs(position_ids)
  rot_mats = rope.get_rot_mats(rot_idxs)                  # decode
  rot_mats = rope.get_rot_mats_table(seq_len)             # prefill
  ```
- **HW**: any.
- **Notes**: YaRN params (`beta_fast`, `beta_slow`, `mscale`, `mscale_all_dim`) read from hf_config.

### RopeSingleCore (V3-B1)
- **What**: Single-core per-token RoPE for decode.
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/rope/op.py`
- **How to use**: `RopeSingleCore.op(x, cos, sin, position_ids, output, ...)`
- **HW**: Blackhole (HEIGHT_SHARDED L1).

---

## RMSNorm

### RMSNorm (V3)
- **What**: Distributed RMSNorm, tile-width-aligned, per-device gamma sharding.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/rms_norm/rms_norm.py`
- **How to use**: `RMSNorm.forward_prefill(x, cfg)` / `.forward_decode(...)`.
- **HW**: any.
- **Notes**: uses `COMPUTE_KERNEL_CONFIG_LOFI`.

### RMSNormSingleCore (V3-B1)
- **What**: Single-core RMSNorm, optional FP32 accum and fast rsqrt.
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/rmsnorm/op.py`
- **How to use**: `RMSNormSingleCore.op(x, gamma, out, epsilon, fp32_dest_acc_en, rsqrt_fast_approx)`
- **HW**: Blackhole single-core.

---

## MoE

### MoE (V3 distributed)
- **What**: Expert dispatch + gating + reduction; all-to-all + all-gather/reduce-scatter for routing.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/moe.py`
- **How to use**:
  ```python
  from models.demos.deepseek_v3.tt.moe import MoE
  weight_cfg = MoE.convert_weights(hf_config, (sd,), out_path, mesh_device)
  model_cfg  = MoE.prefill_model_config(hf_config, mesh_device)
  shared     = MoE.create_shared_state(hf_config, mesh_device)
  out = MoE.forward_prefill(x, run_config(model_cfg, weight_cfg, shared))
  ```
- **HW**: any (1D TP; `mesh_device.shape[1] == 8`).
- **Notes**: uses `AllToAllDispatch` / `AllToAllCombine`; sparse top-k.

### MoEGate
- **What**: Top-K gating + e-score correction + optional sigmoid + routed scaling.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/moe_gate.py`
- **HW**: any.
- **Notes**: `TopKFallback` (bitonic) path for large token counts.

### Experts
- **What**: Stacked shared + routed experts, per-device sharded; mixed BF8 / BF4 quant.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/experts.py`
- **HW**: any (requires `n_routed_experts % num_devices == 0`).
- **Notes**: BF8 `up_proj`, BF4 `gate_proj` / `down_proj`.

### DeepseekMoeGate (V3-B1 single-core)
- **What**: sigmoid → +bias → top-2 groups → top-8 across groups → normalize.
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/op.py`
- **How to use**: `DeepseekMoeGateSingleCore.op(x, bias, out, in_idxs, out_idxs, eps, scaling_factor, enable_sigmoid)`
- **HW**: Blackhole single-core.

---

## Matmul variants

### Linear (V3)
- **What**: `ttnn.linear` wrapped with per-layer memory + program configs for prefill vs decode.
- **Source**: `LinearConfig` in `../tt-metal/models/demos/deepseek_v3/utils/config_dataclass.py`
- **How to use**: `ttnn.linear(x, **cfg["w1_linear"])`.
- **HW**: any.

### Matmul (V3-B1 single-core)
- **What**: `[1..32, K] @ [K, N]` single-core matmul w/ fused sigmoid/silu, MOP-looped over K.
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/matmul/op.py`
- **How to use**: `Matmul.op(A, B, out, fp32_dest_acc_en, transpose, fused_activation)`
- **HW**: Blackhole.

### DRAM-streaming matmul (V3-B1)
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp`
- **HW**: Blackhole.

### KN-sliced matmul (V3-B1)
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/kn_sliced_matmul/op.py`
- **HW**: Blackhole.

---

## Quantization

### dequantize_tensor
- **What**: Block-wise inverse-scale dequantization; host-side torch util.
- **Source**: `../tt-metal/models/demos/deepseek_v3/utils/dequantize.py`
- **How to use**:
  ```python
  from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor
  deq = dequantize_tensor(q_weights, inv_scales, block_shape)
  ```
- **Notes**: block shape from `hf_config.quantization_config["weight_block_size"]`; pads non-divisible dims.

**Weight dtype convention (V3):**
- `ttnn.bfloat8_b` for experts `up_proj`.
- `ttnn.bfloat4_b` for experts `gate_proj` / `down_proj`, MLA projections, LM head.
- `ttnn.bfloat16` for RMSNorm gamma and RoPE tables.

---

## CCL / distributed

### CCL helper
- **What**: Per-axis semaphore pool for all_gather / reduce_scatter; link-count management.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/ccl.py`
- **How to use**:
  ```python
  from models.demos.deepseek_v3.tt.ccl import CCL
  ccl = CCL(mesh_device)
  cfg2 = ccl.populate_all_gather_runtime_args(static_cfg)
  y = ttnn.experimental.all_gather_async(x, **cfg2)
  ```
- **HW**: any multi-device.
- **Notes**: up to 4 links/axis; Ring or Linear topology (`USE_TORUS_MODE` env).

### AllGatherAsync / ReduceScatterAsyncMinimal / AllToAllDispatch / AllToAllCombine
- **Source**: config dataclasses in `../tt-metal/models/demos/deepseek_v3/utils/config_dataclass.py`
- **HW**: any.

### D2D exchange (V3-B1)
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/d2d_exchange/op.py`
- **HW**: Blackhole multi-chip.

---

## KV cache

### KvCacheConfig (V3)
- **What**: Paged cache shape `(num_blocks, num_heads, block_size, kvpe_dim)`.
- **Source**: `../tt-metal/models/demos/deepseek_v3/utils/config_dataclass.py` (`KvCacheConfig`)

### KV cache update (V3-B1)
- **What**: Single-core KV write-back + cache-index advance; supports split K / V (MLA compressed cache + V separate).
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/kv_cache_update/op.py`
- **HW**: Blackhole.

### PagedAttentionConfig (reference, not yet wired for DeepSeek)
- **Source**: `../tt-metal/models/tt_transformers/tt/common.py`

---

## SDPA

### Flash MLA decode + pre/post-SDPA (V3-B1)
- **What**: Multi-block S-attention, separate reduce phase, prefetch-pipelined.
- **Sources**:
  - `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`
  - `../tt-metal/models/demos/deepseek_v3_b1/unified_kernels/sdpa_reduce_worker.hpp`
  - `../tt-metal/models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/`
  - `../tt-metal/models/demos/deepseek_v3_b1/fused_ops/post_sdpa/`
- **HW**: Blackhole.

### ttnn.sdpa (V3 prefill)
- **What**: Generic library SDPA used in V3 prefill.
- **HW**: any.

---

## Sampling / generation

### SamplingOp (V3-B1)
- **What**: Argmax (k=1) + optional top-p, multi-core reduce to final core; fabric-capable.
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/sampling/op.py`
- **How to use**: `SamplingOp.op(scores, indices, out, k=1, p=1.0, final_core_coord=...)`
- **HW**: Blackhole.

### DeepseekGenerator (V3)
- **What**: End-to-end prefill + decode loop around RowBatchedModel + LMHead.
- **Source**: `../tt-metal/models/demos/deepseek_v3/tt/generator.py`
- **How to use**:
  ```python
  from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
  with DeepseekGenerator(mesh_device=dev, model_path=p, cache_dir=c) as gen:
      out = gen.generate(prompts, max_new_tokens=32)
  ```
- **HW**: any.
- **Notes**: teacher-forcing accuracy mode; random-weight smoke-test hook; profiling/trace hooks.

---

## Infrastructure utilities

### config_helpers
- **Source**: `../tt-metal/models/demos/deepseek_v3/utils/config_helpers.py`
- **Key fns**: `shard_and_save`, `dequantize`, `find_largest_divisor`, plus pre-tuned `COMPUTE_KERNEL_CONFIG_HIFI2` / `_LOFI`.

### run_config
- **What**: Merges model_config + weight_config + model_state into a ready-to-call RunConfig; auto-loads tensors onto mesh.
- **Source**: `../tt-metal/models/demos/deepseek_v3/utils/run_config.py`
- **How to use**:
  ```python
  from models.demos.deepseek_v3.utils.run_config import create_run_config
  run_cfg = create_run_config(model_cfg, weight_cfg, state)
  ```

### test_utils
- **Source**: `../tt-metal/models/demos/deepseek_v3/utils/test_utils.py`
- **Key**: `system_name_to_mesh_shape()` — `"TG"→(4,8)`, `"DUAL"→(8,8)`, `"QUAD"→(16,8)`, etc.

### Host I/O (V3-B1)
- **What**: Socket-based H2D token ingress / D2H logits egress (PCIe-aligned FIFO pages). Prefill = `DEVICE_PULL`, decode = `HOST_PUSH`.
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/model.py`, `../tt-metal/models/demos/deepseek_v3_b1/micro_ops/host_io/op.py`
- **HW**: Blackhole.

### UnifiedKernelDescriptor (V3-B1 micro-op framework)
- **What**: Single `.cpp` → per-RISC / per-core specialization via compile-time args (dead-code-eliminated).
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/unified_kernel_descriptor.py`
- **HW**: Blackhole.

### V3-B1 utils
- **Source**: `../tt-metal/models/demos/deepseek_v3_b1/utils.py`
- **Key fns**: `float_to_uint32`, `float_to_bfloat16_packed` for kernel runtime args.

---

## Summary — pick-list for V4 bring-up

| Component   | V3 (portable)                          | V3-B1 (Blackhole)                         |
|-------------|----------------------------------------|-------------------------------------------|
| MLA         | `tt/mla/mla1d.py`                      | `micro_ops/flash_mla/`                    |
| RoPE (YaRN) | `tt/rope.py`                           | `micro_ops/rope/op.py`                    |
| RMSNorm     | `tt/rms_norm/`                         | `micro_ops/rmsnorm/op.py`                 |
| MoE gate    | `tt/moe_gate.py`                       | `micro_ops/deepseek_moe_gate/op.py`       |
| Experts     | `tt/experts.py`                        | (fused in MoE)                            |
| Matmul      | `ttnn.linear` + `LinearConfig`         | `micro_ops/matmul/op.py`                  |
| SDPA        | `ttnn.sdpa`                            | `micro_ops/flash_mla/` (decode)           |
| Sampling    | —                                      | `micro_ops/sampling/`                     |
| KV cache    | `KvCacheConfig`                        | `micro_ops/kv_cache_update/`              |
| CCL         | `tt/ccl.py`                            | fabric (blaze-metal branch)               |
| Host I/O    | —                                      | `model.py` + `micro_ops/host_io/`         |
| Generator   | `tt/generator.py`                      | `model.py`                                |

### Gaps / risks for V4
- No paged-attention implementation wired into DeepSeek (reference stub in `tt_transformers/common.py`).
- V3-B1 prefill is incomplete (host I/O only implemented for decode).
- Prefill-MoE multi-chip optimizations live on `remotes/origin/blaze-metal`; may need cherry-pick if we hit CCL bottlenecks on Quiet Box.
- V4-specific deltas TBD: GQA head-grouping changes vs V3, possible FP4 quant (current: BF8 / BF4).
