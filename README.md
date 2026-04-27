# DeepSeek-V4-Flash on Tenstorrent

DeepSeek-V4-Flash inference on Tenstorrent hardware (Quiet Box, Galaxy). The
full model — weight loading, sharding, attention, MoE routing, decode loop —
lives in a single `inference.py` with no external project dependencies. Hot
ops are tt-lang fused kernels; everything else is ttnn.

## Status

Running on a 4×8 BH Galaxy, bf16 activations end-to-end, bfp4_b for routed
experts (offload-time re-quantized; see `_dequant_to_native_bfp4` in
`inference.py`). Coherent text out, quality acceptable.

```
[debug] full decode: ' In dwelling, be close to the land. In meditation, go deep in the heart heart.. In dealing with others, be gentle and kind. In speech, be true. In ruling, be just. In business, be competent. In action'
[phase] trace replay: 32 tokens in 21.00s (1.52 tok/s)

[timing] per-phase breakdown (inclusive; sort by total):
phase                      total(s)    n    avg(ms)
  decode_step                 21.00    32      656.3
  pre_stage                    0.63    31       20.2

[timing] pre-warm phases (untraced steps before trace capture, expected to be higher and not totally representative of traced run):
phase                      total(s)    n    avg(ms)
  decode_step                 33.43    15     2228.5
  blocks                      28.88    15     1925.6
  block.attn                  17.90   645       27.8
  block.ffn                    6.61   645       10.2
  attn.topk                    4.33   645        6.7
  attn.topk.indexer            4.13   315       13.1
  attn.q                       3.96   645        6.1
  attn.o                       2.92   645        4.5
  attn.kv                      2.58   645        4.0
  block.hc_pre                 2.24  1290        1.7
  attn.compress                2.04   615        3.3
  attn.sparse                  1.81   645        2.8
  block.hc_post                1.72  1290        1.3
  pre_stage                    0.61    16       38.3
  block.norm                   0.37  1290        0.3
  attn.kv_update               0.09   645        0.1
  head                         0.05    15        3.6
  head.norm_and_argmax_device  0.05    15        3.6        # tt-lang kernel 7x speedup expected 
  embed                        0.02    15        1.5
```

## Architecture (where tt-lang vs ttnn)

Decode step (per layer x 43, except hash gates use a host MoE fallback):

```
embed                                                   ttnn (lookup + replicate)
└── for each block:
    ├── block.norm           (RMSNorm)                  TT-LANG (rmsnorm)
    ├── block.hc_pre         (MHC head-channel pre)
    │     ├── pre_norm_fn / pre_norm_fn_ksplit          TT-LANG
    │     ├── pre_split_mixes                           TT-LANG
    │     ├── sinkhorn (HC iterations)                  TT-LANG
    │     └── pre_apply_mix / pre_apply_mix_h           TT-LANG
    ├── block.attn
    │     ├── attn.q          (wq_a -> norm -> wq_b -> rsqrt-norm -> rotary)   ttnn
    │     ├── attn.kv         (wkv -> norm -> rotary -> act_quant)
    │     │     └── act_quant_block (KV nope quant round-trip)                 TT-LANG
    │     ├── attn.compress   (compressor + emit branch)
    │     │     ├── compressor_slot_shift (state buffer rotation)              TT-LANG
    │     │     └── compressor_softmax_sum_norm (slice/concat/softmax/mul/sum/RMSNorm)  TT-LANG
    │     ├── attn.topk       (window topk + indexer + compress-K topk)        ttnn
    │     ├── attn.sparse     (gather + scaled DPA + sink-concat softmax)      ttnn (paged_flash_mla)
    │     └── attn.o          (block-diagonal wo_a + wo_b)                     ttnn
    ├── block.hc_post
    │     └── post                                                             TT-LANG
    └── block.ffn (MoE)
          ├── DeviceMoEGate (sigmoid score -> topk)                             ttnn (custom gate)
          ├── routed experts (Path D: grouped fp4 -> bfp4_b matmul + mask)      ttnn matmul on bfp4_b
          ├── shared expert (w1, w2, w3 matmuls + SwiGLU)                       ttnn (+ swiglu kernel below)
          │     └── swiglu                                                      TT-LANG
          └── all_reduce across mesh                                            ttnn CCL
head
  ├── final RMSNorm                                                             TT-LANG (rmsnorm)
  ├── DeviceLMHead (vocab-sharded matmul)                                       ttnn
  └── per-chip topk(k=1) + 4-byte token id pull                                 ttnn (host pull only)
```

The whole decode step is captured as a ttnn trace. Two traces are
recorded — one for "no compressor emit" and one for the every-4th-step
emit branch — and replayed in alternation. After warmup, replay is the
only host work besides the 4-byte token id pull.

## tt-lang kernels in use

All kernels are inlined into `inference.py`; standalone copies in
`tt-lang-kernels/` for reference and microbench.

| kernel | site | notes |
|---|---|---|
| `rmsnorm` | block pre-norm, post-norm, head norm, every Q/KV norm | row-tile streaming |
| `pre_norm_fn`, `pre_norm_fn_ksplit` | MHC pre | ksplit is multi-core for the wide K shapes |
| `pre_split_mixes` | MHC pre | unpack mhc=4 mix channels |
| `sinkhorn` | MHC pre | HC log-domain Sinkhorn iterations, fp32 dest acc |
| `pre_apply_mix`, `pre_apply_mix_h` | MHC pre | `_h` is the h-sharded multi-core variant |
| `post` | MHC post | mhc reduction back to dim |
| `act_quant_block` | attn.kv | per-block max-abs scale round-trip on KV nope |
| `compressor_slot_shift` | attn.compress emit | tile matmul-based 4-buffer rotation; replaces 16 ttnn dispatches |
| `compressor_softmax_sum_norm` | attn.compress emit | fuses slice/concat view + softmax + weighted-sum + RMSNorm |
| `swiglu` | shared expert | fused silu(y1) * y3 |
| `attention_matmul`, `fused_matmul_rmsnorm`, `matmul_softmax` | available as building blocks; not all currently wired into the hot path | see `tt-lang-kernels/` |

## Routed-expert weights: fp4 -> bfp4_b

DeepSeek-V4-Flash ships routed-expert weights as fp4 e2m1 + per-K-block e8m0
scale. We store them on device as native `ttnn.bfloat4_b` and let
`ttnn.matmul(x_bf16, w_bfp4_b)` do the unpack inside the matmul kernel —
no per-call dequant.

Pipeline (one-time):

1. **Offline preprocess** (`scripts/preprocess_routed_experts.py`): for each
   fp4 nibble, write a bf16 lattice value `0.25 * mag_index * sign` such
   that `ttnn.from_torch(dtype=bfloat4_b)` round-trips bit-for-bit. Output
   is one `.tensorbin` per (layer, weight, scale), pre-sharded for the
   target mesh. Lossless.
2. **Offload-time re-quant** (`_dequant_to_native_bfp4` in `inference.py`):
   load the lattice .tensorbin, typecast bf4->bf16, algebraic remap from
   bfp4 lattice {0, ±0.25, ..., ±1.75} to fp4 magnitudes {0, ±0.5, ..., ±6},
   multiply by the e8m0 scale, then `typecast(bf16, bfloat4_b)` to absorb
   the scale into bfp4's intrinsic per-face-row exponent. ~13s at startup
   for all 40 MoE layers; e8m0 scale tensors are dropped after.
3. **Hot path**: single `ttnn.matmul`, no scratch.

Per-element bit layout: both formats are 4 bits/element with the sign at
bit 3. fp4_e2m1 spends `2 exp + 1 mantissa` per element; bfp4_b spends
`3 mantissa` per element with an 8-bit exponent shared across each
16-element face row. The bijection works because both formats encode 8
magnitudes in the lower 3 bits in the same bit positions; the algebraic
remap translates the magnitude *interpretation* (logarithmic vs linear).

Quality: greedy decode of "the quick brown fox" matches bit-exact for 5
tokens, then diverges to a sensible continuation. Expected from the per-
face-row vs per-K-block scale granularity change; perplexity not measured.

## Where tt-lang is NOT used (yet)

These run on ttnn op chains and are the next candidates:

- **attn.q, attn.kv, attn.o** — a few slices/reshapes/rotaries/concats per call. ttnn `paged_flash_mla` covers the sparse-attn body.
- **attn.topk** — the topk reduce + index gather is ttnn (`ttnn.topk`); writing topk in tt-lang is non-trivial.
- **MoE routed-expert matmul** — `ttnn.matmul(x_bf16, w_bfp4_b)` directly. The matmul kernel itself is ttnn; its bfp4 unpack is internal. tt-lang can't currently fuse mixed-dtype matmul (bf16 act + bfp4_b weight).
- **all_reduce / reduce_scatter** — ttnn CCL.
- **`DeviceMoEGate`, `DeviceLMHead`** — ttnn op chains; argmax via `ttnn.topk(k=1)` with pre-allocated UINT16 indices for trace stability. A tt-lang two-pass streaming argmax kernel (~7× faster than `ttnn.topk(k=1)`: 2× faster than `ttnn.argmax+max`, which itself is 3× faster than `ttnn.topk(k=1)`) is a drop-in candidate.

## Layout

```
inference.py                    standalone, all kernels + decode loop
tt-lang-kernels/                reference standalone kernel files + microbench harnesses
scripts/
  preprocess_routed_experts.py  offline fp4 -> bfp4_b lattice cache builder (one-time)
  device_moe_gate.py            standalone MoE gate kernel test
  test_*.py                     per-op PCC tests
  prompts/                      sample inputs
device-compressor-indexer/      compressor + indexer device-port reference cases
TileKernels/                    DeepSeek's reference tilelang kernels (read-only; tilelang doesn't run on TT)
briefs/                         design notes (mesh utilization plan, tracing plan, MoE gate offload)
```

## Dependencies

**Code**: `ttnn` and `ttl` (tt-lang) Python packages, plus `torch`. No
project-internal deps; `inference.py` is self-contained.

**Weights**:

- `state_dict.pt` — HF DeepSeek-V4-Flash export with fp4 routed-expert
  weights + e8m0 scales and bf16/fp8 everything else.
- `state_dict_bfp4_routed/` — preprocessed routed-expert cache, produced by
  `scripts/preprocess_routed_experts.py` from `state_dict.pt`. One
  `.tensorbin` per (layer, weight) pre-sharded for the target mesh shape.
  Pointed to by `$DS_ROUTED_EXPERT_CACHE`.

**Hardware**: a Tenstorrent mesh device. Tested on 1×4 BH (Quiet Box) and
4×8 BH (Galaxy).

## Running

One-time cache build:

```
python scripts/preprocess_routed_experts.py \
  --state-dict /path/to/state_dict.pt \
  --out /path/to/state_dict_bfp4_routed \
  --mesh-rows 4 --mesh-cols 8 --validate
```

Then:

```
export DS_ROUTED_EXPERT_CACHE=/path/to/state_dict_bfp4_routed
python inference.py --prompt scripts/prompts/<prompt>.txt
```
