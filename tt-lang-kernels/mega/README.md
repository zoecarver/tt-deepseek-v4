# Mega-kernels: one tt-lang kernel per zone between CCLs

Each kernel here covers all the work between two collective ops in the
decode step. The goal is one tt-lang dispatch per region, replacing the
~ttnn op chains the model uses today. Sharding/pipes are intentionally
left out — once each kernel exists with a correct functional body and a
PCC test, we layer 2D K/N fanout on top.

Decode is M=1 throughout. Shapes below are logical (un-sharded) sizes; the
fused kernel needs to handle whatever output sharding the surrounding CCL
expects.

V4-Flash constants:
- `dim=4096`, `mhc=4`, `D = mhc*dim = 16384`, `num_tokens_pad=32` (TILE).
- `q_lora_rank=1024`, `head_dim=512`, `rope_head_dim=64`, `n_heads=64`.
- `o_groups=8`, `o_lora_rank=1024`, `moe_inter_dim=2048`.
- `n_routed_experts=256`, `topk=8`, `vocab=129280`.
- `window_size=128`, `compress_ratio ∈ {0, 4, 128}` per layer.

---

## L0: embed_prep + first hc_pre + attn_norm + wq_a

Layer-0 only. After layer 0 this is replaced by `Lk-A` since the residual
stream is already in `a_tt` format.

**Pre-CCL**: `embedding all_gather` (cluster_axis=1).
**Post-CCL**: `wq_a all_gather`.

**Ops absorbed:**
- typecast(embed_bf16 → fp32)
- repeat across `mhc` (4 streams)
- pad to `num_tokens_pad`
- mhc pre: `norm_fn` (rmsnorm + linear) → `split_mixes` → `sinkhorn` → `apply_mix_h`
- typecast(fp32 → bf16)
- attn_norm rmsnorm
- matmul(x, wq_a)

Note: tt-lang cannot handle typecasts, so typecast before the op in ttnn (if needed_ and move sinkhorn to bf16 to avoid the second typecast (maybe this removes the first too).

**Inputs:**
- `embed_tt`: `[1, 1, dim]` bf16 (replicated)
- `wq_a`: `[dim, q_lora_rank]` bf16
- `hc_attn_fn`: `[mhc_mult3=24, D]` fp32 (`mhc_mult3 = 2*mhc + mhc² = 24`)
- `hc_attn_scale, hc_attn_base`: `[3]`, `[24]` fp32
- `attn_norm_gamma`: `[dim]` bf16

**Output:**
- `wq_a_partial`: `[1, 1, q_lora_rank]` bf16 (sharded along last dim).
- `a_tt_next`: `[num_tokens_pad, D]` fp32 (the residual stream for the next layer's hc_pre — must also be emitted; this is the "stash" used by hc_post downstream).

---

## Lk-A: hc_pre_attn + attn_norm + wq_a (per layer)

**Pre-CCL**: previous layer's `wo_b all_gather` (or routed-expert `all_reduce` for layer 0 transition — see Lk-F).
**Post-CCL**: `wq_a all_gather`.

**Ops absorbed:**
- mhc pre: `norm_fn` → `split_mixes` → `sinkhorn` → `apply_mix_h`
- typecast(fp32 → bf16)
- attn_norm rmsnorm
- matmul(x, wq_a)

**Inputs:**
- `a_tt`: `[num_tokens_pad, D]` fp32
- `hc_attn_fn`: `[24, D]` fp32
- `hc_attn_scale, hc_attn_base`: `[3]`, `[24]` fp32
- `attn_norm_gamma`: `[dim]` bf16
- `wq_a`: `[dim, q_lora_rank]` bf16

**Output:**
- `wq_a_partial`: `[1, 1, q_lora_rank]` bf16

---

## Lk-B: q_norm + wq_b

**Pre-CCL**: `wq_a all_gather`.
**Post-CCL**: `wq_b all_gather`.

**Ops absorbed:**
- reshape
- q_norm rmsnorm
- matmul(qr, wq_b)

**Inputs:**
- `q_lora_tt`: `[1, 1, q_lora_rank]` bf16 (replicated)
- `q_norm_gamma`: `[q_lora_rank]` bf16
- `wq_b`: `[q_lora_rank, n_heads * head_dim]` bf16

**Output:**
- `q_full_partial`: `[1, 1, n_heads * head_dim = 32768]` bf16

(The `fused_matmul_rmsnorm.py` skeleton in `tt-lang-kernels/` is the right
prototype.)

---

## Lk-C: q_rsqrt_norm + q rotary + wkv

**Pre-CCL**: `wq_b all_gather`.
**Post-CCL**: `wkv all_gather`.

**Ops absorbed:**
- reshape to `[B, S, H, D]`
- q_rsqrt_norm: `q *= rsqrt(mean(q², dim=-1) + eps)` per head
- pick cos/sin via `embedding(start_pos, table)`
- slice q into nope/rope, rotary on rope half, concat
- matmul(x, wkv)

**Inputs:**
- `q_full_tt`: `[1, 1, n_heads * head_dim]` bf16
- `cos_full, sin_full`: `[max_seq_len, rope_head_dim/2]` bf16
- `start_pos_tt`: `[1, 1]` uint32
- `x_tt`: `[1, 1, dim]` bf16 (the residual from before attn_norm — needs
  to be threaded through; alternatively this kernel takes both q-stack and
  x-residual as inputs and runs in parallel)
- `wkv`: `[dim, head_dim]` bf16

**Outputs:**
- `q_tt`: `[1, 1, n_heads, head_dim]` bf16 (rotated, normed; lives in scratch)
- `wkv_partial`: `[1, 1, head_dim]` bf16

(Alternatively split into Lk-C1 = `q_rsqrt_norm + q rotary` and Lk-C2 = `wkv matmul`,
since they consume different inputs and Lk-C2 already needs an all_gather output.)

---

## Lk-D1: kv_norm + kv rotary + act_quant_block

**Pre-CCL**: `wkv all_gather`.
**Post-CCL**: indexer.wq_b `all_gather` (ratio=4 layer, T_active>0)
              OR compressor.wkv `all_gather` (every compressor layer)
              OR Lk-Dsparse path begins (no compressor layer).

**Ops absorbed:**
- reshape
- kv_norm rmsnorm
- pick cos/sin, slice nope/rope, rotary on rope half, concat
- act_quant_block on the nope half (block-wise max-abs round-trip in bf16)
- concat back

**Inputs:**
- `kv_tt`: `[1, 1, head_dim]` bf16
- `kv_norm_gamma`: `[head_dim]` bf16
- `cos_full, sin_full`: `[max_seq_len, rope_head_dim/2]` bf16
- `start_pos_tt`: `[1, 1]` uint32

**Output:**
- `kv_tt`: `[1, 1, head_dim]` bf16 (normed, rotated, nope-quantized)

---

## Lk-D-idx: indexer body (only ratio=4 layers, T_active>0)

The indexer has 4 internal CCLs; this is really 4 small mega-kernels in
sequence. List them as one section since the topology is shared.

### Lk-D-idx-q: indexer q-stack

**Pre-CCL**: `wkv all_gather` (sharing input with Lk-D1).
**Post-CCL**: `indexer.wq_b all_gather`.

**Ops absorbed:**
- matmul(qr, indexer.wq_b)
- reshape
- pick cos/sin, slice nope/rope, rotary, concat
- Walsh-Hadamard rotation (matmul against H constant, head_dim=128)

**Inputs:**
- `qr_tt`: `[1, 1, q_lora_rank]` bf16 (from Lk-B output, threaded through)
- `indexer.wq_b`: `[q_lora_rank, index_n_heads * index_head_dim = 64*128 = 8192]` bf16
- `cos_full, sin_full`: `[max_seq_len, rope_head_dim/2]`
- `H`: `[index_head_dim, index_head_dim] = [128, 128]` bf16 (Hadamard/√d)

**Output:**
- `q_idx_tt`: `[1, 1, index_n_heads, index_head_dim] = [1,1,64,128]` bf16

### Lk-D-idx-cmp-wkv: indexer compressor wkv + APE + state-front update

**Pre-CCL**: `indexer.wq_b all_gather` (or fold this into Lk-D-idx-q output path).
**Post-CCL**: `indexer.compressor.wkv all_gather`.

Tiny zone. Reasonable to merge into Lk-D-idx-cmp-wgate as one combined kernel.

**Ops absorbed:**
- matmul(x, compressor.wkv)

### Lk-D-idx-cmp-wgate: indexer compressor wgate + APE + state updates

**Pre-CCL**: `indexer.compressor.wkv all_gather`.
**Post-CCL**: `indexer.compressor.wgate all_gather`.

**Ops absorbed:**
- matmul(x, compressor.wgate)
- embedding(start_pos, ape_padded), reshape, add(score, ape_slot)
- slice front/back ×4 (overlap=True)
- 4× paged_update_cache to state_front / state_back / score_front / score_back

(The 4 `paged_update_cache` calls write at `state_slot_tt`, which is
pre-uploaded by `pre_stage_decode`. They could remain ttnn-experimental
calls outside the kernel, OR be replaced with tt-lang slot writes.)

**Inputs:**
- `x_tt`: `[1, 1, dim]` bf16
- `compressor.wgate`: `[dim, 2*head_dim = 256]` bf16
- `ape_padded`: `[max_seq_len, 2*head_dim]` bf16
- `start_pos_tt`: `[1, 1]` uint32
- state buffers (in/out): `kv_state_{front,back}_tt`, `score_state_{front,back}_tt`,
  each `[1, 1, ratio_pad=32, head_dim/2=64]` bf16
- `state_slot_tt`: `[1]` int32

**Output:**
- updated state buffers (in place).

### Lk-D-idx-emit: emit branch (only when (start_pos+1) % 4 == 0)

**Pre-CCL**: `indexer.compressor.wgate all_gather`.
**Post-CCL**: `indexer.weights_proj all_gather`.

**Ops absorbed:**
- `compressor_softmax_sum_norm` (existing TTL kernel, d=128) → kv_normed
- pick cos/sin (compressor variants), slice nope/rope, rotary, concat
- `_device_rotate_activation` (Walsh-Hadamard, d=128)
- paged_update_cache to compressed kv_cache
- 4× `compressor_slot_shift` (existing TTL kernel) + 4× ttnn.copy

**Inputs:**
- state buffers (`kv_state_{front,back}_tt`, `score_state_{front,back}_tt`) at d=128
- `cssn_mask_{front,back,pad}`: `[TILE, TILE]` bf16
- `gamma`: `[TILE, head_dim]` bf16 (from compressor.norm)
- `cos_compressor, sin_compressor`: `[max_seq_len, rope_head_dim/2]` bf16
- `H`: `[128, 128]` bf16
- `kv_cache`: `[1, 1, T_pad, head_dim=128]` bf16 (the indexer's own cache)
- `emit_slot_tt`: `[1]` int32
- `shift_P`: `[ratio_pad=32, ratio_pad=32]` bf16

**Output:**
- updated indexer kv_cache + state buffers (in place).

### Lk-D-idx-score: indexer weights_proj + score reduce

**Pre-CCL**: `indexer.weights_proj all_gather`.
**Post-CCL**: none (score is consumed by Lk-D-topk on the same chips).

**Ops absorbed:**
- matmul(x, weights_proj) → w_tt
- multiply(w_tt, scale)
- transpose(kv_cache)
- matmul(q_idx, kv_cacheᵀ) → score
- relu(score)
- transpose, reshape(w), multiply, sum(dim=-1) → reduced score `[1, 1, T_pad]`

**Inputs:**
- `x_tt`: `[1, 1, dim]` bf16
- `weights_proj`: `[dim, index_n_heads=64]` bf16
- `q_idx_tt`: `[1, 1, 64, 128]` bf16
- `kv_cache`: `[1, 1, T_pad, 128]` bf16
- `scale`: scalar = `softmax_scale * (n_heads ** -0.5) = 1/sqrt(128) * 1/sqrt(64)`

**Output:**
- `score_tt`: `[1, 1, T_pad]` bf16

---

## Lk-D-topk: bucketed pad+mask topk

**Pre-CCL**: post-Lk-D-idx-score (no CCL between them — score is per-chip
identical when weights_proj is replicated/gathered).
**Post-CCL**: none (output feeds Lk-Dsparse directly).

**Ops absorbed:**
- ttnn.copy(score, indexer_score_in_buffer) (or fold staging into the kernel)
- slice(score, [0, 0, 0], [1, 1, bucket])
- lt(ramp_int, t_active) → mask_bool
- typecast bool→bf16, subtract 1, multiply 1e4 → additive mask
- add(score_slice, mask_add) → masked
- topk(masked, k=k_fixed)
- lt(vals, -1000) → invalid; correction math: `idxs_winned = idxs + win`,
  `cmp_idxs = idxs_winned - (idxs_winned + 1) * invalid_int`

**Inputs:**
- `score_tt`: `[1, 1, T_pad]` bf16 (or its per-bucket slice)
- `ramp_int`: `[1, 1, bucket]` int32 (per-bucket persistent buffer)
- `t_active_tt`: `[1, 1, bucket]` int32 (staged outside the trace)
- `bucket, k_fixed, win`: compile-time constants

**Output:**
- `cmp_idxs_int`: `[1, 1, k_fixed]` int32 (with -1 sentinels)

---

## Lk-D-comp: attn-side compressor (only when compress_ratio set)

Same shape as the indexer's compressor body, except `head_dim=512`,
overlap depends on ratio (True for ratio=4, False for ratio=128). For
ratio=128 the cssn kernel doesn't apply (non-overlap path uses ttnn
softmax + multiply + sum + the existing rmsnorm TTL kernel).

**Pre-CCL**: depends on which compressor matmul you're absorbing.
- compressor.wkv: pre-CCL is `wkv all_gather` (or `indexer.weights_proj
  all_gather` if indexer is present).
- compressor.wgate: pre-CCL is `compressor.wkv all_gather`.

**Post-CCL**: `compressor.wgate all_gather` for the matmul tail; the
emit body itself has no post-CCL (writes are local + paged_update_cache).

**Ops absorbed:** same set as Lk-D-idx-cmp-wgate / Lk-D-idx-emit but at
`d=512` and without the rotate=True step (attn-side compressor doesn't
Hadamard-rotate). For ratio=128 the emit body uses the existing
`rmsnorm` TTL kernel inside the kernel rather than `cssn`.

---

## Lk-Dsparse: sparse_attn body + inverse rotary + wo_a + wo_b

**This is the largest mega-zone in the model.**

**Pre-CCL**: depends on what came last —
- compressor present: `compressor.wgate all_gather`
- indexer present without compressor (impossible — indexer always has compressor)
- no compressor: `wkv all_gather`.

**Post-CCL**: `wo_b all_gather`.

**Ops absorbed:**
- kv_cache_tt update at window slot via `paged_update_cache(kv_slot_tt)`
- Build idxs/valid mask: `_idxs_int_tile_to_idxs_and_mask` (lt, reshape,
  where, clamp, reshape, typecast, to_layout)
- gather: `embedding(idxs, kv_cache)` → kv_gather
- score: `matmul(q, kv_gatherᵀ)`
- multiply by softmax_scale, add valid_mask
- concat sink column → softmax → drop sink slice
- output matmul: `matmul(probs, kv_gather)` → o
- inverse rotary on o[..., -rd:] (cos/sin from embedding(start_pos, …))
- group reshape `[B, S, H, D] → [G, B*S, H*D/G]`
- block-diag wo_a matmul (replicated weight, no CCL inside)
- reshape, permute, reshape
- wo_b matmul

**Inputs:**
- `q_tt`: `[1, 1, n_heads, head_dim]` bf16 (from Lk-C)
- `kv_tt`: `[1, 1, head_dim]` bf16 (from Lk-D1, post act_quant_block)
- `kv_cache`: `[1, 1, kv_cache_size_pad, head_dim]` bf16 (in place)
- `kv_slot_tt`: `[1]` int32
- `topk_idxs`: `[1, 1, win + k_compressed]` int32 (window indices via
  `embedding(start_pos, _win_idxs_padded_tt)` concatenated with
  `cmp_idxs_int` from Lk-D-topk; for non-indexer compressor layers the
  compressed slice comes from `_compress_idxs_ramp_tt`)
- `attn_sink`: `[n_heads]` bf16
- `cos_full, sin_full`, `start_pos_tt`
- `wo_a`: `[G=8, head_dim*n_heads/G=4096, o_lora_rank=1024]` bf16 (replicated)
- `wo_b`: `[G * o_lora_rank = 8192, dim=4096]` bf16

**Output:**
- `wo_b_partial`: `[1, 1, dim]` bf16 (sharded along last dim).

---

## Lk-E: hc_post_attn + hc_pre_ffn + ffn_norm + shared expert + gate

**Pre-CCL**: `wo_b all_gather`.
**Post-CCL**: shared expert `all_reduce` (after w2 partial).

**Ops absorbed:**
- hc_post (attn-side): `mhc_post_kernel` plus residual layout / slice / pad / typecast
- packing into `a_tt` for ffn-side hc_pre
- hc_pre (ffn-side): `norm_fn → split_mixes → sinkhorn → apply_mix_h`
- typecast(fp32 → bf16)
- ffn_norm rmsnorm
- shared expert: `matmul(x, w1)`, `matmul(x, w3)`, clamp ×2, silu, multiply, `matmul(mid, w2)` → partial
- (optional) MoE gate: `matmul(x, gate_w)`, softplus, sqrt, etc. — gate
  reads the same `x` as the shared expert; can run concurrently inside
  the kernel and emit gate weights+indices alongside the partial.

**Inputs:**
- `wo_b_out`: `[1, 1, dim]` bf16 (the attn output, replicated post-CCL)
- `prev_a_tt`: `[num_tokens_pad, D]` fp32 (residual from before attn,
  threaded through Lk-A → Lk-E since `mhc_post` needs the matching
  pre/comb stash)
- `hc_attn_post_mix, hc_attn_comb_sk` (the stash from this layer's Lk-A
  hc_pre — must persist across the layer's inner CCLs)
- `hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_gamma`
- shared expert weights `w1, w2, w3`: `[inter, dim], [dim, inter], [inter, dim]` bf16
- (optional) `gate_w`: `[n_routed_experts=256, dim]` bf16 (replicated)
- (optional) `gate_bias`: `[256]` bf16 (replicated; non-hash layers only)

**Outputs:**
- `shared_partial`: `[1, 1, dim]` bf16 (sharded; awaiting all_reduce)
- (optional) `gate_weights`: `[1, topk]` bf16, `gate_indices`: `[1, topk]` int32
- `next_a_tt`: `[num_tokens_pad, D]` fp32 (the residual stream the next
  layer's hc_pre will consume — output of hc_post_attn → hc_pre_ffn)

---

## Lk-F: gate post-process + routed experts + reduce

**Pre-CCL**: shared expert `all_reduce`.
**Post-CCL**: routed expert `all_reduce`.

**Ops absorbed:**
- if not folded into Lk-E: gate post-processing (sqrt-softplus, add bias,
  topk, gather, sum, div, multiply by route_scale)
- selection mask: `eq(indices, chip_local_ids)`, typecast, multiply by
  weights, sum
- routed experts: reshape, repeat across per_chip, `matmul(x, w1_bfp4_b)`,
  `matmul(x, w3_bfp4_b)`, clamp ×2, silu, multiply, `matmul(y1, w2_bfp4_b)`
- multiply by mask, sum across local experts → `y_local`
- (optional) add shared_out (from Lk-E) inline before all_reduce

**Inputs:**
- `x_tt`: `[1, 1, dim]` bf16 (post-shared-all_reduce + ffn_norm output, replicated)
- `gate_weights, gate_indices`: from Lk-E if folded, else computed here
- `chip_local_ids_tt`: `[1, per_chip, 1, 1]` int32 (sharded over rows × cols)
- routed-expert weights `w1, w2, w3`: bfp4_b, sharded
  `[1, per_chip, dim, inter], [1, per_chip, inter, dim], [1, per_chip, dim, inter]`
- `shared_out_tt` (if folding shared_expert add inline): `[1, dim]` bf16

**Output:**
- `y_local`: `[1, 1, 1, dim]` bf16 (per-chip local sum, awaiting all_reduce)

(After all_reduce: `y_full` is the layer's MoE output, replicated. The
add of `shared_out` happens after the all_reduce in today's code; folding
it before requires routing the shared partial through this kernel.)

---

## Final: head logits + topk

**Pre-CCL**: last layer's routed-expert `all_reduce`.
**Post-CCL**: none (host pulls 4 bytes after the trace).

**Ops absorbed:**
- (last layer's hc_post_ffn — currently emitted into `a_tt` for the head)
- `slice(a_tt, last row)` → `x_2d`: `[1, mhc*hidden=16384]` fp32
- hc combiner: `matmul(x_2d, hc_fn_t)` → mixes; rsqrt(mean(x²)+eps);
  multiply mixes by rsqrt by hc_scale; add hc_base; sigmoid; add hc_eps
- `matmul(pre_3d, x_3d)` → y `[1, 1, hidden]` fp32
- typecast(fp32 → bf16)
- final RMSNorm (currently `ttnn.rms_norm`; absorb into the kernel)
- lm_head matmul: `matmul(y, w_lmhead)` → logits `[1, 1, vocab_padded]` bf16
- per-chip `topk(k=1)` → `(top_val, top_idx)`

**Inputs:**
- `a_tt`: `[num_tokens_pad, D]` fp32 (residual from last layer)
- `hc_head_fn`: `[mhc, mhc*hidden] = [4, 16384]` fp32 (transposed for matmul)
- `hc_head_scale`: `[1]` fp32
- `hc_head_base`: `[mhc=4]` fp32
- `final_norm_gamma`: `[dim=4096]` bf16
- `w_lmhead`: `[dim=4096, vocab_padded]` bf16 (sharded along vocab; per-chip
  slice depends on mesh)

**Output:**
- `top_val_tt`: `[1, 1, 1]` bf16 per chip
- `top_idx_tt`: `[1, 1, 1]` uint16 per chip

---

## Note on naming

`compressor_softmax_sum_norm` is a great example of a name for a fused ttlang kernel. Please name like this "ttl_<component>_<list_of_ops_fused>_lk_a" where the prefix (ttl) and suffix (lk_a) is optional. Don't worry about being verbose, this will help readers understand what the kernel does.

## Note on bringup

It is FINE to under utilize hardware for now and to have multiple kernels for one mega kernel for now. The goal is to get the logic ALL in tt-lang, not perf right now. Once we have it in tt-lang, fusion becomes easier and we can iterate on perf.

## Committing

After you complete a kernel and it passes, commit the kernel with a detailed description. Start your commit with something like "Implement a tt-lang kernel from the following ttnn reference..."

## Prototype in isolation

If you need a component, you can prototype it in isolation, for example you could create a topk kernel in tt-lang in /tmp and iterate on that before dropping it in with the other logic.

## Existing TTL kernels these absorb

The eleven kernels already wired in `inference.py` are the building blocks:

| existing kernel | mega-zone(s) it lives in |
|---|---|
| `rmsnorm` | L0, Lk-A, Lk-B, Lk-C-bridge, Lk-D1, Lk-E, Final |
| `mhc_norm_fn_ksplit` | L0, Lk-A, Lk-E |
| `mhc_split_mixes` | L0, Lk-A, Lk-E |
| `mhc_sinkhorn` | L0, Lk-A, Lk-E |
| `mhc_apply_mix_h` | L0, Lk-A, Lk-E |
| `mhc_post` | Lk-E |
| `act_quant_block` | Lk-D1 |
| `compressor_softmax_sum_norm` (cssn) | Lk-D-idx-emit, Lk-D-comp |
| `compressor_slot_shift` | Lk-D-idx-emit, Lk-D-comp |

The mega-kernels are how to fuse these together (and the surrounding
ttnn ops) into one dispatch per zone.

## CCL summary (per layer + global)

Per layer:
1. `wq_a all_gather` (after Lk-A)
2. `wq_b all_gather` (after Lk-B)
3. `wkv all_gather` (after Lk-C)
4. — *indexer-only:* `indexer.wq_b all_gather` (after Lk-D-idx-q)
5. — *indexer-only:* `indexer.compressor.wkv all_gather`
6. — *indexer-only:* `indexer.compressor.wgate all_gather`
7. — *indexer-only:* `indexer.weights_proj all_gather`
8. — *compressor-only:* `compressor.wkv all_gather`
9. — *compressor-only:* `compressor.wgate all_gather`
10. `wo_b all_gather` (after Lk-Dsparse)
11. `shared expert all_reduce` (after Lk-E)
12. `routed expert all_reduce` (after Lk-F)

Global:
- `embedding all_gather` (1× per token, before L0)
- No CCL after Final (host pull only).

Total at decode (43 layers, 40 with compressor, 20 with indexer):
- 43 × (3 attn linears + wo_b + 2 reduces) = 258
- 40 × 2 (compressor.wkv + wgate) = 80
- 20 × 4 (4 indexer linears) = 80
- 1 (embedding) = 1
- **≈ 419 CCLs/token**.

Each CCL is one mega-kernel boundary.
