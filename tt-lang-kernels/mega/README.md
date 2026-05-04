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

## IMPORTANT: all tt-lang code must be in the file

You can duplicate tt-lang kernels from inference, but they need to all be in the same file. This will be important for the optimization stage (later) when we need to tweak the implementation and actually fuse the kernels together. You can start by reference kernels from inference.py, but then please inline them. Multiple kernels is OK for now, so you can have multiple factory functions, but all tt-lang source code needs to be in the file.

You are allowed to create a single library of tt-lang kernels file if you want to share one kernel across several mega kernels and you are SURE that it will use the same config in each mega kernel. Still default to duplicating if you are unusure, but it is OK to create a library file next to _refs.py and use that, this library file should contain only tt-lang kernels (and factory functions) and maybe some light helpers.

## Note on helpers and kernel boilerplate

Sometimes you will need some helpers to reshape or typecast inputs before you invoke the tt-lang kernel(s), this is OK but not ideal. We can handle this in the next phase, but ideally, please keep this setup minimal. If at all possible, do this in the tt-lang kernel instead of in the caller in ttnn. For example, can reshape or slice be implemented with data movement?

## Note on keeping things in ttnn

Don't do it :) The point of this is to produce mega kernels in tt-lang, if there is logic in between in ttnn, this defeats the purpose and will block future fusion and optimization. If you absolutely need to keep something in ttnn, please add the following TODO so that we can revisit: "TODO: mega fusion blocked: ttnn used for X" (make sure to say "TODO: mega" exactly so it can be grepped for).

## Note on chaining ops in one ttl.operation

There is no limit in tt-lang on how many matmuls you can have in a single `ttl.operation` — chain as many matmuls and elementwise/reduce/etc. ops together as you want; fusion is never blocked by a "one matmul per operation" rule. You can declare many `PipeNet`s and `make_dataflow_buffer_like` CBs in the same operation and combine any ops you need across compute / dm_read / dm_write threads.

If you hit a compiler error about too many dfbs or pipenets in one operation, skip that fusion — we have fused as much as we can.

## Important note on remotes

For this part of the project, only use sterling-all.conf, this is the machine we have dedicated to writing mega kernels in tt-lang.

## element_read/write

You now have the element_read and element_write ops, you can find examples and docs in /Users/zcarver/Developer/tt-lang/element_read_write. If you want any guidance on using them or have issues, pause and discuss with me.

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

Note argmax_2pass.py also exists but may need some help for integration, you can search the tt-lang-kernels directory to see all tt-lang kerenls.

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

## Current status: 

we have now implemented all kernels in tt-lang! Well done. Now your job is to turn to performance, please add a benchmark section to each test where the kernel is compiled up front, then run for two warmup runs, then 5+ timed runs and print the output. Do this for the reference and the new tt-lang kernel. Then optimize the kernel. Please optimize as much as possible aiming for a 10x+ speedup. You can improve core utilization, add pipes, tune block sizes. You can make changes one at a time, commit with a succinct and descriptive message describing the speedup and the specific change made. You do not have to hit any particular benchmark to commit as long as pcc passes and perf improves. Keep iterating. Apply changes that helped to multiple kernels. First task: go through and grep for TODO: mega to make sure everything is in tt-lang before we start, then the low hanging fruit is fusing tt-lang kernels together. If there are multiple tt-lang kernels used to build a mega kernel, fuse them into a single mega kernel. Continue iterating autonomously. Keep measuring, trying things, and making small improvements. Ultimate goal: a single fused tt-lang kernel for each section (full decode step in tt-lang, each section is defined as all logic between ccls), that is highly tuned and optimized.

Note on l1: you can move input tensors to L1, but please be careful about this, think about if this would work for the overall inference and model and what the sharding implications would be. This is a powerful lever, but use it carefully.

Block size is another big lever that you have: larger blocks will improve memory bandwidth. 

Pipes require careful consideration but should absolutely be leveraged, you can see how mcast is used in summa matmul and ksplit is used in argmax and ../tt-lang/benchmarks for distributing work and gathering it all in L1, if you need help with pipes, ask me for assistance, this is a place you will be able to see 10x speedups.

## Next task: collapse each test_ file to a single ttl.operation

Block-size and core-utilization tuning is mostly useless until the inter-op
ttnn glue is gone. With multiple `ttl.operation`s per test file we are
double-buffering through DRAM between every stage, and any future block-size
tuning risks running out of L1 because we cannot see the full per-kernel
budget. **The next task is fusion.**

The goal: every `test_*.py` mega test ends up with **one** `ttl.operation`
that produces the file's output. Some ttnn ceremony before and after the
kernel is OK (loading inputs, final layout flips, host pulls). What is not
OK is ttnn slice/reshape/pad/typecast/matmul **between** ttl.operations
inside the file's main kernel body.

Maintain the existing optimizations (ksplit configs, block sizes, PipeNet
mcast patterns, swap-matmul-rotary fusions, etc.) as you fuse. Don't
regress PCC. Don't regress benchmark numbers. Commit each fused kernel
incrementally with a short descriptive message and a before/after timing.

### Worked examples

The three files below have inline `# TODO: claude:` comments calling out
exactly what to fuse. They cover the three difficulty tiers you will see
across the rest of the directory.

#### test_lk_b.py — easy: rmsnorm + matmul fusion

Today: `rms_kernel` produces `[TILE, K]` in DRAM, `matmul_kernel` reads it
back. Both are tt-lang.

Fusion: one ttl.operation on the matmul's grid. PipeNet broadcasts the
rsqrt scalar. Each core:

1. Loads its K-shard of the input.
2. Squares + accumulates locally → contributes to a global ssq reduction
   via PipeNet (same shape as the matmul's existing reductions).
3. Receives `rsqrt(ssq/D + eps)` via PipeNet.
4. Multiplies its K-shard by `gamma * rsqrt` in place.
5. Feeds the normalized K-shard into the matmul accumulate it was already
   going to do.

The `reshape [1,1,K]→[1,K]` / `pad → [TILE,K]` / `slice y_padded → [1,N]` /
`copy → [1,1,N]` ceremony is **calling-convention skew**, not real work.
Either change the kernel contract to take `[TILE, K]` and write `[TILE, N]`
directly (caller pads/unpads), or do the pad in `dm_read` (zero-fill the
unused tile rows) and the slice in `dm_write` (only emit row 0).

#### test_final.py — medium: prologue + rmsnorm + matmul, argmax stays out

Fusion target: `hc_combiner + (small bf16 matmul prologue) + rmsnorm + lmhead_summa`.
Demote `a_tt` to bf16 to drop the explicit `ttnn.typecast` boundary.

The "ttnn glue" between hc_combiner and rmsnorm — sub-tile
slice/reshape/matmul/typecast/pad — collapses to a tiny prologue:
`y[h] = sum_{m=0..MHC-1} pre[0,m] * a_tt[last, m*DIM + h]`. Read `pre_tile`
row 0 (MHC scalars), read MHC strided spans of `a_tt`'s last row, accumulate,
hand to the rmsnorm stage in-CB. No DRAM roundtrip.

Then `rmsnorm + lmhead_summa` is the same pattern as test_lk_b above.

**Known blocker (acceptable for now):** argmax over `[TILE, VOCAB=129280]`.
The root cause is bf16's 8-bit mantissa: integer indices > 256 cannot be
encoded exactly into the `i + sign(v - max) * BIG` value used by
`argmax_2pass`. The fix is hierarchical argmax (windows of ≤256, then a
cross-window stage), but until that lands, leave argmax as
`ttnn.argmax + ttnn.max` after the fused tt-lang kernel. So this file ships
as **two** ttl.operations (fused-body + argmax). Mark with
`# TODO: mega fusion blocked: hierarchical argmax not implemented`.

#### test_lk_a.py / test_l0.py — hard: kill sub-tile data movement glue

The expensive ttnn glue here is sub-tile rearrangement
(`MHC=4 < TILE=32`, `NUM_TOKENS=1 < TILE=32`):

- "Take 16 elements out of a TILE row, arrange as a 4×4 block padded to
  32×32 with sentinel."
- "Take a single row's first 4 cols and pad to a TILE×TILE block."
- "View `a_tt[1, MHC*DIM]` as `[MHC, DIM]` padded to `[TILE, DIM]`."

Two complementary ways to dissolve this:

1. **Fix the producer's output layout.** All of the producers
   (`norm_fn`, `split_mixes`, `sinkhorn`, `apply_mix_h`) are tt-lang
   kernels we own. Inside one fused ttl.operation, intermediates live in
   CBs — there is no boundary to convert across. Have the producer write
   directly into the consumer's expected tile layout. Most of the
   slice/pad/reshape gymnastics disappears.
2. **element_read in DM threads** for genuine sub-tile gather. Works for
   runtime tile-row indexing today; loop reserve/wait + multi-tile gather
   still has bugs (see memory). Use it where it is known-good and fall
   back to (1) where it is not.

Demote fp32 intermediates (`mixes_tt`, `pre_tt`, `comb_tt`,
`apply_mix_out_tt`) to bf16 if the precision budget allows — it removes
the explicit `ttnn.typecast` boundary and keeps the chain in one dtype.

This is the biggest redesign of the three because the producer/consumer
CB layouts inside the fused kernel need to be co-designed. Do it after
the easier files have set the pattern.

### bf16 across the board (default policy)

Default policy for the fused kernels in this phase: **everything is bf16**.
That includes intermediates that were previously fp32 (`mixes_tt`,
`pre_tt`, `comb_tt`, `apply_mix_out_tt`, `pre_tile`, etc.). This is
load-bearing for fusion because every `ttnn.typecast` boundary is a
DRAM roundtrip we are trying to delete — going bf16 end-to-end inside
the fused kernel eliminates them entirely.

Procedure:

1. Demote intermediates to bf16 as part of the fusion change.
2. Run the file's PCC test.
3. If PCC passes → great, ship the bf16 fusion.
4. If PCC drops → revert that specific intermediate to fp32 and add
   `# TODO: mega fusion blocked: bf16 too lossy at <stage> (PCC <x>)`
   so we can revisit. Keep the rest of the kernel bf16; only the
   stage that actually needed the precision stays fp32.

Do not preemptively keep things in fp32 "to be safe" — the typecast
elimination is one of the larger wins available, and fp32 stages should
have to earn their keep with measured PCC evidence.

### Generalization to every test_ file

For each `test_*.py` in this directory:

1. Read the kernel body. Count the `ttl.operation`s and the ttnn ops
   between them.
2. Classify each ttnn op:
   - **Glue between ttl ops** (slice/reshape/pad/typecast/copy that just
     bridges layouts) → must be absorbed into the fused kernel via DM
     threads or producer-layout fixes.
   - **Real compute** (matmul, rms_norm, embedding, argmax, etc.) → must
     either become part of a ttl.operation or, if blocked, be marked
     with `# TODO: mega fusion blocked: <reason>`.
   - **Pre/post ceremony** (input upload, final host-shaped output copy)
     → fine to leave outside.
3. Pick the easiest fusion next (rmsnorm+matmul shape if present is
   usually the cleanest win). Apply the worked-example pattern.
4. Run the test. PCC must pass. Benchmark must not regress. Commit with
   a descriptive message: `<file>: fuse <a> + <b>, <before>→<after>ms`.
5. Repeat until the file has **one** ttl.operation in its main body
   (plus optional argmax/topk if blocked).

Move from easy to hard. The easier files (rmsnorm+matmul shape) set the
PipeNet-broadcast-the-reduction pattern that the harder files reuse.

### Blocked cases

If you genuinely cannot fuse a stage:
- Add a `# TODO: mega fusion blocked: <reason>` comment exactly so it
  greps cleanly.
- Keep moving. Do not regress the file. Do not move logic from tt-lang
  back into ttnn or host to "simplify" — that is a hard rule.
- If the blocker is a real tt-lang primitive gap (not just unwired work),
  add a "Known tt-lang bug" subsection to this README documenting the
  repro and the workaround.

### Known tt-lang bug: reduce_* on fp32 is broken

`ttl.math.reduce_max` / `reduce_sum` on fp32 inputs lose precision in a way
that is not just rounding — the SFPU reduce path collapses to roughly a
10-bit mantissa regardless of input dtype, so e.g. `reduce_max` of an fp32
tile containing `113306` returns `113296` (off by 10), and `--no-ttl-reduce-
full-fp32` + `fp32_dest_acc_en=True` produce byte-identical compute
binaries. Confirmed via `/tmp/reduce_max_isolate.py` and
`/tmp/reduce_max_variants.py`.

**The bf16 reduce path works correctly** within bf16's representable
range — confirmed via `/tmp/reduce_max_bf16.py` (exact across magnitudes
17 → 65636). Every reduce_* call in this directory should run in bf16.

**Scope of the bug today: zero live fusions.** All reduce sites in our
mega kernels (rmsnorm ssq, softmax max/sum, sinkhorn, act_quant abs,
sparse_attn streaming combine, apply_mix) already run on bf16 and pass
PCC. The only knock-on is the Final head argmax over vocab=129280, which
appears affected because `argmax_2pass` uses fp32 — but the deeper issue
there is bf16's 8-bit mantissa not holding integer indices > 256, not the
fp32 reduce bug. The fix for Final is hierarchical argmax (windows of
≤256 so each per-window local index fits in bf16); see the test_final.py
TODO. Until that fusion lands, Final keeps using `ttnn.argmax`/`ttnn.max`.

Do not introduce new fp32 reduce_* call sites. If you find one, switch the
upstream tile to bf16.

### What's actually unwired vs primitive-blocked

After the BUGS.md cleanup, every "TODO: mega fusion blocked" left in the
test_*.py files falls into one of these buckets. **Most of them are
unwired-work, not primitive gaps.**

1. **Cache writes (`paged_update_cache`)** — Lk-Dsparse, Lk-D-comp,
   Lk-D-idx-emit, Lk-D-idx-cmp. Primitive **available**: `element_write`
   (see `/Users/zcarver/Developer/tt-lang/element_read_write`). Unwired.
   Tracked as task C10.
2. **Cos/sin row gather (`embedding(start_pos, …)`)** — Lk-C, Lk-D1,
   Lk-D-idx-q, Lk-D-idx-cmp, Lk-D-idx-emit, Lk-D-comp, Lk-Dsparse.
   Primitive **available**: `element_read`. Unwired. Tracked as C8.
3. **KV-cache row gather (`embedding(idxs, kv_cache)` in Lk-Dsparse)** —
   primitive available in principle but the multi-tile loop pattern is
   known-buggy (`feedback_ttl_loop_dfb_still_buggy.md`). Tracked as C9
   and may stay blocked on the upstream loop-DFB bug.
4. **Topk** — Lk-D-topk (T_active ≤ 2048, k=8) and Lk-F gate post
   (N_ROUTED=256, k=8). No working tt-lang topk yet — initial iterative-
   argmax attempt hung the device (see `tt-lang-kernels/topk_iter_hangs.py`).
   For now treat topk as a ttnn boundary: per file, build a pre_topk mega
   kernel and (if the post-topk work has tt-lang consumers) a post_topk
   mega kernel. Lk-D-topk only needs pre_topk (mask_build, already done);
   Lk-F needs both.
5. **Final head argmax** — vocab=129280. **Genuinely blocked at the
   primitive level**: bf16 cannot encode indices in [0, 129280) into the
   `i + sign(v - max) * BIG` trick (need ≥17 bits; bf16 has 8). Fix is
   hierarchical argmax (3 stages of ≤256). Punt indefinitely; Final
   keeps `ttnn.argmax`.
6. **Sub-tile slice/reshape/typecast (`fp32→bf16`, `int32→bf16`,
   `bool→bf16`)** — test_final, test_lk_d_topk, test_lk_f. Mostly NOT a
   primitive gap: slice/reshape/pad dissolve on fusion (DM threads pick
   the right offsets, zero-fill unused tile rows). int32/bool→bf16 casts
   sidestep via the bf16 sign trick when values fit in bf16's 8-bit
   mantissa (indices < 256, etc.). Genuine gap is only fp32↔bf16 typecast
   when fp32 precision is load-bearing.
7. **Per-expert / block-diagonal matmul (`y[e] = y1[e] @ w2[e]`)** —
   Lk-Dsparse wo_a, Lk-F routed-expert w2 body. Not a primitive gap:
   multiple matmuls fit in one ttl.operation by partitioning the grid into
   per-expert sub-grids, each running standard SUMMA against its own
   input + weight slice. Just unwired kernel-design work. (The w13
   variant `y13 = x @ w13[e]` collapses to a single wide matmul by
   concatenating per-expert weights at offload.)

When adding a new "TODO: mega fusion blocked" in code, please tag it with
the bucket number above so future agents can sort blockers from
unwired-work at a glance.

### When the directory is fully fused

Only once **every** `test_*.py` in `tt-lang-kernels/mega/` has a single
ttl.operation in its main body (with at most one explicit blocked-and-
documented exception per file), update this section to state the fusion
phase is complete and we are returning to block-size / L1 / pipe tuning
on the now-fused kernels. Until then, leave this section in place — the
next agent should read it and continue burning down the list.