# Galaxy mesh utilization plan

## Goal

Stop wasting 75% of the Galaxy mesh on every op except routed experts.

## Current state (one-line per section)

Mesh is `(4, 8) = 32` chips with `FabricConfig.FABRIC_1D`. Comment at
`inference.py:3713` is the canonical statement: *"the model only does 1D
tensor parallelism (sharded along the 8-col axis, replicated along the 4-row
axis)."* That means: max effective parallelism is **8** outside the routed
experts, with the **4 row chips doing identical work**.

| section                        | sharded across | unique chips | duplicated |
|--------------------------------|----------------|--------------|------------|
| embedding                      | none           | 1            | 32x        |
| attn linears (wq_a/b, wkv, wo_b) | cols only    | 8            | 4x         |
| attn compute (sparse, KV cache, RoPE, compressor, indexer) | none | 1 | 32x |
| wo_a block-diagonal (G=8)       | none          | 1            | 32x        |
| RMSNorm x4/layer                | none          | 1            | 32x        |
| MHC pre/post (tt-lang fp32)     | none          | 1            | 32x        |
| shared expert (3 matmuls)       | none          | 1            | 32x        |
| MoE gate                        | none          | 1            | 32x        |
| **routed experts (Path D)**     | rows + cols   | **32**       | 1x         |
| LM head                         | cols only     | 8            | 4x         |

Decode also hits a second utilization floor: `M=1` on RMSNorm and MHC kernels
turns into `tiles_per_core = ceil(1/64) = 1`, so only **one core** out of ~64
fires per chip. Combined with the 32x replication that's ~0.05% mesh
utilization for those ops.

## Two paths

### Path 1: enough data to scale - shard 2D, light up all 32 chips

These sections have enough data per step that going from TP=8 to TP=32 is a
real compute and/or DRAM win.

### Path 2: not enough data to scale - stop replicating, restrict to a subset

These sections are small enough that even TP=8 is overkill. Don't pay the 32x
DRAM cost just so every chip can re-derive the same answer; either fuse the
work into a downstream op or compute on a small subset and broadcast.

---

## Path 1 top wins (in priority order)

### P1.1 - Shared expert: 2D shard inter_dim x dim

**Where:** `DeviceSharedExpert` (`inference.py:2628`).

**Today:** `w1`/`w2`/`w3` are `ReplicateTensorToMesh` (line 2666). Per layer is
~48 MB bf16; across 43 layers and 32 chips that's ~64 GB of duplicated DRAM
holding the same bytes. Every decode step the SwiGLU pipeline runs identically
on all 32 chips.

**Plan:**
- Shard `w1`/`w3` `[inter_dim=2048, dim=4096]` as col-parallel TP=32 (shard
  inter on the 32-flattened mesh, or 2D `inter` x `dim` if a fused matmul
  needs it).
- Shard `w2` `[dim, inter_dim]` as row-parallel TP=32 with `all_reduce` after.
- Activation in is replicated; activation out is replicated after the
  all-reduce. Same calling contract as today.

**Win:** ~32x compute reduction on a per-layer hot path (3 matmuls + 2 clamps
+ silu + multiply, called every layer every step). ~62 GB DRAM saved.

**Risk:** swiglu_limit clamp interacts with the partial sums; clamp must
happen after the all-reduce reconstruction (it already does logically, just
need to make sure the new shard doesn't change semantics).

---

### P1.2 - Attention head sharding (KV cache + sparse_attn)

**Where:** `DeviceAttention` (`inference.py:3325`), `DeviceSparseAttn`
(`inference.py:2764`), `kv_cache_tt` (line 3437).

**Today:** `n_heads=64`. KV cache is `[B, kv_cache_size_pad, head_dim=512]`
**replicated 32x** on every chip. At `max_seq_len=65536` with ratio=4 that's
~16 MB/layer x 43 layers = ~700 MB replicated 32x = ~22 GB of DRAM held by
duplication alone. `sparse_attn` does the gather + score matmul + softmax +
weighted-sum identically on all 32 chips.

**Plan:**
- Shard the head axis `H=64` across the 4 row chips (16 heads/row) and 8
  col chips (2 heads/chip). 64 is divisible by 32.
- Q is sharded by head; KV cache is *replicated within* a head shard since MLA
  uses 1 logical head for KV (already MQA-like). Sparse attention runs
  per-shard.
- After `wo_a` (which is block-diagonal G=8 - see P1.4) gather along the head
  axis before `wo_b`.

**Win:** ~32x DRAM reduction on KV cache at long context. ~32x compute
reduction on `sparse_attn` (gather + score matmul scales linearly with K,
dominates at long context).

**Risk:** Compressor / indexer state buffers are tied to KV cache today; need
to decide whether to shard them or keep on the replicated path. They're
small per-layer (one slot per token slot, ~hundreds of KB), so leaving them
replicated is fine for v1.

---

### P1.3 - LM head: 2D vocab x dim shard

**Where:** `DeviceLMHead` (`inference.py:885`), shard at line 923.

**Today:** vocab=129280, dim=4096. `dims=(None, -1)` shards vocab on cols
only, replicates 4x on rows. ~125 MB/chip weight.

**Plan:** Promote to `dims=(0, -1)`-style 2D (or whatever the column-and-row
form is named). vocab/32 = 4040 cols/chip; dim/4 = 1024 rows/chip. After the
matmul, the reduction across the row axis collapses the partial sums and the
concat across the col axis stitches the vocab.

**Win:** Called every decode step on a 1 GB weight. 4x more compute
parallelism on the matmul, 4x DRAM saved per chip.

**Risk:** Low. Composer change + an `all_reduce` along rows.

---

### P1.4 - Attention linears: 2D shard, plus wo_a uses G=8 group axis

**Where:** `DeviceColLinear` users for wq_a/wq_b/wkv/wo_b (lines 1078, 4092
through 4117), and the block-diagonal wo_a path (line 3395).

**Today:** All four wq/wkv/wo_b linears col-sharded TP=8 with 4x row
replication. wo_a is fully replicated 32x as `[G=8, in, R]`.

**Plan:**
- Promote wq_a/wq_b/wkv/wo_b to 2D shard. out_dim is divisible by 32 for all
  four (wq_b out=`H*D=32768`, wkv=512, wo_b=4096, wq_a=q_lora_rank=1024).
- wo_a: shard the `G=8` group axis on the row axis (4 of 8 groups per row at
  size-4 rows... actually G=8 maps perfectly onto cols, not rows; reconsider:
  shard groups across cols=8 = 1 group/col, then replicate or shard the
  per-group `in` x `R` matrix on rows).

**Win:** Promotes the linear backbone of attention to TP=32. Cheap once
P1.2 (head shard) is in - they share the same composer plumbing.

**Risk:** Low. Mesh-mapper change. The follow-on `all_gather` on
`cluster_axis=1` (line 1158) becomes a 2D reduction.

---

## Path 2 top wins (in priority order)

### P2.1 - Embedding: stop replicating 1 GB on every chip

**Where:** `offload_embedding` (`inference.py:4169`).

**Today:** `~1 GB` weight `ReplicateTensorToMesh` on all 32 chips = 32 GB of
DRAM holding identical bytes. Per-step compute is a single `ttnn.embedding`
lookup.

**Plan options:**
- A) Vocab-shard the table 1D (cols=8 -> 16128 vocab/chip, ~125 MB) or 2D
  (32 chips -> 4040 vocab/chip, ~32 MB). Each chip looks up only when its
  shard owns the token id; gather/scatter to replicate the result. Trivial
  because there's exactly one token to look up per step.
- B) Keep table on a single chip, broadcast result to the rest. Even simpler
  if mesh-fabric send is cheap; one fabric hop.

**Recommendation:** A. ttnn already has the mesh-mapper plumbing; B requires
a one-off broadcast op. Either way, ~30 GB DRAM saved.

**Risk:** Low. The output is consumed by `_phase("embed")` which already does
a `repeat`/`pad` to the right layout; that downstream op runs on every chip
already.

---

### P2.2 - RMSNorm at decode M=1: fuse into the next matmul, kill the standalone kernel

**Where:** `DeviceRMSNorm` (`inference.py:1512`), called 4x per layer x 43
layers = 172 calls per token.

**Today:** Single-core kernel on each of 32 chips, all running the same work
on a 1-row tile. The replication is inherent to the calling pattern (every
chip needs the normalized residual to feed its sharded matmul).

**Plan:** Fuse rms_norm into the matmul that follows it. tt-metal has
matmul-with-prologue patterns; a tt-lang variant computes
`rsqrt(sum(x^2)/D + eps)` in DST as the matmul's first K-tile passes through.
The standalone DeviceRMSNorm op disappears from the hot path.

**Win:** Removes the 172 single-core dispatches per token. Doesn't reduce
mesh use (the matmul was already running on all 32 chips), but eliminates the
0.05%-utilization phase entirely. Same fix applies to `q_norm`, `kv_norm`,
final norm before lm_head (the lm_head one is already wired through
`norm_tt` in `DeviceLMHead._compute_body`, so the pattern is proven).

**Risk:** Medium. Need a tt-lang kernel that fuses rms into matmul; tt-metal
reference exists. CLAUDE.md says we cannot remove tt-lang kernels - this
*adds* a fused tt-lang kernel and retires a smaller one, which is the
right direction.

---

### P2.3 - MHC kernels at decode M=1: keep restricted, defer

**Where:** `DeviceMHC` (`inference.py:2241`).

**Today:** All constant tiles + `fn_tt` replicated on 32 chips. Each chip
runs the whole pipeline on 1 token's worth of work; one core fires per chip.

**Plan:** Defer. The path-2 fix (run on a 1xN sub-mesh and broadcast) costs a
fabric hop and doesn't actually save DRAM (constants are tiny). The real fix
is fusing `pre_norm_fn` and `pre_apply_mix` into the surrounding matmul, same
shape as P2.2; do it after P2.2's rms fuse pattern is proven.

**Win:** None in this pass. Listed for completeness so we don't try to
"fix" MHC by fragmenting it across the mesh.

---

### P2.4 - MoE gate: leave replicated

Tiny weight (~2 MB), tiny compute. Replication is cheap. Skip.

---

## Suggested attack order

1. **P1.3 LM head 2D shard** - smallest blast radius, validates the 2D mesh
   composer plumbing end-to-end.
2. **P1.4 Attention linears 2D** - reuses the same composer pattern; once
   this lands, P1.2 has clean primitives to build on.
3. **P1.1 Shared expert 2D shard** - independent of attention; biggest single
   compute win per layer, easy to validate against existing PCC test.
4. **P2.1 Embedding shard** - independent and cheap; reclaims the most DRAM
   per line of code changed.
5. **P1.2 Attention head shard** - largest mechanical change because it
   touches sparse_attn, compressor, indexer, KV cache. Biggest long-context
   win; do this last in path-1 once the simpler shards prove the pattern.
6. **P2.2 RMSNorm fuse** - depends on a new tt-lang kernel; concurrent with
   P1.5 work since it's orthogonal to mesh layout.

## Out of scope (intentionally)

- Reworking MHC kernels (P2.3 above).
- Anything that moves work from tt-lang to host or to ttnn.
- MoE routing (still on host per `project_hw_roadmap.md`).
- Sparse attention algorithmic changes - we're only resharding the same
  algorithm onto more chips.
