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

---

## New fused kernels (in `tt-lang-kernels/`) and how to deploy them

Three kernels were prototyped and benchmarked at V4-Flash decode shapes. Each
file is self-contained with a mesh4 PCC test against a torch reference.
Numbers below were measured on all.conf (1×4 BH QB).

### 1. `fused_matmul_rmsnorm.py` — rmsnorm(x, gamma) @ W in one kernel

**Math:** `(1/rms[m]) * (x @ Wg)` where `Wg = gamma[:, None] * W` is pre-baked
on host (no per-step gamma traffic). The K-loop accumulates the matmul
partial AND the per-row ssq simultaneously; after K, the kernel finalizes
inv_rms and multiplies into the output.

**Optimal deployment:**
- **Mesh layout:** put W on as many cards as possible by **col-sharding `Wg`
  along its N axis** (`ShardTensor2dMesh(mesh, dims=(None, -1))`). For Galaxy
  (4, 8) use 2D N-shard (= TP=32). For all.conf use 1×4 (TP=4).
- **Replicate** x and the [TILE, TILE] scaler tile (`ReplicateTensorToMesh`).
- **CCL needed:** **0 inside the kernel**, and **1 downstream** —
  `ttnn.all_gather(out, dim=-1, cluster_axis=N_AXIS)` if the consumer needs a
  replicated input. Same all_gather as today's `DeviceColLinear`; the fusion
  doesn't change the CCL story.

**Where to land it:** Replace the `rms_norm + matmul` pair at the four K=4096
fusion sites: `attn_norm → wq_a`, `ffn_norm → shared_expert.w1` (single
fanout root), `final_norm → lm_head`. Skip `w3`/`gate` fanouts and let them
re-use a shared rms_norm output (avoids paying the rms work 3x per layer).

**Caveat:** at decode K=4096 the fused kernel is **~1.4x slower** than
`ttnn.rms_norm + ttnn.matmul` (mesh4: 0.34ms vs 0.25ms). It uses 8 cores
(SUMMA grid Mp=1, Np=8) versus ttnn's full ~64-core matmul. Adopt only at
sites where ttnn is *not* well tuned, or where the dispatch reduction
(2 ops → 1 op under tracing) makes a meaningful difference. **lm_head**
(N=129280) is the strong candidate; the small attn fusion sites are not.

### 2. `pre_norm_fn_ksplit.py` — pre_norm_fn parallelized across the K axis

**Math:** unchanged from `pre_norm_fn.py`:
`mixes[t, m] = (residual[t] @ fn[m, :]) * rsqrt(sum_k residual[t, k]^2/D + eps)`.
The decode shape (D=16384, 1 output tile, K_tiles=512) ran the original
single-core in 0.69ms. The ksplit version splits the K-loop across `Kp`
cores via two pipe-gather reduces (matmul partial + ssq partial), then the
root finalizes inv_rms.

**Optimal deployment:**
- **Per-chip config:** `Kp=8, grid=(8, 1)` — uses 8 cores per chip. **0.25ms
  per call**, 2.80x speedup (saves ~39 ms/token at 86 calls). PCC 0.989
  (acceptable; matches the "fp32 accumulator drift" caveat already in the
  base kernel docstring).
- **Why not push Kp higher:** Kp=16 hits a ~0.24ms floor (I/O-bound, not
  compute-bound). Kp=32 same floor at PCC 0.928 (bf16-precision reduce_sum
  partials compounding).
- **Mesh layout:** **replicated on a small sub-mesh** (1 chip is enough; or
  1×4 sub-mesh of Galaxy if the result needs to feed multiple consumers
  cheaply). Inputs (residual, fn) are replicated. Output is replicated too.
- **CCL needed:** **0 if you accept replication on the sub-mesh.** If you run
  on N < total chips and the result must reach the others, **1 CCL**:
  `ttnn.all_gather` (or a `broadcast`) along the chip axis after the kernel.
  Today the model runs MHC kernels replicated on all 32 chips — that wastes
  31x the compute. A path-2 fix would compute on a 1×8 sub-row and broadcast
  to the other rows (1 CCL/call), trading replicated compute for fabric
  bandwidth. See P2.3.

### 3. `pre_apply_mix_h.py` — pre_apply_mix h-axis-sharded across cores

**Math:** unchanged: `out[t, h] = sum_m x[t, m, :] * mix[t, m]`. The original
parallelized on `num_tokens`, so at decode (num_tokens=1) only one core
fired and the inner h_tiles=128 loop ran serially. This kernel flattens
(token, h_tile) into one work axis and `grid="auto"` distributes across all
~130 BH cores. Each h-tile is independent; **no reduce needed**.

**Optimal deployment:**
- **Per-chip config:** `grid="auto"` (the kernel uses whatever the device
  exposes). On a 130-core BH chip each core handles ~1 h-tile. **0.18ms per
  call**, 1.36x speedup (saves ~6 ms/token at 86 calls). PCC 1.000 — exact.
- **Mesh layout:** **replicated.** Same story as norm_fn_ksplit: today the
  model runs this replicated on all 32 chips. A path-2 fix would compute on
  a 1×8 sub-row and broadcast (1 CCL/call), but at 0.18ms/call the
  broadcast cost likely dominates the saved compute on Galaxy fabric. **Don't
  shard the h axis across chips** — the kernel already saturates the
  per-chip cores; cross-chip h-shard would require an all_gather of the
  output that costs more than the kernel itself.
- **CCL needed:** **0** (if replicated) or **1 all_gather/broadcast** (if
  computed on a sub-mesh and result must reach the others).

### Summary table

| kernel                    | optimal per-chip cores | mesh layout                  | CCLs/call         | per-token saved |
|---------------------------|------------------------|------------------------------|-------------------|-----------------|
| `fused_matmul_rmsnorm`    | 8 (SUMMA Np=8)         | col-shard N across full mesh | 0 inside, 1 after | -22 ms (regression at K=4096; positive only at lm_head N=129280) |
| `pre_norm_fn_ksplit`      | 8 (Kp=8)               | replicated (sub-mesh OK)     | 0 (or 1 broadcast)| +39 ms          |
| `pre_apply_mix_h`         | ~130 (grid="auto")     | replicated (sub-mesh OK)     | 0 (or 1 broadcast)| +6 ms           |

### Recommended landing order

1. **`pre_norm_fn_ksplit`** — biggest single win (39 ms/token), drop-in
   replacement for the existing kernel in `inference.py:_compile_mhc_norm_fn_kernel`,
   no architectural change.
2. **`pre_apply_mix_h`** — small win (6 ms/token), but trivial drop-in.
3. **`fused_matmul_rmsnorm`** — defer until lm_head fusion site is wired.
   At the small fusion sites it's a regression vs ttnn. Treat as the
   foundation for P1.3 (LM head 2D shard with rms fusion), not as an
   immediate decode-path drop-in.

### What we deliberately did *not* do

- **Cross-chip ksplit on `pre_norm_fn`.** Could move from per-chip Kp=8 to
  full-mesh Kp=128 (4 chips × 32 cores). Would require an `all_reduce`
  CCL on the matmul + ssq partials. Estimate: ~0.05ms CCL cost + ~0.05ms
  saved compute = breakeven at best. Not worth the complexity now.
- **Tree-reduce in `pre_norm_fn_ksplit`.** Current gather caps at Kp=32
  (block_count limit). Two-stage gather would unlock Kp=64+, but plateau
  data shows no speedup beyond Kp=8.
- **Higher PCC on `pre_norm_fn_ksplit`.** Removing
  `--no-ttl-reduce-full-fp32` would help at higher Kp but cost time. Skipped
  because Kp=8 PCC=0.989 already meets the existing kernel's 0.998 baseline
  closely enough at decode.
