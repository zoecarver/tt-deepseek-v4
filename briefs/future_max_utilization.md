# Future max utilization

The model is rewritten so every chip and every core does unique work on every
decode step. No replicated weights, no idle row-replicas, no "1 tile per chip"
matmul leftovers. This is the absolute target shape; everything below is the
spec, not a list of options.

## Target

Sustained ≥50% Galaxy mesh utilization at M=1 decode: ≥2080 of 4160 BH cores
active per step, averaged over the post-trace replay. Concretely:

- Every linear in the model fires on every chip with K-fanout ≥ 8 cores per
  chip on small-N shapes and N-fanout ≥ 32 N-tiles per chip on large-N shapes.
- Sparse attention runs unique work on every chip: zero KV-cache replication.
- MHC pre/post and RMSNorm run unique work on every chip via mesh-wide
  K-split. No 32× replicated MHC pipelines.
- Decode is one trace per emit branch (2 total), as today. Trace count does
  not grow.

Wall-time goal: 5x current tok/s after the rewrite lands.

## Mesh, fabric, submeshes

Mesh is `(rows, cols) = (4, 8)`. Fabric is the 2D form that supports
`all_reduce` and `reduce_scatter` independently on `cluster_axis=0` and
`cluster_axis=1`. The four rows are first-class compute lanes; expose them as
four `SubMesh` handles bound to the parent mesh, plus one full-mesh handle.
Every `Device*` class binds to a specific submesh at offload time and never
issues to a different one.

Submesh handles:

- `mesh_full`     — `(4, 8)` 32 chips. Used by mega-kernels that need full K-shard.
- `mesh_row[i]`   — `(1, 8)` 8 chips. Used by lane work in the input-fanout
  and FFN-fanout zones.
- `mesh_col[j]`   — `(4, 1)` 4 chips. Used for sequence-sharded KV cache
  reductions where the head axis is col-sharded.

A single `step_decode` issues ops to multiple submeshes within one trace.
The trace-capture region wraps the whole step; ttnn dispatches the per-chip
command queues independently, so submeshes execute concurrently when their
chip sets are disjoint.

## Sharding contract: every weight

The contract is uniform: every linear's weight is sharded on both mesh axes.
K (input dim) goes on rows (4-way). N (output dim) goes on cols (8-way). The
matmul's output is K-partial on rows and N-sharded on cols. A
`reduce_scatter` on `cluster_axis=0` (rows) collapses K-partials; the result
stays N-sharded on cols, ready for the next consumer to read directly.

The exception is when K or N fails to tile-align under that split. In those
cases the brief calls out the alternate axis assignment explicitly. There is
no auto-selection: every weight has one and only one declared sharding.

| Weight | Shape `(out, in)` | Submesh | Row axis (4-way) | Col axis (8-way) | Per-chip tile work |
|---|---|---|---|---|---|
| `embed` | `[vocab=129280, dim=4096]` | `mesh_full` | `dim` (K-equiv) | `dim` (K-equiv), flat-shard 32-way over `dim` | per-chip table = `[vocab, dim/32 = 128]`, single-row lookup |
| `attn_norm.gamma` | `[4096]` | `mesh_full` | replicated | replicated | constant, fits in one tile |
| `wq_a` | `[1024, 4096]` | `mesh_full` | `in` (K, 1024 K-tiles/row) | `out` (N, 4 N-tiles/col) | 32 × 4 = 128 tile-products |
| `q_norm.gamma` | `[1024]` | `mesh_full` | replicated | replicated | constant |
| `wq_b` | `[32768, 1024]` | `mesh_full` | `in` (K, 8 K-tiles/row) | `out` (N, 128 N-tiles/col) | 8 × 128 = 1024 tile-products |
| `wkv` | `[512, 4096]` | `mesh_full` | `in` (K, 32 K-tiles/row) | `out` (N, 2 N-tiles/col) | 32 × 2 = 64 tile-products |
| `kv_norm.gamma` | `[512]` | `mesh_full` | replicated | replicated | constant |
| `wo_a` | `[8 × 1024, 4096]` block-diag | `mesh_full` | `in` (K, 32 K-tiles/row) | group axis (G=8, 1 group/col) | 32 × 32 = 1024 tile-products per group |
| `wo_b` | `[4096, 8192]` | `mesh_full` | `in` (K, 64 K-tiles/row) | `out` (N, 16 N-tiles/col) | 64 × 16 = 1024 tile-products |
| `compressor.wkv` | `[2 × 512, 4096]` | `mesh_full` | `in` (K, 32 K-tiles/row) | `out` (N, 4 N-tiles/col) | 32 × 4 = 128 tile-products |
| `compressor.wgate` | `[2 × 512, 4096]` | `mesh_full` | `in` (K, 32 K-tiles/row) | `out` (N, 4 N-tiles/col) | 128 tile-products |
| `indexer.wq_b` | `[64 × 128, 1024]` | `mesh_full` | `in` (K, 8 K-tiles/row) | `out` (N, 32 N-tiles/col) | 8 × 32 = 256 tile-products |
| `indexer.weights_proj` | `[64, 4096]` | `mesh_full` | `in` (K, 32 K-tiles/row) | `out` (N, 0.25 N-tile/col) | K-only fanout, 32 K-tiles/col across cores |
| `indexer.compressor.wkv/wgate` | `[2 × 128, 1024]` | `mesh_full` | `in` (K, 8 K-tiles/row) | `out` (N, 1 N-tile/col) | 8 K-tiles/core |
| `gate.w` | `[256, 4096]` | `mesh_full` | `in` (K, 32 K-tiles/row) | `out` (N, 1 N-tile/col) | K-only fanout |
| `tid2eid` (hash gates) | `[129280, 8]` | `mesh_full` | `vocab` | `vocab`, flat-shard 32-way | per-chip slice ~4040 rows |
| `shared_expert.w1` | `[2048, 4096]` | `mesh_full` | `in` (K, 32 K-tiles/row) | `out` (N, 8 N-tiles/col) | 32 × 8 = 256 tile-products |
| `shared_expert.w3` | `[2048, 4096]` | `mesh_full` | `in` (K, 32 K-tiles/row) | `out` (N, 8 N-tiles/col) | 256 tile-products |
| `shared_expert.w2` | `[4096, 2048]` | `mesh_full` | `in` (K, 16 K-tiles/row) | `out` (N, 16 N-tiles/col) | 256 tile-products |
| `routed_expert.w1/w3` | `[2048, 4096]` × 256 experts | `mesh_full` | expert axis (8 experts/chip across mesh) | expert axis (8 experts/chip across mesh) | per-chip 8 experts × matmul; intra-chip K-fanout |
| `routed_expert.w2` | `[4096, 2048]` × 256 | `mesh_full` | expert axis | expert axis | as above |
| `lm_head` | `[129280, 4096]` | `mesh_full` | `in` (K, 32 K-tiles/row) | `out` (N, 506 N-tiles/col) | 32 × 506 = 16K tile-products |
| `final_norm.gamma` | `[4096]` | `mesh_full` | replicated | replicated | constant |
| All MHC `hc_*_fn` (4 per layer) | `[24, 16384]` | `mesh_full` | `D` (K, 128 K-tiles/row) | `D` (K, 16 K-tiles/col), folded in row × col | 128 × 16 = 2048 K-tiles, K-fanout |

Replication is allowed only for tensors that are pure compile-time constants
(masks, RoPE cos/sin tables, gamma vectors, Hadamard) and for activations
that have already been all_reduce'd (residual stream `x`, post-CCL outputs).

The 32× replicated KV caches are deleted. Both the attention layer's
`kv_cache_tt` and the indexer's `kv_cache_tt` are sequence-sharded as
specified below.

## Compute lanes

The decode step is divided into zones by the major CCL boundaries. Within a
zone, independent op chains run on disjoint submeshes (lanes). Lanes converge
at the next CCL.

### Zone 0: embed + hc_pre + attn_norm (full mesh)

One mega-kernel per phase, full mesh:

- `ttl_embed_pad_repeat_full_mesh` reads the `[1, 1]` token id, gathers the
  per-chip slice of the dim-sharded embedding table, fans out to the
  `[num_tokens_pad, D=16384]` residual layout, all in one fused kernel. Output
  is a row-K-partial / col-N-sharded `a_tt` device tensor.
- `ttl_mhc_norm_fn_split_sinkhorn_apply_mesh_ksplit` is the mesh-wide
  successor to `mhc_norm_fn_ksplit`. K = 16384 = 512 K-tiles. Split as
  4 rows × 8 cols × Kp = 16 cores per chip = 512-way total K-split:
  each core gets one K-tile, ssq + matmul partials reduce via PipeNet
  intra-chip and `all_reduce` (cluster_axis=0 ∪ 1) across the mesh. The
  same kernel folds in the existing `pre_split_mixes`, `sinkhorn`, and
  `pre_apply_mix_h` into its post-reduce stage so the row-K-partial flows
  end-to-end before the first CCL.
- `ttl_attn_norm_full_mesh` runs RMSNorm fused into the same kernel root;
  there is no standalone DeviceRMSNorm dispatch in the hot path.

Pre-CCL: none (zone consumes the post-MoE output of the prior layer).
Post-CCL: `reduce_scatter(cluster_axis=0)` on the K-partials, leaving
`x_norm` col-N-sharded on `mesh_full`.

### Zone 1: attention input fanout (lanes across rows)

After Zone 0 the residual `x_norm` is col-N-sharded (8-way). Each row reads
the same N-shard of `x_norm` and runs an independent op chain. Convergence
is a single `all_gather(cluster_axis=0)` at the end of the zone.

| Lane | Submesh | Mega-kernel |
|---|---|---|
| Lane Q | `mesh_row[0]` | `ttl_lk_a_wq_a_qnorm_wq_b_rotary` |
| Lane KV | `mesh_row[1]` | `ttl_lk_d1_wkv_kvnorm_rotary_actquant` |
| Lane CMP | `mesh_row[2]` | `ttl_lk_d_idx_cmp_wkv_wgate_apeshift_state` (if compressor present) or no-op |
| Lane GATE | `mesh_row[3]` | `ttl_moe_gate_full_pipeline` (gate matmul, sqrt-softplus, bias-add, topk, gather, normalize, route-scale) |

Each lane mega-kernel issues one `reduce_scatter(cluster_axis=1)` at its
output to collapse the K-partials within its row. No row writes outside its
own buffers during the zone. After all four lanes complete, a single
`all_gather(cluster_axis=0)` broadcasts each lane's output to every row so
the next zone reads everything from one place.

The lane mega-kernels each fuse what is currently 6+ ttnn ops:

- Lane Q absorbs `wq_a → q_norm rsqrt → wq_b → q_rsqrt_norm → rotary`. K-shards
  on the row axis (1 K-tile per chip after row-only since this is one row of
  `mesh_full`'s row-shard); intra-chip K-fanout takes care of the rest.
- Lane KV absorbs `wkv → kv_norm rsqrt → rotary → act_quant_block` into one
  kernel. The current standalone `act_quant_block` kernel is folded in.
- Lane CMP runs both `compressor.wkv` and `compressor.wgate` in parallel
  inside the kernel (independent matmuls on the same input), the APE
  embedding-add, and the four `paged_update_cache` writes to state buffers.
  When the layer has no compressor, the lane chip set is reassigned to
  Lane GATE for that step (still inside the trace; the dispatch is selected
  at compile time per layer).
- Lane GATE runs the full gate body. For hash-gate layers it embedding-looks
  the `tid2eid` table; for non-hash it does the topk on biased scores.

### Zone 2: indexer body (lanes within a row)

Only on layers with `compress_ratio == 4` and `T_active > 0`. The indexer's
four-CCL pipeline collapses to one CCL plus three intra-trace reduce_scatter
calls. Both `mesh_row[2]` and `mesh_row[3]` are recruited as a single
`mesh_row_pair` submesh (2 × 8 = 16 chips):

- `ttl_lk_d_idx_q_walsh` fuses `indexer.wq_b → reshape → rotary → Hadamard`.
- `ttl_lk_d_idx_cmp_emit` fuses the indexer's compressor emit body
  (`cssn` kernel + rotary + Hadamard + `paged_update_cache`).
- `ttl_lk_d_idx_score_topk` fuses `weights_proj → score = Q @ kv_cacheᵀ →
  relu → multiply by w → sum → bucket pad-and-mask → topk → +win correction`
  into one mega-kernel. The bucketed pad-and-mask path stays as in the
  current code; the kernel inlines it so the topk K is a compile-time
  constant per bucket and the trace does not branch.

The score matmul uses the sequence-sharded indexer kv_cache (see KV section
below): `Q` is replicated within the lane, `kv_cache` is sequence-sharded on
the lane's col axis, score = Q @ kv_localᵀ → partial-K per col, three small
all_reduces (max + sum for softmax/relu, then sum for output).

### Zone 3: sparse_attn (full mesh, head + sequence parallel)

Sparse_attn is the largest zone. Q is head-sharded on cols (8 heads/col),
KV cache is sequence-sharded on cols (T_pad/8 slots/col, replicated along
rows for cache redundancy is rejected — see KV section). `wo_a` block-diag is
exactly G=8 groups × 1 group per col, so the per-col head-shard feeds wo_a
without any head all_gather.

Mega-kernel: `ttl_sparse_attn_seq_head_parallel`. Compute body:

1. KV cache slot update (one chip per col holds the slot for this position;
   write is local; replicated within row).
2. For each chip's local `T_local = T_pad / 8` slots, filter the topk_idxs
   that fall in the chip's slice (predicate `i*T_local ≤ idx < (i+1)*T_local`).
   The mask path stays the same as today's `_idxs_int_tile_to_idxs_and_mask`,
   just per-chip on `T_local`.
3. Local gather: `kv_local_gather = embedding(idxs_local, kv_local)`.
4. Local scores: `scores_local = Q_head @ kv_local_gatherᵀ` → `[H_local, K_local]`.
5. Cross-col softmax: per-row `max → all_reduce(max, cluster_axis=1) → sub →
   exp → sum → all_reduce(sum, cluster_axis=1) → divide`.
6. Local output: `o_local = probs_local @ kv_local_gather` → `[H_local, head_dim]`.
7. Cross-col `all_reduce(sum, cluster_axis=1)` for output.
8. Inverse rotary on `o[..., -rd:]`.
9. Block-diag `wo_a` matmul: each col holds 1 group of wo_a. No CCL inside.
10. `wo_b` matmul, K = G × R = 8192. Row-K-shard, col-N-shard.

The whole zone is one tt-lang kernel with three fabric all_reduces folded
into the compute body via `ttnn.experimental` collective primitives that the
kernel can issue directly. Per-step work shrinks 32× on gather, 8× on
score/output (head-shard), 8× on output (seq-shard cross-col). Net per-chip
work is gather + score + output on `K_local = 80` (was 640) × `H_local = 8`
(was 64).

Post-CCL: `reduce_scatter(cluster_axis=0)` on the wo_b K-partials. Output
stays col-N-sharded for the next zone.

### Zone 4: hc_post + ffn_norm + hc_pre + lanes (Lk-E)

After the sparse_attn output reaches the MHC zone, lane parallelism opens
back up:

| Lane | Submesh | Mega-kernel |
|---|---|---|
| Lane HC-POST | `mesh_row[0]` | `ttl_lk_e_hc_post_attn` (the `mhc_post_kernel` plus the residual-pack prologue and the FFN-side hc_pre prologue) |
| Lane FFN-NORM | `mesh_row[1]` | `ttl_lk_e_ffn_norm_to_a` (RMSNorm fused into the head of the next mega-kernel) |
| Lane SHARED | `mesh_row[2]` | `ttl_lk_e_shared_expert` (`w1, w3, clamp, silu, mul, w2` fused) |
| Lane GATE-FFN | `mesh_row[3]` | `ttl_moe_gate_full_pipeline` for non-hash layers (hash layers: `tid2eid` lookup) |

Lane SHARED's K-axis row-shard pairs with intra-row reduce_scatter; its
output is col-N-sharded. Lane GATE-FFN produces `(weights, indices)` topk
output replicated. Convergence: one `all_gather(cluster_axis=0)` to lift
both lane outputs to all rows.

### Zone 5: routed experts (full mesh)

Same shape as today: 8 experts per chip, indices/weights replicated, masked
selection and per-chip grouped MLP. The change is intra-chip: each of the
three matmuls (`w1`, `w3`, `w2`) becomes a ttl mega-kernel with PipeNet
K-split across cores (Kp = 16). Output is per-chip partial sum across the
8 local experts; one `all_reduce` across the full mesh produces the final
MoE output. The shared expert add happens inside the same kernel before the
all_reduce so the two paths fuse into one CCL boundary.

### Zone 6: head (full mesh)

`ttl_head_hc_combiner_norm_lmhead_topk1` fuses the entire head body:
hc_combiner (slice last row, `matmul(x, hc_fn_t)`, rsqrt mean, sigmoid,
multiply, add base, sigmoid, eps), final RMSNorm, lm_head matmul, per-chip
topk(k=1). One mega-kernel writes the per-chip top-1 into the pre-allocated
4-byte buffer the host pulls.

## Mega-kernel inventory

The hot path consists of these 16 fused tt-lang mega-kernels per layer
(Lk-A through Lk-F) plus one global head kernel and one global embed kernel.
No standalone tt-lang kernel below a mega-kernel survives in the hot path:
all of `rmsnorm`, `pre_norm_fn`, `pre_split_mixes`, `sinkhorn`,
`pre_apply_mix_h`, `post`, `act_quant_block`, `compressor_slot_shift`,
`compressor_softmax_sum_norm`, `swiglu` get inlined into the listed
mega-kernels' compute bodies. The standalone files in `tt-lang-kernels/`
remain as microbench harnesses.

| Mega-kernel | Layer phase | Submesh | Per-chip shape it operates on |
|---|---|---|---|
| `ttl_embed_pad_repeat_full_mesh` | global L0 | `mesh_full` | `[1, dim/32 = 128]` lookup, fan to `[Mpad, D/32 = 512]` |
| `ttl_lk_a_hc_pre_attn_norm_wq_a` | Lk-A (per layer) | `mesh_full` | hc_pre on full mesh K-split + attn_norm + wq_a 2D shard |
| `ttl_lk_b_qnorm_wq_b` | Lk-B | `mesh_full` | q_norm row-K, wq_b 2D shard |
| `ttl_lk_c_q_rotary_wkv` | Lk-C | Lane Q + Lane KV (rows 0+1) | rotary inline, wkv on lane KV |
| `ttl_lk_d1_kv_norm_rotary_actquant` | Lk-D1 | Lane KV (row 1) | fused, no CCL inside |
| `ttl_lk_d_idx_q_walsh` | Lk-D-idx-q (ratio=4 only) | `mesh_row_pair` (rows 2+3) | indexer wq_b + rotary + Hadamard |
| `ttl_lk_d_idx_cmp_state` | Lk-D-idx-cmp-* | `mesh_row[2]` | wkv + wgate + APE + 4 paged_update_cache |
| `ttl_lk_d_idx_emit` | Lk-D-idx-emit | `mesh_row[2]` | cssn + rotary + Hadamard + paged_update_cache |
| `ttl_lk_d_idx_score_topk` | Lk-D-idx-score + topk | `mesh_row_pair` | weights_proj + Q@KVᵀ + relu + sum + bucket pad-mask + topk |
| `ttl_lk_d_comp_attn` | Lk-D-comp (ratio in {4,128}) | Lane CMP (row 2) | attn-side compressor body |
| `ttl_lk_dsparse` | Lk-Dsparse | `mesh_full` | sparse_attn head+seq parallel + inverse rotary + wo_a + wo_b |
| `ttl_lk_e_hc_post_ffn_norm_hc_pre_shared_gate` | Lk-E | row lanes 0..3 | per-lane bodies as listed above |
| `ttl_lk_f_routed_experts` | Lk-F | `mesh_full` | grouped MLP + ksplit + mask + sum + add(shared_out) + all_reduce |
| `ttl_head_hc_combiner_norm_lmhead_topk1` | global Final | `mesh_full` | head body fused end-to-end |

`prebuild_ttl_decode_kernels` is rewritten to compile this list at startup.
The current `_TTL_KERNEL_CACHE` keys are all renamed to match the mega-kernel
names. The legacy `ttl_rmsnorm_*`, `ttl_mhc_norm_fn_ksplit_*`,
`ttl_mhc_split_mixes_*`, etc. globals are deleted from the hot path; the
standalone kernels keep their factories for the microbench harnesses in
`tt-lang-kernels/`.

## KV cache: sequence-sharded

Both KV caches are sequence-sharded on the col axis (8-way). Replication
across rows is removed: each `(row, col)` chip holds the same `T_local =
T_pad / 8` slots, and the row axis is reused for unrelated work (the
head-shard within sparse_attn). Per chip:

- `attn.kv_cache_tt`: `[1, T_local, head_dim = 512]` bf16, sharded on
  `cluster_axis=1`. At max_seq_len = 65536, T_local = 8192, so 8 MB per chip.
- `indexer.kv_cache_tt`: `[1, T_local_idx, index_head_dim = 128]` bf16. At
  T_local_idx = 16384/8 = 2048, 0.5 MB per chip.
- `indexer.compressor` and `attn.compressor` state buffers
  (`kv_state_*`, `score_state_*`): unchanged shape; replicated, since
  they are tiny (32 rows × 256 cols × 2 bytes = 16 KB).

Slot-write rule: at decode position `pos`, the cache slot index is
`slot_global = pos % win` (window) or `slot_offset + pos // ratio`
(compressed). The chip that owns that slot is `col = slot_global // T_local`.
The other 7 cols issue a no-op write. This is implemented by passing the
slot tensor and a per-chip-active mask to `paged_update_cache`; only the
owning chip executes the L1 store.

Per-step host upload changes: the `pre_stage` phase writes per-chip slot and
mask into the existing `_kv_slot_tt` buffer (now shape `[8, 1]` int32) and
a new `_kv_slot_owner_mask_tt` buffer (`[8, 1]` int32) before the trace
replays.

## Sparse_attn full kernel body

`ttl_lk_dsparse` is the largest single kernel in the model. It owns the
sparse_attn body, the inverse rotary, the block-diag wo_a, and the wo_b
matmul. Its trace-stable shapes:

- `Q`: `[1, S=1, H_local=8, head_dim=512]` head-sharded on cols.
- `kv_local`: `[T_local=8192, head_dim=512]` seq-sharded on cols.
- `topk_idxs`: `[1, K_total = win + index_topk = 128 + 512 = 640]` int32,
  replicated within the row (each row has its own copy after the prior
  zone's all_gather).
- `K_local` is variable per chip (depends on which idxs fall in the
  chip's slice). Padded to `K_local_max = K_total / 8 + slack` with
  `-inf` mask; this keeps the trace shape stable.

Compute body (one ttl operation):

1. Index filter: per-chip `idxs_local`, `valid_local` from the global
   `topk_idxs` and the chip's `(slot_lo, slot_hi)` constants.
2. Gather: `kv_gather = embedding(idxs_local, kv_local)`.
3. Score: `scores = Q @ kv_gatherᵀ + valid_local + sink_bias`, scale.
4. Concat sink column: `[scores | sink]`.
5. Softmax over `K_local_max + 1` columns. Cross-col reduction:
   `max → ttnn.experimental.fabric_all_reduce(cluster_axis=1, op=max) →
   sub → exp → sum → fabric_all_reduce(sum) → divide`.
6. Drop sink slice; weighted sum: `o = probs @ kv_gather`.
7. Cross-col `fabric_all_reduce(sum)` on `o`. Output `o`: `[H_local, head_dim]`.
8. Inverse rotary on `o[..., -rd:]`.
9. Permute `[B, S, H_local, D] → [G_local=1, B*S, H_local*D]`. With G=8
   and 1 group per col, no permute across chips needed.
10. `wo_a` matmul: `[1, B*S, in_per_group] @ [1, in_per_group, R]` → `[1, B*S, R]`.
11. `wo_b` matmul: K-row-shard × N-col-shard. Output is row-K-partial.
12. `reduce_scatter(cluster_axis=0)` on the wo_b output → col-N-sharded
    `wo_b_partial`.

Steps 5, 7, 12 are the three CCLs in the kernel. They are issued via
`ttl.fabric_all_reduce` primitives that the kernel can pipe through
PipeNet-equivalent fabric pipes. ttnn-level CCL ops are not used inside
the kernel body.

## CCL plan

Per-token CCL count target after the rewrite:

- Zone 0: 1 reduce_scatter (rows) on the hc_pre + attn_norm + wq_a row-K-partial.
- Zone 1: 4 reduce_scatter (cols, one per lane) + 1 all_gather (rows)
  to converge lane outputs.
- Zone 2 (indexer layers only, 20 of 43 layers): 3 fabric all_reduces inside
  `ttl_lk_d_idx_score_topk` + 1 reduce_scatter at its tail.
- Zone 3: 3 fabric all_reduces inside `ttl_lk_dsparse` + 1 reduce_scatter
  on the wo_b output.
- Zone 4: 4 reduce_scatter (cols, per lane) + 1 all_gather (rows).
- Zone 5: 1 all_reduce (full mesh) on the routed-experts + shared output.
- Zone 6: 1 reduce_scatter (rows) on the lm_head output.

Per-token total:

- 43 × (1 + 5 + 1 + 5 + 1) = 43 × 13 = 559 CCLs from zones 0/1/3/4/5
- 20 × 4 = 80 CCLs from indexer layers
- 1 head + 1 embed = 2 global CCLs
- Total ≈ 641 CCLs/token vs 419 today.

The rewrite trades ~50% more CCLs for ~10× more useful compute. The CCLs
are smaller (each is on a single-row or single-col reduction, not the full
mesh) so per-CCL latency drops. The fabric throughput envelope is what's
budgeted against; the count is not.

## Tracing

The whole step is captured in the existing two-trace (no-emit / emit)
shape. Constraints:

- Every mega-kernel uses pre-allocated output buffers
  (`output_tensor=` / `optional_output_tensor=` everywhere). No allocation
  inside the kernel body.
- Submesh dispatch is set up before `begin_trace_capture` via the four
  `mesh_row[i]` handles. The trace records per-chip command queues; lanes
  on disjoint chip sets execute concurrently at replay.
- `pre_stage` writes per-step host→device data
  (`input_ids`, `start_pos`, kv slots, kv slot owner masks, indexer
  T_active per bucket) into stable device buffers before
  `execute_trace`. None of these uploads happen inside the trace.
- Fabric all_reduces inside `ttl_lk_dsparse` and
  `ttl_lk_d_idx_score_topk` use `ttl.fabric_all_reduce` (the in-kernel
  primitive). These are part of the recorded body; they do not require
  a separate ttnn-level CCL op.
- The trace warmup loop runs 16 untraced steps, same as today, so every
  lazy `_alloc_decode_tensors` path fires before capture.

## Order of work

The rewrite lands in this sequence. Each step has a 2-token bit-exact
greedy gate against the prior step before merging.

1. **Submesh plumbing.** Open `mesh_full` plus the four `mesh_row[i]` handles
   in `_open_mesh`. Wire every existing `Device*` class to take an explicit
   `submesh` parameter. No behavior change yet; default is `mesh_full`.
2. **Sequence-sharded KV cache.** Reshape `attn.kv_cache_tt` and
   `indexer.kv_cache_tt` to the col-axis seq-shard layout. Update
   `paged_update_cache` call sites to pass per-chip slot + owner mask.
   Sparse_attn body is still ttnn-replicated at this point but reads the
   seq-sharded cache through an all_gather wrapper. Validate end-to-end.
3. **Sparse_attn rewrite.** Replace the ttnn-op chain with the
   `ttl_lk_dsparse` mega-kernel including the three fabric all_reduces.
   Drop the all_gather wrapper from step 2. Validate.
4. **Lane parallelism: Zone 1.** Build `ttl_lk_a_*`, `ttl_lk_d1_*`,
   `ttl_lk_d_idx_cmp_state`, `ttl_moe_gate_full_pipeline`. Bind to
   `mesh_row[0..3]`. Add the convergence all_gather at the lane boundary.
   Validate.
5. **Lane parallelism: Zone 4.** Same shape for `ttl_lk_e_*` mega-kernels
   and the FFN-side gate/shared lanes. Validate.
6. **2D weight shard for the remaining linears.** `wq_b`, `wo_b`, `wo_a`,
   `lm_head`, `shared_expert.w1/w2/w3`, `compressor.wkv/wgate`,
   `indexer.wq_b/weights_proj`, `indexer.compressor.*`. Each becomes a
   `ttl_*` mega-kernel with row-K + col-N shard and a row-axis
   `reduce_scatter`. Validate after each.
7. **MHC mesh-wide ksplit.** Replace `ttl_mhc_norm_fn_ksplit_K512_Kp8`
   with `ttl_mhc_norm_fn_split_sinkhorn_apply_mesh_ksplit` running on
   `mesh_full` with 32 × Kp=16 split. Fold the existing standalone
   `pre_split_mixes`, `sinkhorn`, `pre_apply_mix_h`, `mhc_post`,
   `rmsnorm` into the mega-kernel post-reduce stage.
8. **Routed experts intra-chip ksplit.** Add per-chip K-fanout (Kp=16)
   to each of the three expert matmuls. Output is the same per-chip
   partial sum across local experts.
9. **Head fusion.** Build `ttl_head_hc_combiner_norm_lmhead_topk1`.
   Validate.
10. **Trace re-capture.** Re-record the two traces over the new mega-kernel
    sequence. Confirm 2 traces and not more. Measure tok/s.

Each step is a separate PR with its own PCC test and 2-token coherence
gate. The 8-token gate is run only at the end of each step, before merge.

## What this rewrite explicitly removes

- `DeviceColLinear` cols-only fallback path.
- The 4× row-replication on every linear's output (`_full_mesh = False` branch).
- Replicated KV caches (`kv_cache_tt` 32× broadcast).
- Standalone `DeviceRMSNorm` dispatches in the hot path.
- Standalone MHC kernel dispatches in the hot path.
- The single-chip `DeviceMHC` 32× replicated `fn_tt` path.
- The `ttnn.matmul + ttnn.all_gather` pattern across the entire model.
- The cols-only `ShardTensor2dMesh(dims=(None, -1))` branch.

The standalone kernel files in `tt-lang-kernels/` and their PCC tests
remain. Removing tt-lang kernels is forbidden; what is removed are the
ttnn-glue dispatches that wrap them. Every existing tt-lang kernel's body
is inlined into a mega-kernel listed above.
