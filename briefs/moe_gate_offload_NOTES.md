# MoE Gate offload — v1 notes

## Scope delta from brief

Two things landed different than the original brief:

1. **softmax/topk/gather moved to device.** Originally brief said keep those
   on host; mid-task the primary thread said everything goes to device. We
   comply: the only host-side work is the final `ttnn.to_torch` of the small
   `[M, topk]` weights and indices tensors.
2. **Not integrated into `inference.py`.** Primary thread is untangling
   attention and asked us to hold off. `DeviceMoEGate` lives in
   `scripts/device_moe_gate.py` as a standalone class; drop-in into
   inference.py later is straightforward (constructor already matches the
   brief's signature).

## What was built

- **`tt-lang-kernels/matmul_softmax.py`** — new fused tt-lang kernel:
  `softmax((x @ w), dim=-1)` with the matmul chunked in K-tiles and the
  row-softmax stage consuming the accumulator directly in L1. No DRAM
  round-trip between matmul and softmax. Single-core, `M == TILE == 32`.
  Standalone PCC test in the file.
- **`scripts/device_moe_gate.py`** — `DeviceMoEGate`:
  1. fused tt-lang matmul+softmax (above),
  2. `ttnn.add(bias)`,
  3. `ttnn.topk` on biased scores,
  4. `ttnn.gather(scores, indices)` on the **un-biased** softmax (matches the
     PyTorch reference: `original_scores.gather(-1, indices)`),
  5. `ttnn.multiply(route_scale)`.
- **`scripts/test_moe_gate.py`** — PCC test against PyTorch reference.

## PCC numbers (sterling, `--hw`)

V4-Flash gate shape: `dim=4096`, `n_routed_experts=256`, `topk=8`,
`route_scale=2.446`.

### Baseline — decode (M=1), x_scale=0.5

| metric       | value    |
|--------------|----------|
| scores_pcc   | 0.99996  |
| weights_pcc  | 0.99999  |
| topk overlap | 8/8      |

### Stress — M=32, x_scale=3.0, 10 seeds

| metric                 | worst    | mean    |
|------------------------|----------|---------|
| scores_pcc             | 0.99967  | ~0.9998 |
| weights_pcc            | 0.99925  | ~0.9996 |
| topk overlap (min row) | 7/8      | ~7.9/8  |
| topk overlap (mean)    | 0.980    | ~0.99   |

All seeds cleared the brief's 0.999 bar on both PCC metrics. Top-k set
overlap is 7/8 worst-case per row (expected — see "surprises" below).

## Surprises / gotchas

1. **Topk order-dependence tripped the first run.** Initial `weights_pcc`
   was 0.21 with perfect 1.0 overlap. Cause: `ttnn.topk` and `torch.topk`
   return `(values, indices)` sorted by value descending; when two experts
   have nearly tied biased scores, the ordering can swap between bf16 device
   and fp32 CPU. Positional PCC then compares misaligned slots. Fix: scatter
   each row's topk weights into a dense `[N_EXPERTS]` vector at their
   indices and PCC on that. Order-invariant, handles 7/8 overlap cleanly.

2. **7/8 overlap at the boundary is real, not a bug.** Softmax on 256
   experts produces ~1/256 ≈ 0.004 per score; adding a ~0.01 bias and then
   picking top-8 means the 8th vs 9th expert can trade places under bf16
   rounding. PyTorch CPU runs in fp32 throughout; the device runs softmax
   in bf16 via the tt-lang kernel. Accepted 6/8 as the assertion floor,
   observed 7/8 worst-case.

3. **bf16-everywhere holds.** The tt-lang kernel runs bf16 inputs, fp32
   accumulators in DST, bf16 outputs. Softmax reductions use a bf16 scaler
   with `--no-ttl-reduce-full-fp32` (same workaround as `rmsnorm.py` and
   `sinkhorn.py`). No FP8/BFP8 anywhere.

## Architectural note: Indexer reuse

The Indexer at `inference.py:666` has the same op graph — matmul →
softmax → topk — just at a different shape (head_dim × seqlen/ratio
instead of dim × n_experts). `matmul_softmax.py` is directly reusable for
its fused matmul+softmax phase; only the topk shape differs. Worth
keeping in mind when the sparse-attention path is scoped.

## Not yet done / follow-ups

- **Integration into `inference.py`**: add `DeviceMoEGate` wrapper class,
  `Model.offload_moe_gate(mesh)`, and `--offload-moe-gate` CLI flag.
  Primary thread's task once attention stabilizes.
- **Mesh sharding**: v1 runs on a single card. If the 1×4 mesh is
  preferred, replicate the weight across devices and run per-device;
  the tt-lang kernel itself is per-device so this is a ttnn
  `ReplicateTensorToMesh` change only.
- **`M > TILE` support**: the kernel hard-requires one tile-row. Extending
  to multi-tile-row prefill is a loop around the outer `m`-dim; not done
  since brief only called out decode (M=1) and M=32 stress.
- **FP8-scale assertion in `DeviceColLinear`**: deferred because that
  edits inference.py. Defensive only — gate weight is bf16/fp32, not FP8.
- **Fused full-gate tt-lang kernel**: the next obvious perf play is to
  fuse bias-add + topk + gather + scale into the existing matmul+softmax
  kernel. topk in tt-lang over 256 with k=8 is non-trivial (no primitive);
  would be a second brief.
