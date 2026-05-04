# tt-lang bugs / missing primitives blocking pure-ttl mega kernels

Catalog of issues hit while porting Final-head topk and the Lk-D-topk /
Lk-F-gate-post topk paths off ttnn. Each entry has a minimal repro, the
observed behavior, and the workaround we landed.

## 1. argmax_2pass loses within-tile column resolution at large NUM_ITERS

**Where:** `test_final.py` Final-head topk(k=1) over `VOCAB=129280`
(`n_total_tiles=4040` rounded to `4064`, `K_block=32` → `NUM_ITERS=127`).

**Repro:** the standalone `tt-lang-kernels/argmax_2pass.py` works correctly at
`N=4064, K=32, NUM_ITERS=4` (per-chip slice), returning argmax=393 on a
random fp32 input. Lifting the same kernel to `NUM_ITERS=127` over the full
vocab returns the **tile-aligned start** of the correct answer's tile
(e.g. 113280 instead of 113306 = 113280 + 26).

Confirmed via debug downloads:
- `values_padded_fp32[0, 113306] = 1.359375` (the global max)
- Kernel returns `(top_val=1.359375, top_idx=113280)` → value is correct,
  index is rounded down to the tile-start.

That is, Pass-1 reduce_max yields the right global max value, but Pass-2's
`reduce_max(i + sign(v - max_global) * BIG, dims=[0,1])` over a `(1, K_block)`
block resolves to a within-tile column of 0 instead of 26 for the
winning tile.

**Hypotheses considered (none confirmed):**
- `ttl.math.broadcast(scalar, dst_shape=(1, K_block), dims=[0,1])` may only
  fill the first tile of a multi-tile-col block. Ruled out by the working
  `argmax=393` (not tile-aligned) at `NUM_ITERS=4, K=32` with the same
  block shape.
- bf16→fp32 typecast collisions: the input is bf16-quantized fp32, so many
  adjacent indices share `1.359375`. But torch.argmax over the **same**
  values_padded_fp32 finds 113306 — so the ties don't explain rounding to
  the tile start.
- fp32 DEST accumulation: adding `fp32_dest_acc_en=True` to the argmax
  decorator did not change the result.

**Related but distinct upstream issues:**
- tenstorrent/tt-lang #533 — `reduce_max/reduce_sum dims=[1]` returns zeros
  for fp32 tiles. We use `dims=[0, 1]` here, which the issue says works.
- tenstorrent/tt-lang #2878 — "argmax operator fails accuracy tests when
  shapes are large" (open).

**Workaround landed in test_final.py:** drop the tt-lang argmax pipeline,
use `to_layout(RM) → ttnn.argmax(use_multicore=True, keepdim=True) +
ttnn.max` (variant B from `argmax_2pass.py`'s docstring). Result:
top_idx PASS, top_val PASS, ttl=19.3ms vs ref=34.0ms (~1.76×).

**Open question:** the iterative-argmax B1b/B1c paths use much smaller
`NUM_ITERS` (4 and 8), where the standalone kernel is known to work.
The bug is tied to scale (loop count or total lane count), not the
algorithm shape itself. A targeted reproducer at `NUM_ITERS=127` with
strictly-increasing fp32 inputs (no ties) would help isolate whether
the cause is multi-tile reduce semantics or block-state handoff between
the global-max broadcast and Pass-2.

## 2. No sub-tile (lane) write primitive for iterative-topk output

**Where:** B1c — Lk-F gate post `topk(k=8)` over `N_ROUTED=256` (8 tiles).

To implement iterative argmax in tt-lang and emit `[TOPK]` indices as
output, each iteration `k ∈ [0, TOPK)` needs to write a single scalar
(`idx_k`) into lane `(0, k)` of the output tile. We have no clean
primitive for "store a scalar at a specific lane within a tile" — the
existing `ttl.copy(blk, out[r:r', c:c'])` is tile-granular.

The reasonable workarounds all have caveats:
- **Slot-mask + add accumulator:** maintain a running output tile
  `out_acc`; in iter k compute `out_acc += idx_k * lane_mask_k` where
  `lane_mask_k` is a precomputed [TILE, TILE] tile with a 1 at lane
  `(0, k)` and 0 elsewhere. Cost: `TOPK` extra full-tile loads + adds.
- **Per-iter full-tile broadcast:** each iter emits a tile with `idx_k`
  broadcast to all lanes; host post-processes by slicing lane k. Cost:
  `TOPK` separate output tiles of write bandwidth.
- **Single-tile output via element_write:** the `element_write`
  primitive (B-suite tasks #34/#37/#38, currently pending) would
  allow writing scalar `idx_k` to lane `(0, k)`. **This is the right
  primitive** but is not yet wired into our kernels.

**Workaround landed:** none yet. Lk-F continues to use ttnn.topk;
the topk + gather + post-processing chain remains a ttnn island.
Defer to after `element_write` lands (then revisit B1c).

## 3. Loop-based reserve/wait DFB pattern miscompiles to slot-0 reads

**Where:** previous gather-kernel work; surfaced again as a hypothesis
for the argmax bug above.

After tt-lang #537 (issue #536 fix), the simple consecutive `wait()`
case in `dm_write` works:

```python
def dm_write():
    blk = out_cb.wait()
    ttl.copy(blk, out[0:1, 0:1]).wait()
    blk = out_cb.wait()
    ttl.copy(blk, out[1:2, 0:1]).wait()
```

But the multi-tile gather pattern still produces slot-0 data on
iteration 1:

```python
for c in range(N):
    with reserve() as r:
        r.store(...)
    with wait() as w:
        ...  # always reads slot 0 instead of slot c
```

**Status:** we avoid this pattern in mega kernels by using `ttnn.embedding`
for kv-cache gathers (Lk-Dsparse) and by structuring per-iter computation
with explicit DFBs holding (1,1) state across iters.

## 4. Stale: `Cannot Compare Non-Integer Values` on conditional kernel bodies

**Symptom:** when adding a `transpose_b: bool = False` flag to a SUMMA
kernel constructor with an `if transpose_b:` branch inside the
`@ttl.compute()` body, the kernel fails to compile with
`Cannot Compare Non-Integer Values` at the `@ttl.operation` decorator
line — even when the flag is False at trace time.

A related symptom: defining `bt_cb` only inside `if transpose_b:` (so
it's an unbound free variable in the False case) hits
`ValueError: Cell is empty` during cb-config collection.

**Workaround landed (in test_lk_d_idx_score.py):** factored the SUMMA
kernel into two top-level constructors — `_make_summa_matmul_kernel` and
`_make_summa_matmul_b_t_kernel` — each with a fully-monomorphic body.
Cleaner anyway.

## 5. tt-lang fp32 reduce path issues (background)

**Filed:** tenstorrent/tt-lang #533 — `reduce_max/reduce_sum dims=[1]`
returns zeros for fp32 tiles. Workaround documented in the issue:
`@ttl.operation(options="--no-ttl-reduce-full-fp32")`.

**Impact on this work:** none directly — our argmax_2pass uses
`dims=[0, 1]`, which the issue says works in fp32. But it is the same
fp32_reduce_acc lowering path, so #1 above may share the same root
cause once isolated.
