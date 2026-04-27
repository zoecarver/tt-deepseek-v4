# Tracing plan: per-span traces stitched with Python glue

Goal: capture the largest possible regions of `_block_forward` as ttnn traces
without rewriting kv-cache slot indexing, rotary indexing, or topk-K to be
device-tensor-driven. Each span is one `begin_trace_capture` /
`end_trace_capture` pair, with a `_compute_body()` warmup beforehand.

## Status (2026-04-27): tracing parked; revisit AFTER mesh-utilization wins

A first attempt at re-enabling per-class traces (`DeviceSharedExpert` +
`DeviceMoEGate`, opt-in via `--trace`) hung the device. 83 traces were
captured in 1 s, then SIGKILL during the first decode step. The hang was
not isolated to a single op family. Findings before pausing:

- **Production deepseek_v3 uses one trace per decode step, not per-class**
  (`tt-metal/models/demos/deepseek_v3/tt/generator.py:900-1055`). Per-step
  state (token id, position, rot idxs, page table) lives in pre-allocated
  device tensors that Python writes via `copy_host_to_device_tensor` before
  each `execute_trace(..., blocking=True)`. We can't do one trace per step
  here because `start_pos` drives Python-side slicing of cos/sin, kv-cache
  slots, and topk-K. The brief's per-span shape (below) is the right
  fallback.
- **Per-class granularity is too fine.** 83 separate `execute_trace` calls
  per decode step costs more orchestration than it saves. Spans should
  group consecutive ops that share inputs/outputs, not one op chain per
  Device* class.
- **`blocking=False` likely contributed to the hang.** With non-blocking
  replay, the Python-level `ttnn.copy(x_tt, dev._x_upload_tt)` between
  classes can race the trace replay reading from `_x_upload_tt`. Production
  uses `blocking=True`.
- **JIT during replay is suspect.** The first decode step was still
  JIT-compiling tt-lang kernels for non-traced classes (7 "Compiled kernel
  ready" lines printed before the SIGKILL). Mixing JIT with trace replay is
  plausibly bad.
- **Trace region size was not the bottleneck.** We had 100 MB; production
  uses ~36 MB.

### When we revisit tracing

1. **Start with one class, not all of them.** Pick a single Device* whose
   `_compute_body` is well-behaved (all `output_tensor=` arguments, no
   slice/reshape/pad/transpose). `DeviceSharedExpert` is the strongest
   candidate.
2. **Use `blocking=True`** on `execute_trace` until we have a known-working
   single-class trace; revisit non-blocking only after that baseline holds.
3. **Run a few real decode steps before `_capture_trace`.** All tt-lang
   kernels must be JIT'd and cached before any trace records the op
   sequence; otherwise trace replay races a fresh JIT.
4. **Validate with the 2-token bit-exact gate** before adding a second
   span. Bisect on hang.

The per-span shape below is still the structural target after the
single-class smoke test passes; the order of spans by amortization is
unchanged.

## Spans to trace (per layer)

Order is by recommended landing order (largest amortization first):

1. **MoE Path D body** — `MoE._forward_device_routed_cached`. Selection mask
   + grouped MLP (`_fp4_gemm_via_bfp4` x3) + masked sum + `ttnn.all_reduce`
   + add of shared expert. Pure device, all I/O via `_w*_tt` / `x_tt` /
   `weights_tt` / `indices_tt` / `shared_out_tt`. Shared expert body
   (`DeviceSharedExpert._compute_body`) already has `_capture_trace`; reuse
   that or fold it into the same span.
2. **MHC pre body** — kernel chain `norm_fn → split_mixes → sinkhorn →
   apply_mix`. Glue around the kernels (slice/reshape/pad/transpose at
   2465-2502 / 2581-2611) needs work — see "blockers" below.
3. **MHC post body** — `_post_kernel` plus the matching slice/reshape/pad/
   transpose stash retrieval.
4. **Attention Q-stack** — `wq_a → q_norm → wq_b → _device_q_rsqrt_norm →
   slice(nope|rope) → rotate(rope) → concat`. Cos/sin slice at `start_pos`
   stays OUT of the trace; rotated input is a pre-allocated device tensor
   that Python writes each step.
5. **Attention KV-stack** — `wkv → kv_norm → slice(nope|rope) → rotate(rope)
   → concat → act_quant`. Same pattern as Q-stack for cos/sin.
6. **Sparse_attn body** — `DeviceSparseAttn.forward_device(q, kv, idxs,
   valid, S, K)`. K must be fixed at the captured value across replays;
   today K is built from `win + maybe-compress-k`, both dynamic. See
   "fixes required".
7. **Attention output stack** — inverse-rotary (slice/concat) → block-diag
   `wo_a` matmul (permute/reshape) → `wo_b`.

Gate compute body (`DeviceMoEGate._compute_body`) and DeviceLMHead already
have `_capture_trace`; leave them as-is.

## What STAYS ON HOST / Python glue between spans

These are Python ints / control flow that change per step. Capturing them
inside a span would freeze the value at capture time and replay it
incorrectly.

- **Rotary cos/sin slice indices**: `[start_pos, 0]` and `[start_pos+1,
  rd_half]` (lines 3329-3330, 3633-3636, 3780-3781). Python computes the
  slice; result is passed in as the trace's pre-allocated cos/sin input.
- **kv_cache slot writes**: `ttnn.kv_cache.update_cache_for_token_(...,
  start_pos % win, 0)` (line 3710), `slot_in_ape = start_pos % ratio`
  (3181), `slot = ratio + slot_in_ape` (3190). All happen outside the
  trace.
- **win_table row slice**: `_window_topk_row_for_pos(start_pos, win)` →
  `ttnn.slice(self._win_idxs_table_tt, [0, win_row, 0], ...)` (3673-3677).
  Python builds `topk_idxs_dev`; trace receives it.
- **Compressor early-out and slot-shift loop**: `if (start_pos + 1) % ratio
  != 0: return None` (3206), the `for buf in ...` slot-shift block
  (3258-3264). The whole compressor stays Python-driven; we do not trace
  it. Different op sequences emit/non-emit can't share one capture.
- **Indexer dynamic-K topk**: `k = min(index_topk, T_active)` with
  `T_active = (start_pos + 1) // ratio` (3680, 3686). Indexer body is not
  a trace span. Either keep the indexer fully Python-driven, or apply
  pad-and-mask (fix K to `index_topk`, mask invalid positions) before
  trying to capture it.
- **Hash-gate `input_ids` upload**: `gate_dev.forward_device(input_ids)`
  calls `copy_host_to_device_tensor(host_mesh, self._input_ids_tt)` (line
  1414). Per-step host→device copy is not allowed inside a captured body.
  Hash layers (`layer_id < n_hash_layers`, default 3) skip Path D entirely
  for this reason.
- **Per-block `start_pos` flow control** in `_block_forward`: the layer
  loop and the call into `device_comp.forward_device` / indexer remain
  Python.

## Mechanical blockers and fixes required

1. **slice/reshape/pad/transpose hangs in trace bodies** (per
   `feedback_trace_slice_reshape_pad`). Affects MHC pre/post (lines
   2465-2502, 2581-2611), `block.hc_post` repeat→pad→reshape (741-749),
   q/kv slice+rotate+concat, topk_idxs concat (3700-3702),
   `_mhc_post_to_a_tt` (1717-1724). Each span listed above is a candidate
   only after we replace those with: (a) pre-allocated layout-transformed
   buffers populated by Python before the captured region, or (b) a
   tt-lang kernel that does the layout transform without ttnn.slice/
   reshape/pad/transpose.
2. **Sparse_attn span needs fixed K**. Either always run with `K = win +
   index_topk` and pad invalid slots with `-1` (already a sentinel
   handled by the mask path), or hold off on this span until topk is
   pad-and-masked.
3. **Per-call alloc inside spans**. Every `ttnn.reshape` / `ttnn.slice`
   / `ttnn.pad` / `ttnn.concat` / `ttnn.repeat` / `ttnn.transpose` that
   doesn't take `output_tensor=` allocates fresh on each call. Inside a
   trace replay, the captured addresses must be stable. Anything inside
   a span needs to either accept `output_tensor=` or be hoisted into
   Python glue.
4. **Slice's preallocated-output rejects sub-tile widths** (lines
   2378-2381). Slices into `mhc=4` / `mhc*mhc=16` widths can't go through
   `output_tensor=`. Those slices stay outside spans.
5. **First-call lazy host upload**. ttnn.rms_norm and a few other ops
   trigger a host upload on first call. Every span's `_compute_body()`
   must run once outside the trace before `begin_trace_capture` (the
   pattern already used by `DeviceLMHead._capture_trace`,
   `DeviceMoEGate._capture_trace`, `DeviceRMSNorm._capture_trace`).

## Span scaffolding

For each new span, add to the owning `Device*` class:

- `_alloc_decode_tensors()` — extend if the span needs new pre-allocated
  inputs/outputs. `Model.allocate_decode_tensors()` (line 4745) re-runs
  these before `begin_trace_capture`.
- `_compute_body()` — the exact op sequence to capture. All ops use
  `output_tensor=` / `optional_output_tensor=` against pre-allocated
  buffers. Must run identically inside and outside the trace.
- `_capture_trace()` — call `_compute_body()` once as warmup,
  `synchronize_device`, then `begin_trace_capture` → `_compute_body()` →
  `end_trace_capture`. Store `self._trace_id`.
- `forward_device()` — at runtime, fill the pre-allocated inputs (from
  Python glue), then `ttnn.execute_trace(self.mesh, self._trace_id,
  cq_id=0, blocking=False)`.

## What we're explicitly NOT doing

- Not tracing `_block_forward` end-to-end. The compressor's two-mode emit
  + indexer dynamic-K + hash-gate `input_ids` upload + start_pos-baked
  rotary/kv indices each independently block whole-block capture.
- Not lifting `start_pos` to a device tensor. Out of scope until ttnn
  exposes runtime-variable slice indices and kv-cache slot tensors and
  topk-K. The pad-and-mask path (fixed K with sentinel masking) is the
  cheaper alternative if we ever want to expand sparse_attn / indexer
  into a span.
- Not touching hash-gate layers' MoE. They keep the Python path; the
  trace win is concentrated in the 40 non-hash layers' Path D bodies.
