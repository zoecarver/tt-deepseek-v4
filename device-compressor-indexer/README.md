# DeviceCompressor / DeviceIndexer bring-up

Same playbook as `../tt-lang-kernels/`: build the device port of `Compressor` and
`Indexer` (from `../inference.py`) **out of tree**, validate each case at full
production shapes against the CPU PyTorch reference, and only then drop the
finished classes into `inference.py`.

## Goal

Move the entire `Indexer.forward` (decode path) and `Compressor.forward` (decode
path) onto the 1×4 mesh. State buffers (`kv_state`, `score_state`,
`indexer.kv_cache`) live on device; no per-step download/upload of activations
or state.

## Why a subdir

The integrated path inside `inference.py` is brittle: state buffers live on
both CPU and device for the duration of bring-up, and bug-hunting against the
real model loop is slow (3-minute weight load per attempt). The subdir hosts a
synthetic harness that:

1. Instantiates a real `Compressor` / `Indexer` with **production shapes** but
   tiny `max_seq_len`, so kv_caches are small and tests run in seconds.
2. Drives multiple decode steps in sequence, mutating CPU state as the model
   would.
3. After every step, compares CPU and device state buffers + outputs
   element-wise (PCC + max-abs).
4. Only after every test in this dir passes do we wire the device classes into
   `inference.py` and run the full coherence gate.

## Strategy

Each case is a single self-contained Python file. They share `harness.py` (PCC
helpers, mesh setup, reference state snapshotters) and `torch_refs.py` (a
trimmed copy of the CPU Compressor / Indexer logic — just the decode branch we
actually port).

Cases, smallest-first:

| # | File                          | Compressor / Indexer mode                     | Notes                                                                       |
| - | ----------------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| 0 | `compressor_no_overlap.py`    | Compressor: ratio=128, overlap=False, rotate=F | Ratio=128 Attention layer. Smallest path: linears → APE-add → state-update → softmax-sum → norm → rotary → kv_cache update. |
| 1 | `compressor_overlap.py`       | Compressor: ratio=4,   overlap=True,  rotate=F | Ratio=4 Attention layer. Adds the `kv_state[:, :ratio] = kv_state[:, ratio:]` shift, two-stripe slice/concat, and a bigger state buffer (coff=2 → kv_state shape `[B, 8, 2*head_dim]`). |
| 2 | `compressor_overlap_rotate.py`| Compressor: ratio=4,   overlap=True,  rotate=T | Indexer's internal Compressor. Adds Walsh-Hadamard rotate on the output. (`fp4_act_quant` is a no-op under bf16 policy and is omitted; we'll revisit if/when fp4 becomes a goal.) |
| 3 | `indexer.py`                  | Full `Indexer.forward` decode path             | wq_b → reshape → rotary → rotate_activation → call DeviceCompressor → einsum vs `indexer.kv_cache` → relu·weights → reduce → topk. |
| 4 | `dropin.py`                   | Integration shims                              | Imports the validated classes, exposes `wrap_layer(attn)` to monkey-patch into `inference.py`. |

Each test runs **at production shapes** (`args.dim=4096`, `args.head_dim=512`,
`args.index_head_dim=128`, etc.) using `args.max_seq_len=256` so the kv_cache
length and number of decode steps are small.

## Pass criteria (per case)

- PCC ≥ 0.99 on every device output vs CPU reference at every decode step.
- Element-wise max-abs error on `kv_state`, `score_state`, and `kv_cache` ≤ a
  tight bf16 round-off bound (tracked per case; tightened as we build).
- Final kv_cache matches CPU after running through ratio×2 decode steps so
  multiple compress events land.

## Precision policy

bf16 throughout. The CPU `act_quant` (block=64, fp8 e4m3) and `fp4_act_quant`
(block=32, ue8m0 + FP4 lookup) calls are **omitted** on the device path. Both
are essentially identity under bf16 (clamps don't fire, the fp* casts are the
lossy step), and re-introducing them on device requires real fp* emulation
kernels that aren't a current goal. The token sequence may shift slightly vs
the prior CPU-fp* gate; we accept that as long as text stays coherent.

## tt-lang vs ttnn — default to tt-lang

**Default: write a tt-lang kernel.** Drop to ttnn primitives only when one of
the following is true:

- The op is a single ttnn primitive that already has a hand-tuned tt-lang
  equivalent in `../tt-lang-kernels/` (use the existing kernel via the same
  drop-in pattern as `inference.py` already does for rmsnorm and MHC).
- The op is fundamentally easier in ttnn because it relies on primitives we
  don't have a clean tt-lang story for yet (e.g. `topk` along the last dim,
  `update_cache_for_token_` slot writes, `ttnn.embedding`). For these, use
  ttnn but add a TODO and leave the boundary clean so a tt-lang version can
  drop in later.
- The op is a single matmul against a constant table (e.g.
  `_device_rotate_activation` is `ttnn.matmul(x, H_d)`); ttnn matmul is
  already optimal and tt-lang would just wrap a matmul kernel.

Everything else — elementwise chains, fused reductions, broadcast adds, and
multi-stage compute that touches the same tile multiple times — should go in
tt-lang. Each new kernel goes in `../tt-lang-kernels/` as its own file with
a `solve()` + `__main__` PCC test using `harness.py` and `torch_refs.py`.

### tt-lang kernel candidates from this bring-up

In priority order; each lands in `../tt-lang-kernels/<name>.py`:

1. **`compressor_softmax_sum_norm`** — fuse the weighted softmax-then-reduce
   *plus* the RMSNorm that always follows it on every compress event:
   ```
   y_unnormed[B, d] = sum_i softmax(score_state[:, i, :], dim=i)[B, i, d] * kv_state[B, i, d]
   y[B, d] = rmsnorm(y_unnormed, gamma, eps)
   ```
   Both ops walk the same `[B, ratio, d]` tile region and the intermediate
   `y_unnormed` is otherwise written to DRAM and immediately re-read just to
   normalize. Fused, this is one DRAM read of `kv_state` + `score_state`, an
   online softmax + accumulate, and an in-tile rsqrt-normalize — and we get
   to drop the M-tile padding hack the standalone `DeviceRMSNorm` needs at
   `B=1`. Models exist in `../tt-lang-kernels/softmax_stage.py` (online
   softmax over a streamed axis), `../tt-lang-kernels/attention_matmul.py`
   (softmax × value), and `../tt-lang-kernels/rmsnorm.py` (the rsqrt-
   normalize this would inline). Composing them is the right next kernel
   to write once the ttnn s0 path passes.
2. **`compressor_state_shift`** — overlap=True only: `kv_state[:, :ratio] =
   kv_state[:, ratio:]` and the same for `score_state`. ttnn has
   `update_cache_for_token_` per slot but no batched slot-shift. A small
   tt-lang kernel that streams `[B, ratio, d]` blocks from the back half to
   the front half is cleaner than ratio sequential ttnn writes.
3. **`compressor_apply_rope_one_pos`** — apply rotary at a single freq slot
   to `kv_normed[..., -rd:]`. We already have
   `inference.py:_device_apply_rotary_interleaved` built from ttnn slice +
   complex-pair multiply; promotable to a fused tt-lang kernel that takes
   `(kv, cos, sin)` and writes the rotated output in place. Probably one of
   the simplest gain-over-ttnn fusions.
4. **`indexer_score_reduce`** — `relu(scores[B,S,H,T]) * weights[B,S,H,1]
   .sum(dim=H)`. Pure elementwise + 1D reduce; small kernel, big fusion win
   because `H=64` makes the broadcast multiply expensive in ttnn.
5. **`act_quant_e4m3_block`** — already TODO'd in `inference.py:_device_act_
   quant_block`. Block-wise amax + clamp + (future) fp8 e4m3 emulation. Top
   candidate for a tt-lang fusion.

### Already-tt-lang ops we reuse here

- `../tt-lang-kernels/rmsnorm.py` — drives the `Compressor.norm` step.
- `../tt-lang-kernels/softmax_stage.py` — model for the weighted-softmax-sum.
- `../tt-lang-kernels/attention_matmul.py` — model for softmax × value
  reductions.
- `inference.py:_device_rotate_activation` — single `ttnn.matmul` against
  the Sylvester-Hadamard table; already optimal.

## Run

```bash
export TT_REMOTE_CONF=/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/all.conf
# One case:
/path/to/run-test.sh --hw device-compressor-indexer/compressor_no_overlap.py
# All:
/path/to/run-test.sh --hw device-compressor-indexer/run_all.py
```

Each case prints PCC + max-abs vs CPU and exits non-zero on regression.
