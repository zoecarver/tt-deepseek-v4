# DeepSeek-V4-Flash TT-Lang kernels

TT-Lang kernels for the 5 priority ops listed in `../kernels.md`. Modeled after
`../../tt-lang-import/harness/`: one file per kernel, each self-contained with
a `solve()` entry point plus a `__main__` test harness that runs both smoke
shapes and the actual V4-Flash production shapes.

## Layout

- `harness.py` — shared helpers (PCC, mask-tile builders, slice pack/unpack).
- `torch_refs.py` — CPU torch reference implementations (transcribed from
  `../TileKernels/tile_kernels/torch/mhc.py` so this repo stands alone).
- `sinkhorn.py` — kernel #1: mHC Sinkhorn normalization.
- `bug_repros/` — minimal reproducers for compiler bugs hit during bring-up.
- (one file per kernel going forward)

## Known compiler bugs + workarounds

### Loop-reassignment drops the in-place add (tt-lang#527)

Filed as [tenstorrent/tt-lang#527](https://github.com/tenstorrent/tt-lang/issues/527).
See `../tt-lang/benchmarks/matmul/ksplit_kernel.py:118-133` for a live workaround
in tree.

Inside a `scf.for` (any Python `for _ in range(N)` inside `@ttl.compute`)
an in-place `p += x` or reassignment like `p = p + x` drops the add. Fix is
the explicit reserve/wait pingpong on a partial DFB:

```python
# Broken (inside scf.for): compiler drops the +=.
#   for _ in range(K-1):
#       r = recv_cb.wait()
#       p += r
# Workaround: materialise prev, reserve a fresh block, store prev + r.
for _ in range(K - 1):
    prev = partial_cb.wait()
    r = recv_cb.wait()
    new = partial_cb.reserve()
    new.store(prev + r)
```

Applies to our sinkhorn iteration loop (the `repeat - 1` alternations). We
already use this pattern via the persistent-block pattern described below.

### reduce dims=[1] returns zeros on fp32 tiles

Filed as [tenstorrent/tt-lang#533](https://github.com/tenstorrent/tt-lang/issues/533).

`ttl.math.reduce_max` / `ttl.math.reduce_sum` with `dims=[1]` silently returns
all zeros when the input tile is `fp32`; `bfloat16` works. `dims=[0]` and
`dims=[0, 1]` are unaffected. The broken path is the fp32-accumulation
reduce lowering.

**Workaround:** pass `options="--no-ttl-reduce-full-fp32"` to `@ttl.operation`.
See `bug_repros/reduce_dim1_fp32_returns_zeros.py` (shows the bug) and
`bug_repros/reduce_dim1_nested_with.py` (shows the workaround passing).

```python
@ttl.operation(grid=(1, 1), options="--no-ttl-reduce-full-fp32")
def my_kernel(...):
    ...
```

This affects every softmax/RMSNorm-shaped kernel in this directory.

## TT-Lang patterns to remember

Cribbed from `../tt-lang-import/prompt.md`; the ones that have bitten us here.

- **Reduce fusion fix.** Reduce output cannot be held in a Python variable and
  used in a downstream expression. Store it immediately into its own DFB,
  then `wait()` in the next op. Failing this trips a
  `failed to legalize 'ttl.reduce'` error.

  ```python
  red_dfb.reserve().store(ttl.math.reduce_sum(x, sc, dims=[1]))
  # next op reads via red_dfb.wait()
  inv_dfb.reserve().store(ttl.math.recip(red_dfb.wait()))
  ```

- **Broadcast's second arg is the reserved output block it stores into.**
  Never pass an arbitrary waited block as the shape template.

  ```python
  with out_dfb.reserve() as o:
      o.store(ttl.math.broadcast(red_dfb.wait(), o, dims=[1]))
  ```

- **Python `for _ in range(N)` inside `@ttl.compute` becomes `scf.for`,
  not an unrolled loop.** State reassigned to Python variables does not carry
  across iterations. To carry state, use a persistent reserved block held in
  an enclosing `with dfb.reserve() as s:` and rewrite it each pass via
  `s.store(...)` (see `layernorm_minimal` / the sinkhorn kernel here).

- **fp32 tiles for row softmax.** Apply the `--no-ttl-reduce-full-fp32`
  workaround above; pad cells that must underflow under exp should use a
  sentinel far below the valid-region max but within fp32 range. We use
  `PAD_SENTINEL = -1e4` (defined in `harness.py`) — anything more extreme
  risks overflow after subsequent subtracts.

## Running

This project uses a separate remote from the main V4-Flash bringup so the two
agents don't collide. Always use `sterling-all`:

```bash
export TT_REMOTE_CONF=/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/sterling-all.conf
/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/run-test.sh --hw sinkhorn.py
```

The simulator is out of date for this work, so `--hw` is the default path.

## Test harness shape

Every kernel file's `__main__` block runs:

1. **Smoke**: smallest valid shape (e.g. `n1=1`). PCC check vs torch ref.
2. **V4-Flash**: the production shapes called out in `../kernels.md`. Same
   PCC check.

PCC target: `>0.9995` for fp32 kernels, `>0.999` for fp16/bf16.
