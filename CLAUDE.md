# DeepSeek-V4-Flash bring-up on Tenstorrent

Goal: bring DeepSeek-V4-Flash up on Tenstorrent hardware (Quiet Box, 4 chips, 64 GB DRAM total).

## Setup (do this first, every session)

Load these skills before doing any work:
- `tt-connect-remote-device`
- `ttnn`
- `tt-lang`

Remote config for this project: `/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/all.conf` (ssh to `bh-qbae-15`, docker container `zcarver-ird-all`).

Verify the remote is reachable with the smoke test:

```
export TT_REMOTE_CONF=/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/all.conf
/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/smoke-test.sh
```

For any Python file that imports or defines `ttl`/tt-lang kernels, run it via `scripts/run-test.sh` from the `tt-connect-remote-device` skill. Note: the functional simulator is out of date for this work, so going straight to `--hw` is fine and likely necessary. Pure-CPU scripts can be copied with `copy-file.sh` and run with `remote-run.sh`.

## Hang recovery

**Important: follow these steps exactly. Deviating can leave the device in an unrecoverable state. If anything is unclear, stop and ask the user before running commands.**

### Detecting a hang vs. normal slow progress

The inference script must print clear progress markers so it is obvious whether the device is genuinely hung or just doing slow work:
- Print after **weight loading** completes.
- Print after **weight transfer to device** completes, **before any kernels run**.

If the last marker is "weights loaded" but not "weights transferred", the process is likely still moving data to the device (not hung). If both markers printed and then it stalled, that is a real kernel-level hang.

Also check the **first ~20 lines of the log** for a message indicating another process holds a lock on the device. A device lock will show up early in the log, not as a hang mid-run, and the fix is to clear the other process rather than reset the device.

### Recovery steps

When a hang is confirmed:

1. Kill Python processes **inside the docker container** using `remote-run.sh`:
   ```
   /Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/remote-run.sh pkill -9 python
   ```

2. SSH into the remote host (**not** into the docker container) and run `tt-smi -r`:
   ```
   ssh zcarver@bh-qbae-15 tt-smi -r
   ```

**Do NOT run `tt-smi -r` from inside the docker container.** Running it from docker will put the device in an unrecoverable state.

## Inference script shape

We want **one inference script** that contains all kernels and all inference logic. Do not fragment across many files.

Model it after nanochat's inference script (see `../nanochat`):
- A clean `Model` class.
- Dedicated methods for loading tensors (weights, KV cache setup, device placement).
- A dedicated `step_decode` (or equivalent) for a single decode step.
- Keep the code clean and readable; this file is the reference implementation, not scaffolding.

## Phases

Work proceeds in three phases. Do not skip ahead; each phase is a prerequisite for the next.

### Phase 1: CPU-only

Get the model loading and running end-to-end on CPU first. This is the correctness baseline.

- Download weights from HF (`deepseek-ai/DeepSeek-V4-Flash`).
- Load the model with `transformers.AutoModelForCausalLM.from_pretrained(..., dtype="auto")`.
- Run a short generation and confirm the output is coherent text. This is the validation bar for every later phase.

### Phase 2: Offload post-MoE to QB

The Quiet Box has **64 GB DRAM across 4 chips**. Weights must be sharded per chip. Use existing sharding references in the sibling repos listed below (do not invent a new sharding scheme).

- Start by offloading the post-MoE path to QB.
- Then offload whatever else fits on device.
- Keep the CPU path as the fallback for anything that does not fit or is not yet supported.
- Re-run the Phase 1 coherence check after each offload change; regressions are not allowed to land.

### Phase 3: Optimize

Loop:
1. Confirm the model produces coherent text (validation). **This gate is load-bearing. Do not skip it.**
2. Measure tok/s.
3. Make one optimization.
4. If tok/s improves **and** validation still passes, commit. Otherwise revert.
5. Repeat.

## References

Only these sibling directories are in scope for exploration:

- `../tt-metal`
- `../tt-lang`
- `../engram`
- `../nanochat`
- `../oasis-ttlang`
- `../tt-lang-import`

In-repo reference:
- `./TileKernels` — DeepSeek's reference tilelang kernels for important ops (FP8/FP4 GEMM, sparse attention, HC Sinkhorn, etc.). **Reference only.** Tilelang does not run on Tenstorrent hardware, so these kernels cannot be used as-is. Consult them to understand the intended compute/dataflow, then re-implement in tt-lang for the device.

Rules:
- Do **not** explore `tt-lang/third_party/` (or any `third_party/` inside these repos). The authoritative copy of `tt-metal` lives at `../tt-metal`, which is more up to date than any vendored copy.
- Do **not** do global searches across `~` or `~/Developer`. Stay inside this repo and the six paths above.
