# DeepSeek-V4-Flash bring-up on Tenstorrent

Goal: bring DeepSeek-V4-Flash up on Tenstorrent hardware (Quiet Box, 4 chips, 64 GB DRAM total).

## Autonomous task

This is an autonomous task. Use your best judgement and keep making progress without pausing for questions. Do NOT ask the user to confirm design decisions, file layouts, or next steps — make a reasonable call, document it in code/notes, and move on. Commit incrementally so each step is revertible.

Pause and ask only if one of the following happens:
- A device hang or unrecoverable state (see Hang recovery below).
- `tt-smi` reports hardware errors, or the smoke test fails and you cannot diagnose it from the remote logs.
- A destructive action is needed (force push, wiping caches, removing weights, `rm -rf` beyond per-run scratch files).
- You hit an ambiguity that would meaningfully change the architecture (e.g. choosing a fundamentally different sharding scheme than the references suggest).

Everything else — naming, file structure, minor refactors, choice of initial offload target, which kernel to write first — is yours to decide.

## Precision policy

Activations are **bf16**. On-device weights are bf16 except for routed-expert weights, which use native bfp4_b loaded from the preprocessed cache (`offload_moe_routed_experts` pattern) — at 256 experts/layer × 43 layers the bf16 footprint does not fit on the mesh, and bfp4_b is the only practical path. The bfp4 path is pre-dequant once at offload + `ttnn.typecast(bfloat4_b)`; the hot path is a plain matmul against the native-bfp4 weight.

## Setup (do this first, every session)

Load these skills before doing any work:
- `tt-connect-remote-device`
- `ttnn`
- `tt-lang`

Remote config for this project: `/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/galaxy.conf` (ssh to `g08blx03.tenstorrent.net`, docker container `ubuntu-ird-all`).

Verify the remote is reachable with the smoke test:

```
export TT_REMOTE_CONF=/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/galaxy.conf
/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/smoke-test.sh
```

For any Python file that imports or defines `ttl`/tt-lang kernels, run it via `scripts/run-test.sh` from the `tt-connect-remote-device` skill. Note: the functional simulator is out of date for this work, so going straight to `--hw` is fine and likely necessary. Pure-CPU scripts can be copied with `copy-file.sh` and run with `remote-run.sh`.

### Running scripts on the remote (streaming & monitoring)

**Prefer `run-test.sh`** whenever possible. It already streams output to a file, truncates the on-screen view, and leaves the full log on disk so you can `grep`/`cat` it after the run. No need to pipe `2>&1 | tail -N` yourself — that blocks on the pipeline and hides live progress.

When `remote-run.sh` is required (e.g. for scripts that open a mesh directly, or pure CPU runs), **always tee output to a log file on the remote** so progress can be monitored while the process is still alive:

```
remote-run.sh "cd /tmp && python3 -u myscript.py ... 2>&1 | tee /tmp/mylog.out"
```

To check progress mid-run:

```
remote-run.sh tail -50 /tmp/mylog.out
remote-run.sh cat /tmp/mylog.out | grep '\[phase\]'
```

**Do not** wrap the remote command with `| tail -N` on its own — that buffers and the user can't see anything until the process exits.

### Scratch / probe scripts go in /tmp, not the repo

Ad-hoc, one-shot probe scripts (checking a ttnn API signature, listing device ops, etc.) should live in `/tmp` locally and be copied to the remote, NOT committed under `scripts/`. `scripts/` is for scripts that have durable value (e.g. `test_<op>.py` PCC tests, `prompts/*.txt`). If a probe starts getting reused, graduate it into `scripts/` with a clear name; otherwise delete it when you're done.

## Hang recovery

**Never attempt hang recovery yourself. Do not run `pkill`, `tt-smi`, `tt-smi -r`, or any other recovery command. If you suspect the device is hung or in a bad state, stop and ask the user — they will handle recovery.**

### Detecting a hang vs. normal slow progress

The inference script must print clear progress markers so it is obvious whether the device is genuinely hung or just doing slow work:
- Print after **weight loading** completes.
- Print after **weight transfer to device** completes, **before any kernels run**.

If the last marker is "weights loaded" but not "weights transferred", the process is likely still moving data to the device (not hung). If both markers printed and then it stalled, that is a real kernel-level hang.

Also check the **first ~20 lines of the log** for a message indicating another process holds a lock on the device. A device lock will show up early in the log, not as a hang mid-run.

The exact log line to grep for is:
```
Waiting for lock 'CHIP_IN_USE_1_PCIe' which is currently held by thread TID: <pid>, PID: <pid>
```
If you see this, the current run is queued behind a stale process and will not progress. Stop and ask the user to clear it.

### When a hang is suspected

Stop and ask the user. Do not run any recovery commands.

## Inference script shape

We want **one inference script** that contains all kernels and all inference logic. Do not fragment across many files.

Model it after nanochat's inference script (see `../nanochat`):
- A clean `Model` class.
- Dedicated methods for loading tensors (weights, KV cache setup, device placement).
- A dedicated `step_decode` (or equivalent) for a single decode step.
- Keep the code clean and readable; this file is the reference implementation, not scaffolding.

## Phases

Work proceeds in three phases. Do not skip ahead; each phase is a prerequisite for the next.

### Phase 1: CPU-only DONE

Get the model loading and running end-to-end on CPU first. This is the correctness baseline.

- Download weights from HF (`deepseek-ai/DeepSeek-V4-Flash`).
- Load the model with `transformers.AutoModelForCausalLM.from_pretrained(..., dtype="auto")`.
- Run a short generation and confirm the output is coherent text. This is the validation bar for every later phase.

### Phase 2: Offload post-MoE to QB DONE

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

Notes: 
* optimize however you see fit, ideally by fusing and writing tt-lang kernels. Do not remove tt-lang kernels as mentioned below, even if it improves performance.
* optimize with galaxy in mind: eventually we are going to move to a galaxy and bring moe routing (expert selection) on device, so focus on the parts after that. Don't worry about wall time, just worry about post-moe decode. Try to focus on distributing work in a way that will scale to 32 cards.
* make sure you measure the correct thing: enable tracing and only measure post-moe-routing work, don't measure device transfers that will be removed in the future, those are uninteresting. Syncronize the device before measuring perf with prints and time. You can break work out to micro optimize. You can add sleeps if needed to make sure timing is correct. Many kernels are jit'ed and cached, so only measure perf after the first few tokens are generated and only in the hot paths that we are optimizing (not moe routing that's still on host or other device transfers that will be moved).
* Maybe try doing a longer token generation to see how times amortize.
* Host operations and tensor transfers to device swing all over the place, so the wall time will vary a lot, don't let this distract you. For example, running inference twice in a row with two tokens generated might be anywhere from 180s to 240s at current wall time.

Ideas for optimization:
* Make sure everything is on device in the hot path and tensors are allocated ahead of time.
* Enable tracing (there is a skill for this and some referenced models have tracing enabled)
* Improve sharding and card utilization: use more of the 4 cards, distribute data more, etc.
* Move tensors to L1
* Fuse kernels together and move more to tt-lang, use pipes to mcast and communicate. Try fusing matmul with elementwise kernels.
* Look at metal references to see other ideas for highly optimized models.
* Try things measure perf, if you have a crazy idea try it and revert if it doesn't work.
* If you can think of a creative way to improve moe routing, great, but lower priority.
* Currently we use nn.Module, but this might add some overhead, we could restructure to just use vanilla python for classes and functions. This might be a lot cleaner and better and not allow there to be any implicit lifecycle (inits being called too much) or graph building that might add overhead. At some point I'd like you to refactor to just super simple clean python functions that are called in loops and can be organized in classes if you'd like.
* You are now on galaxy, you can use it! We have 32 cards, see how much you can distribute work across the mesh :)

## References

Only these sibling directories are in scope for exploration:

- `../tt-metal`
- `../tt-lang`
- `../engram`
- `../nanochat`
- `../oasis-ttlang`
- `../tt-lang-import`
- `../gemma` — multi-chip mesh reference (mesh setup, shard/replicate patterns)
- `../lingbot-world` — multi-chip mesh reference

In-repo reference:
- `./TileKernels` — DeepSeek's reference tilelang kernels for important ops (FP8/FP4 GEMM, sparse attention, HC Sinkhorn, etc.). **Reference only.** Tilelang does not run on Tenstorrent hardware, so these kernels cannot be used as-is. Consult them to understand the intended compute/dataflow, then re-implement in tt-lang for the device.

Rules:
- Do **not** explore `tt-lang/third_party/` (or any `third_party/` inside these repos). The authoritative copy of `tt-metal` lives at `../tt-metal`, which is more up to date than any vendored copy.
- Do **not** do global searches across `~` or `~/Developer`. Stay inside this repo and the six paths above.

Make sure to check metal-reuse.md, the kernel you want might already exist.

IMPORTANT, LATEST STATUS: you are working on mega kernels, please make sure to read  /Users/zcarver/Developer/deepseek/tt-lang-kernels/mega/README.md at the start of each session.