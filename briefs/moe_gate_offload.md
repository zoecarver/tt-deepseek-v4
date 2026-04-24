# Brief: Offload the MoE Gate to the Tenstorrent mesh

## Your goal

Move the **MoE `Gate`** module (`inference.py:822`) from CPU onto a 1×4 Blackhole
mesh. This is 43 per-layer scoring linears that run on every decode token. Each
gate matmul is `[dim=4096] × [n_routed_experts=256, dim=4096]`, followed by
softmax/sigmoid and top-k.

You do **not** need the real model weights. You will validate correctness on a
machine that does not have the 150 GB state-dict cached. Use **random tensors
at model shape** against a pure-PyTorch reference.

Once your PCC test passes, the main thread integrates your code into the full
model; integration is not part of this brief.

## Non-goals

- **Do not touch attention code** (the primary thread owns it).
- **Do not build a fused MoE kernel.** Gate only: one linear + softmax + topk.
- **Do not port the hash-layer branch** (`self.hash == True`). v1 supports the
  non-hash branch only; document the limitation and `raise` if `hash` is set.
- **Do not optimize perf.** On-device correctness + clean handoff is the bar.

## The compute you must match

From `inference.py:840-859`, stripped to the non-hash path:

```python
def forward(self, x, bias, weight):
    # x:      [B, S, dim]              bf16 or fp32
    # weight: [n_routed_experts, dim]  (usually bf16 / fp32, no FP8 here)
    # bias:   [n_routed_experts]       float32
    scores = F.linear(x.float(), weight.float())             # [B, S, n_experts]
    scores = scores.softmax(dim=-1)                           # score_func="softmax"
    original_scores = scores
    scores = scores + bias                                    # biased scores
    indices = scores.topk(topk, dim=-1)[1]                    # [B, S, topk]
    weights = original_scores.gather(-1, indices) * route_scale
    return weights, indices
```

Model shapes for `DeepSeek-V4-Flash` (confirm via `config.json`):
- `dim = 4096`
- `n_routed_experts = 256`
- `n_activated_experts (topk) = 8`
- `route_scale ≈ 2.446` (float; doesn't affect correctness of the device path,
  it's a post-multiply)
- `score_func = "softmax"` (the config value for V4-Flash; support this branch
  only)

## What to build

Add a new class in `inference.py` in the device-offload section (near
`DeviceColLinear`, `DeviceRowLinear`):

```python
class DeviceMoEGate(nn.Module):
    """Device-resident MoE gate for one MoE layer.

    Shards the expert-scoring weight [n_experts, dim] col-parallel on the
    1xN mesh (expert dim). Matmul runs on device; softmax/topk/gather stay
    on host.
    """

    def __init__(self, mesh, cpu_weight: torch.Tensor, bias: Optional[torch.Tensor],
                 topk: int, route_scale: float):
        super().__init__()
        self._linear = DeviceColLinear(mesh, cpu_weight)   # reuse existing
        self.bias = bias              # keep as host tensor, tiny
        self.topk = topk
        self.route_scale = route_scale

    def forward(self, x, input_ids=None):
        # MUST match Gate.forward signature so it drops into MoE.forward.
        B, S, D = x.shape
        scores = self._linear(x).float()          # [B, S, n_experts]
        scores = scores.softmax(dim=-1)
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(-1, indices) * self.route_scale
        return weights, indices
```

Plus a wiring method on `Model` parallel to `offload_attn_wq_b`:

```python
def offload_moe_gate(self, mesh):
    self._device_gate = []
    for layer in self.transformer.layers:
        if not isinstance(layer.ffn, MoE):
            continue
        g = layer.ffn.gate
        if g.hash:
            raise NotImplementedError("hash-gate offload not supported (v1)")
        dw = DeviceMoEGate(
            mesh, g.weight, g.bias,
            topk=g.topk, route_scale=g.route_scale,
        )
        layer.ffn.gate = dw
        self._device_gate.append(dw)
```

Plus a CLI flag `--offload-moe-gate` that opens the mesh (if not already open)
and calls `model.offload_moe_gate(mesh)`.

## Known gotcha (important)

The primary thread just uncovered that **`self.weight.data` strips custom
attributes off `nn.Parameter`**. If the weight has a `.scale` attribute (FP8),
passing `.data` loses it. The Gate weight is **bf16 or fp32** (no FP8 scale),
so this likely doesn't affect you — **but pass `g.weight`, not `g.weight.data`**
defensively. `DeviceColLinear.__init__` already accepts either.

Also: **add an assertion in `DeviceColLinear`** (or before you call it) that
errors if `cpu_weight.dtype in (float8_e4m3fn, float4_e2m1fn_x2)` and
`getattr(cpu_weight, "scale", None) is None`. This catches the bug for the
whole codebase, not just the gate.

## Your standalone PCC test

Place at `scripts/test_moe_gate.py`. Does not need the real model. Does not
need inference.py to fully import (can minimally import the three classes it
needs via a small shim if import is too slow — but full import is fine).

```python
"""PCC test for DeviceMoEGate vs CPU reference. Uses random tensors at model shape."""
import torch
import torch.nn.functional as F
import ttnn
from inference import DeviceMoEGate  # plus DeviceColLinear indirectly

DIM = 4096
N_EXPERTS = 256
TOPK = 8
ROUTE_SCALE = 2.446
B, S = 1, 1

def cpu_ref(x, w, bias, topk, route_scale):
    scores = F.linear(x.float(), w.float()).softmax(dim=-1)
    original = scores
    biased = scores + bias
    indices = biased.topk(topk, dim=-1)[1]
    weights = original.gather(-1, indices) * route_scale
    return weights, indices, original

def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    num = ((a - a.mean()) * (b - b.mean())).sum()
    den = (((a - a.mean())**2).sum().sqrt() * ((b - b.mean())**2).sum().sqrt()).clamp_min(1e-9)
    return (num / den).item()

def main():
    torch.manual_seed(0)
    x = torch.randn(B, S, DIM, dtype=torch.bfloat16) * 0.5
    w = (torch.randn(N_EXPERTS, DIM, dtype=torch.float32) / DIM**0.5).to(torch.bfloat16)
    bias = torch.randn(N_EXPERTS, dtype=torch.float32) * 0.01

    w_ref, idx_ref, scores_ref = cpu_ref(x, w, bias, TOPK, ROUTE_SCALE)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        gate = DeviceMoEGate(mesh, w, bias, topk=TOPK, route_scale=ROUTE_SCALE)
        w_tt, idx_tt = gate(x)

        # Score PCC (pre-topk scores must match closely)
        # You will need to expose `original_scores` from DeviceMoEGate for this
        # test — add a debug hook or run once returning scores, once returning topk.
        # (Simplest: add a `return_scores=True` kwarg guarded by an `if`.)

        # Top-k set agreement: for each (B, S) position, the sets of chosen
        # expert indices should overlap heavily. Order may differ when scores
        # are near-tied (bf16 rounding).
        set_ref = set(idx_ref.flatten().tolist())
        set_tt  = set(idx_tt.flatten().tolist())
        overlap = len(set_ref & set_tt) / len(set_ref)

        # Weight value PCC on matched indices
        wp = pcc(w_ref, w_tt)

        print(f"topk set overlap: {overlap:.3f} (expect >= 7/8 = 0.875)")
        print(f"weights PCC: {wp:.5f} (expect >= 0.999)")
        assert overlap >= 7/8, "too many expert picks differ"
        assert wp >= 0.999, "gate weight PCC too low"
        print("PASS")
    finally:
        ttnn.close_mesh_device(mesh)

if __name__ == "__main__":
    main()
```

Add a **second test** that stress-tests with harder inputs:
- `B=1, S=32` (seq dim non-trivial)
- Inputs with wider dynamic range (`* 3.0` instead of `* 0.5`)
- Ten different seeds; report worst-case PCC and set overlap.

## Running your test

This project uses a remote Tenstorrent Quiet Box. The skill that hides the
SSH/docker plumbing is `tt-connect-remote-device` (see project `CLAUDE.md`).

```bash
export TT_REMOTE_CONF=/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/sterling-all.conf
# Smoke test (first time only):
/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/smoke-test.sh

# Iterate on the HW (simulator is out of date for this project):
/Users/zcarver/.claude/skills/tt-connect-remote-device/scripts/run-test.sh --hw scripts/test_moe_gate.py
```

Put any throwaway probes in `/tmp/`, not `scripts/`.

## What "done" looks like

1. `DeviceMoEGate` class in `inference.py`.
2. `Model.offload_moe_gate` method and `--offload-moe-gate` CLI flag.
3. `scripts/test_moe_gate.py` passing on `--hw` with:
   - PCC ≥ 0.999 on `original_scores` output
   - Top-k set overlap ≥ 7/8 across 10 seeds on synthetic bf16 random data at
     model shape.
4. The FP8-scale-missing assertion added to `DeviceColLinear`.
5. A short `NOTES.md` (~20 lines) in the PR description: what you built, the
   shapes, the PCC numbers, and any surprises.

**Do not attempt to run the full model** to validate end-to-end coherence —
that requires the 150 GB state dict you don't have. Standalone PCC is your bar.
The primary thread runs the full-model coherence gate after merging your work.

## Ask-for-help triggers

Stop and ask only if:
- `smoke-test.sh` fails.
- `run-test.sh` hangs (device stuck). See `CLAUDE.md` "Hang recovery".
- PCC is < 0.99 even with synthetic `w * 0.01` (very small values) — that would
  imply a shape or sharding bug in `DeviceColLinear` on the gate shape (256
  out-dim across 4 chips = 64 per chip, must be tile-aligned to 32 → 2 tiles per
  chip; fine).
- You discover `DeviceColLinear` does not work at all on this shape (it should;
  `wq_a` offload passes at similar shape).

Everything else — file layout, test naming, how you scaffold probe scripts —
your call.
