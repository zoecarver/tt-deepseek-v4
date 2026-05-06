"""Seam dump helper for cross-pipeline PCC triage.

Both `inference.py` (legacy) and `run_mega.py` (mega) call `dump(...)` at
named seams. The two pipelines call into the same names with the same
schedule, so post-hoc PCC of matching files identifies the first
divergent point.

Activation: this module is **disabled by default** (`_ENABLED = False`).
To enable, edit `_ENABLED = True` below, run each pipeline, then flip
back. There are no env-var knobs because run-test.sh does not propagate
them.

Pipeline selection: each script calls `set_pipeline("legacy")` or
`set_pipeline("mega")` once at startup. This determines the sub-dir
under `_BASE`. Dumps go to `{_BASE}/{pipeline}/T{tok:02d}_L{layer:02d}_{name}.pt`
(layer=-1 for non-layer seams).

Filters: edit `_TOKENS` and `_LAYERS` below to limit dumps.
"""
from __future__ import annotations
import os
from pathlib import Path
import torch


_ENABLED = False
_BASE = "/home/ubuntu/zcarver/seam_dumps"
_TOKENS: set[int] | None = {0}        # set to None for all
_LAYERS: set[int] | None = None        # set to None for all (or e.g. {0, 1, 2})

_DIR: Path | None = None
_PIPELINE: str | None = None


def set_pipeline(name: str) -> None:
    """Call once at script startup to select the dump sub-dir."""
    global _DIR, _PIPELINE
    _PIPELINE = name
    if not _ENABLED:
        return
    _DIR = Path(_BASE) / name
    _DIR.mkdir(parents=True, exist_ok=True)
    print(f"[seam_dump] enabled pipeline={name} dir={_DIR} "
          f"tokens={_TOKENS} layers={_LAYERS or 'all'}")


def enabled() -> bool:
    return _ENABLED and _DIR is not None


def dump(name: str, layer: int, tok: int, t) -> None:
    """Save one tensor at a named seam.

    `t` may be a torch.Tensor, a ttnn.Tensor, or None. None is skipped.
    ttnn tensors are downloaded with `to_torch` on chip 0 (caller should
    only dump tensors that are replicated, or a chip-0 shard view).
    """
    if not _ENABLED or _DIR is None or t is None:
        return
    if _TOKENS is not None and tok not in _TOKENS:
        return
    if _LAYERS is not None and layer != -1 and layer not in _LAYERS:
        return
    arr = _to_torch(t)
    if arr is None:
        return
    fn = _DIR / f"T{tok:02d}_L{layer:02d}_{name}.pt"
    torch.save(arr, fn)


def _to_torch(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().contiguous()
    try:
        import ttnn
    except Exception:
        return None
    if isinstance(t, ttnn.Tensor):
        try:
            full = ttnn.to_torch(t, mesh_composer=None)
        except Exception:
            full = ttnn.to_torch(t)
        return full.detach().cpu().contiguous()
    return None
