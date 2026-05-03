"""Shared helpers for mega-kernel PCC tests.

The reference for each mega-zone runs the *exact* ttnn / ttl op chain that
`inference.py` runs between two CCLs today (sans CCLs). The placeholder
mega kernel writes nothing today, so PCC is expected to fail until the
kernel is implemented; once it lands, it should reproduce the reference's
output bit-equivalent (or within the bf16/fp32 PCC bar).

Everything imported from `inference.py` is the canonical implementation;
do not re-derive op chains here — copy them faithfully so later perf
comparisons are apples-to-apples.
"""
from __future__ import annotations

import math
import pathlib
import sys

# Make `inference` importable from the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

# Default mesh shape for these tests. 1×4 = QB-sized; max 4 cards per
# the test bar (no Galaxy). Drop to (1, 1) inside an individual test if
# the kernel does not need any sharding.
DEFAULT_MESH_SHAPE = (1, 4)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    am, bm = a - a.mean(), b - b.mean()
    num = (am * bm).sum()
    den = (am.norm() * bm.norm()).clamp_min(1e-12)
    return (num / den).item()


def report_pcc(name: str, expected: torch.Tensor, actual: torch.Tensor,
               threshold: float = 0.999) -> bool:
    if expected.shape != actual.shape:
        print(f"[{name}] FAIL shape mismatch: expected={tuple(expected.shape)} "
              f"actual={tuple(actual.shape)}")
        return False
    p = pcc(expected, actual)
    d = (expected.float() - actual.float()).abs().max().item()
    ok = (not math.isnan(p)) and p > threshold
    status = "PASS" if ok else "FAIL"
    print(f"[{name}] {status} pcc={p:.6f} max_abs_diff={d:.4e} (threshold={threshold})")
    return ok


def open_mesh(shape=DEFAULT_MESH_SHAPE):
    import ttnn
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape))


def close_mesh(mesh):
    import ttnn
    ttnn.close_mesh_device(mesh)


def replicate(mesh, t: torch.Tensor, dtype, layout=None):
    """Upload a host tensor as fully-replicated across the mesh."""
    import ttnn
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    return ttnn.as_tensor(
        t.contiguous(), device=mesh, dtype=dtype, layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def alloc_replicated(mesh, shape, dtype, layout=None,
                     fill_dtype=torch.bfloat16):
    """Allocate a zero-initialised replicated ttnn tensor of the given
    shape/dtype. `fill_dtype` is just the host buffer dtype used to seed
    the device tensor; it does not affect the on-device dtype."""
    import ttnn
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    return ttnn.from_torch(
        torch.zeros(*shape, dtype=fill_dtype),
        device=mesh, dtype=dtype, layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def download_chip0(mesh, mesh_shape, t_tt) -> torch.Tensor:
    """Download a replicated tensor and return chip (0,0)'s view, sliced
    back to the logical shape (mirrors `_readback_replicated_2d` in
    inference.py)."""
    import ttnn
    composer = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape, dims=(1, 0))
    raw = ttnn.to_torch(t_tt, mesh_composer=composer)
    shape = tuple(t_tt.shape)
    return raw[tuple(slice(0, s) for s in shape)]
