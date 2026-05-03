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

# Make `inference` importable. In-repo it lives two dirs up; on the
# remote sandbox the file is colocated with this module under /tmp.
_HERE = pathlib.Path(__file__).resolve().parent
for _candidate in (_HERE, *_HERE.parents):
    if (_candidate / "inference.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

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


def benchmark(label: str, thunk, mesh,
              warmup: int = 2, runs: int = 5,
              sleep_ms: int = 5) -> float:
    """Warmup + timed runs. Returns min wall-time in seconds.

    `thunk()` should run the full kernel chain end-to-end. The kernel is
    expected to write into pre-allocated outputs (no per-call allocs).
    """
    import time
    import ttnn
    for _ in range(warmup):
        thunk()
    ttnn.synchronize_device(mesh)
    times = []
    for _ in range(runs):
        if sleep_ms:
            time.sleep(sleep_ms / 1000)
        t0 = time.perf_counter()
        thunk()
        ttnn.synchronize_device(mesh)
        times.append(time.perf_counter() - t0)
    best = min(times)
    median = sorted(times)[len(times) // 2]
    print(f"[bench {label}] best={best * 1e3:.3f}ms "
          f"median={median * 1e3:.3f}ms over {runs} runs")
    return best


def download_chip0(mesh, mesh_shape, t_tt) -> torch.Tensor:
    """Download a replicated tensor and return chip (0,0)'s view, sliced
    back to the logical shape (mirrors `_readback_replicated_2d` in
    inference.py)."""
    import ttnn
    composer = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape, dims=(1, 0))
    raw = ttnn.to_torch(t_tt, mesh_composer=composer)
    shape = tuple(t_tt.shape)
    return raw[tuple(slice(0, s) for s in shape)]
