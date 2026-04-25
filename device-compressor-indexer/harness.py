"""Shared helpers for DeviceCompressor / DeviceIndexer bring-up cases.

PCC + max-abs comparators, mesh open/close context, and a thin loader for the
real `inference` module so each case can grab `Compressor`, `Indexer`,
`DeviceColLinear`, `DeviceRMSNorm`, and the device helpers (`_device_apply_
rotary_interleaved`, `_device_rotate_activation`, `_sylvester_hadamard`,
`_RMS_TILE`).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from typing import Any

import torch


def load_inference():
    """Locate and import the inference module from /tmp (remote) or repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "/tmp/inference.py",
        os.path.join(os.path.dirname(here), "inference.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("inf", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["inf"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"inference.py not found in any of: {candidates}")


def replicated(ttnn, mesh, t: torch.Tensor, layout=None, dtype=None):
    """Upload a CPU tensor to the mesh, replicated on every chip."""
    return ttnn.as_tensor(
        t.to(torch.bfloat16).contiguous(),
        device=mesh,
        dtype=dtype if dtype is not None else ttnn.bfloat16,
        layout=layout if layout is not None else ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def to_torch_replicated(ttnn, mesh, t_tt) -> torch.Tensor:
    """Read back a replicated tensor from chip 0 (axis-1 concat over the chips
    is then sliced; we only need chip 0 for replicated tensors)."""
    composer = ttnn.ConcatMesh2dToTensor(mesh, tuple(mesh.shape), dims=(1, 0))
    return ttnn.to_torch(t_tt, mesh_composer=composer)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() == 0:
        return 1.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def report(label: str, cpu: torch.Tensor, dev: torch.Tensor, pcc_min: float = 0.99,
           max_abs_max: float | None = None) -> bool:
    p = pcc(cpu, dev)
    m = maxabs(cpu, dev)
    flag = "OK" if p >= pcc_min and (max_abs_max is None or m <= max_abs_max) else "FAIL"
    print(f"  [{flag}] {label:<24}  PCC={p:.6f}  max|.|={m:.4f}  shape={tuple(cpu.shape)}")
    return flag == "OK"


@contextmanager
def open_mesh(ttnn, shape=(1, 4)):
    """Open a 1xN mesh with 1D fabric enabled (matches inference._open_mesh).
    Fabric is required for CCL ops like ttnn.all_gather used by DeviceColLinear."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def make_test_args(inf, max_seq_len: int = 256):
    """Production-shape ModelArgs trimmed to a short context window so kv-cache
    buffers stay tiny but the rest of the shapes match production."""
    args = inf.ModelArgs()
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    return args


def fresh_compressor(inf, args, compress_ratio: int, head_dim: int, rotate: bool,
                    seed: int = 0):
    """Return two clones (cpu_ref, cpu_dev_seed) of a Compressor with identical
    random weights and freshly-zeroed state. Caller wires them up the same way
    so both observe the same input stream."""
    g = torch.Generator().manual_seed(seed)

    def _build():
        m = inf.Compressor(args, compress_ratio=compress_ratio,
                           head_dim=head_dim, rotate=rotate).eval()
        for p in m.parameters():
            with torch.no_grad():
                p.copy_(torch.empty_like(p).normal_(generator=g) * 0.02)
        # Tie kv_cache + freqs_cis in the same way Attention does.
        m.kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, head_dim)
        rd = m.rope_head_dim
        m.freqs_cis = inf.precompute_freqs_cis(
            rd, args.max_seq_len, args.original_seq_len, args.compress_rope_theta,
            args.rope_factor, args.beta_fast, args.beta_slow,
        )
        return m

    cpu = _build()
    g = torch.Generator().manual_seed(seed)  # rebuild same weights
    dev = _build()
    dev.load_state_dict(cpu.state_dict())
    dev.kv_cache = torch.zeros_like(cpu.kv_cache)
    dev.freqs_cis = cpu.freqs_cis
    return cpu, dev
