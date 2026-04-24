"""CPU torch references for the V4-Flash priority kernels.

Transcribed from `../TileKernels/tile_kernels/torch/mhc.py` and friends so
this directory stands on its own. Keep these in sync with the TileKernels
reference; if the upstream reference changes, update here and note why.
"""
from __future__ import annotations

import torch


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """mHC Sinkhorn normalization.

    x: [..., m, m] (operates on the last two dims). Alternates row- and
    column-normalization for `repeat` iterations. First pass is a row-softmax
    followed by a column-normalize with eps added once to break symmetry.
    """
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x
