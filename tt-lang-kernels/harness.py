"""Shared test-harness helpers for the V4-Flash TT-Lang kernels.

The helpers fall into three groups:
  - PCC / tolerance checks (pcc, assert_pcc).
  - Pack/unpack between a small 4x4-per-slice torch tensor and a
    32x32-per-slice ttnn tensor. V4-Flash mhc ops all operate on 4x4
    subblocks; we embed each slice into the top-left of a 32x32 tile,
    padding with either a neutral value (zero) or a softmax-sentinel
    (very negative). The pack helpers hide this detail from callers.
  - Constant-tile builders (mask, eps_mask, scaler).
"""
from __future__ import annotations

from typing import Tuple

import torch


TILE = 32
# exp(x - row_max_of_valid) must underflow to 0. -1e4 gives exp(-1e4 - anything) = 0
# in fp32 without tripping overflow after subsequent subtracts (fp32 max ~3.4e38).
PAD_SENTINEL = -1e4


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation on flattened fp32 values. >0.9995 is our fp32 bar."""
    a, b = a.flatten().float(), b.flatten().float()
    a_m, b_m = a - a.mean(), b - b.mean()
    num = (a_m * b_m).sum()
    den = (a_m.norm() * b_m.norm()).clamp_min(1e-12)
    return (num / den).item()


def assert_pcc(expected: torch.Tensor, actual: torch.Tensor, threshold: float = 0.9995) -> Tuple[float, float]:
    """Compute PCC + max abs diff, print a one-line report, raise on failure.

    NaN PCC (e.g. from a NaN in the output) is treated as a failure. Returns
    (pcc, max_abs_diff).
    """
    import math

    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: expected={tuple(expected.shape)} actual={tuple(actual.shape)}")
    p = pcc(expected, actual)
    d = (expected.float() - actual.float()).abs().max().item()
    ok = (not math.isnan(p)) and (not math.isnan(d)) and p > threshold
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] pcc={p:.6f} max_abs_diff={d:.4e} (threshold={threshold})")
    if not ok:
        raise AssertionError(f"PCC {p:.6f} max_abs_diff {d:.4e} failed threshold {threshold}")
    return p, d


def pack_4x4_slices(x: torch.Tensor, pad_value: float = 0.0,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Pack a [num_tokens, 4, 4] tensor into [num_tokens * 32, 32] at `dtype`.

    Each 4x4 slice lives in the top-left of its own 32x32 tile. The rest of
    the tile is filled with `pad_value`. For softmax-style ops use
    `PAD_SENTINEL` so exp(pad - row_max) underflows cleanly.
    """
    if x.shape[-2:] != (4, 4):
        raise ValueError(f"expected trailing dims (4, 4), got {tuple(x.shape)}")
    num_tokens = x.shape[0]
    out = torch.full((num_tokens, TILE, TILE), pad_value, dtype=dtype)
    out[:, :4, :4] = x.to(dtype)
    return out.reshape(num_tokens * TILE, TILE).contiguous()


def unpack_4x4_slices(packed: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Inverse of pack_4x4_slices: [num_tokens * 32, 32] -> [num_tokens, 4, 4]."""
    if packed.shape != (num_tokens * TILE, TILE):
        raise ValueError(f"expected ({num_tokens * TILE}, {TILE}), got {tuple(packed.shape)}")
    return packed.reshape(num_tokens, TILE, TILE)[:, :4, :4].contiguous()


def mask_tile(valid: int = 4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A 32x32 tile with 1s in the top-left `valid`x`valid` region, 0s elsewhere."""
    m = torch.zeros(TILE, TILE, dtype=dtype)
    m[:valid, :valid] = 1.0
    return m


def eps_mask_tile(eps: float, valid: int = 4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A 32x32 tile with `eps` in the top-left `valid`x`valid` region, 0s elsewhere.

    Used to add eps only inside the valid region without polluting the padded
    sentinel cells.
    """
    m = torch.zeros(TILE, TILE, dtype=dtype)
    m[:valid, :valid] = eps
    return m


def scaler_tile(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A 32x32 tile of ones, used as the scaler input for reductions."""
    return torch.ones(TILE, TILE, dtype=dtype)
