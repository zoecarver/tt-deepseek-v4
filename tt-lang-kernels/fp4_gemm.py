"""FP4 e2m1 GEMM via ttnn.bfloat4_b storage + algebraic remap.

The trick
=========

DeepSeek's routed-expert weights are FP4 e2m1: 4-bit floats with a sign,
2-bit exponent, 1-bit mantissa, and a per-block (32-element) e8m0 scale
(power-of-two). The 8 magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}.

We want to keep these on device at FP4 density (4 bits / element) AND avoid
the rounding loss of an extra bfloat8_b cast. The clean solution turns out
to be an exact bijection with TT's native bfp4 (`ttnn.bfloat4_b`):

  fp4 e2m1 4-bit nibble       bfp4 4-bit nibble (block_exp s.t. ULP = 0.25)
  ───────────────────────     ──────────────────────────────────────────────
  sign | exp(2) | mant(1)     sign | mant(3)
  bit pattern abcd            same bits abcd
  → magnitude index = bcd       → magnitude index bcd → 0.25·bcd lattice

Every bit pattern in {0..15} maps to:
  fp4 magnitude  ∈ {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
  bfp4 unpack    ∈ {0, ±0.25, ±0.5, ±0.75, ±1, ±1.25, ±1.5, ±1.75}

So if we encode each fp4 nibble as the bf16 lattice value `(bcd) * 0.25 * sign`
and round-trip through `ttnn.bfloat4_b`, the bytes ttnn stores ARE the original
fp4 nibbles bit-for-bit, and the bf16 we read back is bit-exact.

The only thing left is to remap `{0, ±0.25, ..., ±1.75}` → `{0, ±0.5, ..., ±6}`
on read. Because the function is odd, an algebraic identity does it without
any comparisons or bit ops:

  f(b) = 2·b
       + 2·(relu(b − 1)   − relu(−b − 1))
       + 4·(relu(b − 1.5) − relu(−b − 1.5))

After remap, multiply by the per-K-block fp4 scale, then matmul.

Density / loss
==============

Storage on DRAM   : 4 bits / element (same as native FP4).
Transient at GEMM : bf16 expansion of one weight matrix (typecast → remap
                    → scale) lives in L1/DRAM only for the duration of one
                    matmul call, then is freed.
Numeric loss     : ZERO. bfp4 round-trip is bit-exact for our lattice values
                    because ttnn picks block_exp such that every multiple of
                    0.25 in the face is representable; the remap is exact;
                    the scale multiply is a pure exponent bump (e8m0 = 2^k);
                    bf16 matmul accumulation in fp32 DST.

TODO: tt-lang port
==================

This is implemented in ttnn (not tt-lang) because tt-lang currently requires
all input/output tensors of a single kernel to share one dtype, blocking the
bfp4-weight + bf16-activation pattern. A tt-lang port would fuse the remap +
scale + matmul prologue into one kernel:

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def fp4_remap_gemm(a, w, scale, out):  # all bf16 if mixed dtype landed
        # read bfp4 tile -> hardware unpack to bf16 in CB
        # remap: y = 2b + 2(relu(b-1) - relu(-b-1)) + 4(relu(b-1.5) - relu(-b-1.5))
        # multiply by per-K-block scale (broadcast across K within the block)
        # matmul accumulate

    Fusing remap+matmul saves ~1 DRAM round-trip per matmul (the bf16 transient
    above) and ~3 elementwise kernel launches.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import ttnn

TILE = 32
FP4_BLOCK_K = 32

# 8-magnitude fp4 e2m1 lattice (no sign).
FP4_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Corresponding bfp4 lattice values (k * 0.25 with k ∈ {0..7}).
BFP4_MAGS = tuple(k * 0.25 for k in range(8))


# -----------------------------------------------------------------------------
# Host-side helpers
# -----------------------------------------------------------------------------


def fp4_bytes_to_bfp4_lattice_bf16(w_fp4_packed: torch.Tensor) -> torch.Tensor:
    """Convert fp4 e2m1 packed bytes to a bf16 tensor whose values lie on the
    bfp4 lattice {0, ±0.25, ..., ±1.75}, in such a way that round-tripping
    through `ttnn.from_torch(..., dtype=ttnn.bfloat4_b)` followed by
    `ttnn.to_torch` (or unpacking on the device) is bit-exact.

    Input layout (matches inference.py:_dequant_fp4_weight):
      w_fp4_packed: [N, K/2] uint8/int8/float4_e2m1fn_x2.
                    Each byte holds 2 fp4 nibbles along K (low nibble = even).

    Output: bf16 tensor [N, K] with values in the bfp4 lattice.
    """
    if w_fp4_packed.dtype == torch.float4_e2m1fn_x2:
        b = w_fp4_packed.view(torch.uint8)
    elif w_fp4_packed.dtype in (torch.int8, torch.uint8):
        b = w_fp4_packed.view(torch.uint8)
    else:
        raise TypeError(f"unexpected fp4 dtype {w_fp4_packed.dtype}")

    N, Kh = b.shape
    K = Kh * 2

    low = (b & 0x0F).to(torch.long)       # bit3 = sign, bits2:0 = magnitude index
    high = ((b >> 4) & 0x0F).to(torch.long)

    # Build the 16-entry signed bfp4 lattice: indices 0..7 are positive,
    # 8..15 are negative (sign bit = bit 3 of nibble).
    signed_lat = torch.tensor(list(BFP4_MAGS) + [-v for v in BFP4_MAGS],
                              dtype=torch.float32)

    out = torch.empty(N, K, dtype=torch.bfloat16)
    out[:, 0::2] = signed_lat[low].to(torch.bfloat16)
    out[:, 1::2] = signed_lat[high].to(torch.bfloat16)
    return out


def fp4_bytes_to_fp4_dequant_bf16(
    w_fp4_packed: torch.Tensor, w_sf_e8m0: torch.Tensor, block_k: int = FP4_BLOCK_K
) -> torch.Tensor:
    """Reference dequant: fp4 e2m1 + e8m0 scale → bf16 [N, K]. Lossless."""
    if w_fp4_packed.dtype == torch.float4_e2m1fn_x2:
        b = w_fp4_packed.view(torch.uint8)
    elif w_fp4_packed.dtype in (torch.int8, torch.uint8):
        b = w_fp4_packed.view(torch.uint8)
    else:
        raise TypeError(f"unexpected fp4 dtype {w_fp4_packed.dtype}")
    N, Kh = b.shape
    K = Kh * 2

    low = (b & 0x0F).to(torch.long)
    high = ((b >> 4) & 0x0F).to(torch.long)
    fp4_lat = torch.tensor(list(FP4_MAGS) + [-v for v in FP4_MAGS], dtype=torch.float32)

    vals = torch.empty(N, K, dtype=torch.float32)
    vals[:, 0::2] = fp4_lat[low]
    vals[:, 1::2] = fp4_lat[high]

    if w_sf_e8m0.dtype == torch.float8_e8m0fnu:
        sf_u = w_sf_e8m0.view(torch.uint8)
    elif w_sf_e8m0.dtype == torch.uint8:
        sf_u = w_sf_e8m0
    else:
        raise TypeError(f"unexpected scale dtype {w_sf_e8m0.dtype}")
    if sf_u.shape != (N, K // block_k):
        raise ValueError(
            f"scale shape {tuple(sf_u.shape)} != ({N}, {K // block_k})"
        )
    sf_f32 = (sf_u.to(torch.int32) << 23).contiguous().view(torch.float32)
    sf_exp = sf_f32.repeat_interleave(block_k, dim=1)
    return (vals * sf_exp).to(torch.bfloat16)


def expand_e8m0_scale_to_bf16_kn(
    w_sf_e8m0: torch.Tensor, K: int, N: int, block_k: int = FP4_BLOCK_K
) -> torch.Tensor:
    """e8m0 [N, K/block_k] → bf16 [Kb*TILE, N] for tile-aligned device upload.

    Each fp4 K-block (32 elements) gets one scalar scale per output channel n.
    To match a tt-lang/ttnn matmul's weight tile layout, we replicate each
    [1, N] scale row 32 times along K, producing a [(K/block_k)*TILE, N] tensor.

    Returns bf16 tensor [(K/block_k)*32, N]. With block_k=32 and TILE=32, that
    is exactly [K, N] (one scale row per K-tile, replicated across the tile).
    """
    if block_k != FP4_BLOCK_K:
        raise NotImplementedError(f"only block_k=32 supported (got {block_k})")
    if K % block_k:
        raise ValueError(f"K {K} must be a multiple of block_k {block_k}")
    if w_sf_e8m0.dtype == torch.float8_e8m0fnu:
        sf_u = w_sf_e8m0.view(torch.uint8)
    elif w_sf_e8m0.dtype == torch.uint8:
        sf_u = w_sf_e8m0
    else:
        raise TypeError(f"unexpected scale dtype {w_sf_e8m0.dtype}")
    Kb = K // block_k
    if sf_u.shape != (N, Kb):
        raise ValueError(f"scale shape {tuple(sf_u.shape)} != ({N}, {Kb})")
    sf_f32 = (sf_u.to(torch.int32) << 23).contiguous().view(torch.float32)
    # [N, Kb] -> [Kb, N] (matmul-order transpose)
    sf_kn = sf_f32.transpose(0, 1).contiguous()
    # Replicate each scale row 32× along K-dim so a 32-row tile sees one scale
    # value per output column.
    sf_expanded = sf_kn.repeat_interleave(TILE, dim=0)
    return sf_expanded.to(torch.bfloat16)


# -----------------------------------------------------------------------------
# Device-side helpers (ttnn ops)
# -----------------------------------------------------------------------------


def _remap_bfp4_lattice_to_fp4_mags(b_tt: ttnn.Tensor) -> ttnn.Tensor:
    """f(b) = 2b + 2(relu(b-1) - relu(-b-1)) + 4(relu(b-1.5) - relu(-b-1.5))

    Verified by hand for all 16 lattice values. Recovers fp4 e2m1 magnitudes
    {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} from bfp4 lattice {0, ±0.25, ..., ±1.75}.
    """
    # Pieces of the formula. Avoid in-place ops; ttnn returns new tensors.
    two_b = ttnn.multiply(b_tt, 2.0)

    pos1 = ttnn.relu(ttnn.subtract(b_tt, 1.0))                       # relu(b - 1)
    neg1 = ttnn.relu(ttnn.subtract(ttnn.neg(b_tt), 1.0))              # relu(-b - 1)
    step1 = ttnn.multiply(ttnn.subtract(pos1, neg1), 2.0)             # 2 * (pos1 - neg1)

    pos15 = ttnn.relu(ttnn.subtract(b_tt, 1.5))
    neg15 = ttnn.relu(ttnn.subtract(ttnn.neg(b_tt), 1.5))
    step15 = ttnn.multiply(ttnn.subtract(pos15, neg15), 4.0)

    return ttnn.add(ttnn.add(two_b, step1), step15)


def fp4_gemm_via_bfp4(
    x_tt: ttnn.Tensor,           # bf16 [M, K] on device
    w_bfp4_tt: ttnn.Tensor,      # bfp4 [K, N] on device (matmul-order)
    scale_tt: ttnn.Tensor,       # bf16 [K, N] on device (per-K-block scale, replicated within tile)
) -> ttnn.Tensor:
    """Run y = x @ dequant(w_fp4, scale) on device, with weights stored at
    fp4 density (bfp4 storage) and zero numeric loss.

    Steps on device:
      1) typecast bfp4 → bf16 (transient L1/DRAM bf16 expansion of weight)
      2) algebraic remap: lattice values → fp4 e2m1 magnitudes
      3) per-block scale multiply (broadcast across K within block)
      4) bf16 @ bf16 matmul

    Returns bf16 [M, N].
    """
    # 1) bfp4 -> bf16 (lossless because every lattice value is bf16-exact)
    w_bf16 = ttnn.typecast(w_bfp4_tt, ttnn.bfloat16)

    # 2) remap bfp4 lattice {0, ±0.25, ..., ±1.75} -> fp4 mags {0, ±0.5, ..., ±6}
    w_remap = _remap_bfp4_lattice_to_fp4_mags(w_bf16)

    # 3) apply per-K-block scale. scale_tt is [K, N] with each K-tile of 32 rows
    #    holding the same per-output-column scale.
    w_scaled = ttnn.multiply(w_remap, scale_tt)

    # 4) matmul
    y = ttnn.matmul(x_tt, w_scaled)

    return y


# -----------------------------------------------------------------------------
# Test harness
# -----------------------------------------------------------------------------


def _make_random_fp4_weight(N: int, K: int, block_k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    w_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g)
    # Narrow scale range so values stay near unit magnitude; e8m0 byte centred
    # at 127 (= 2^0). Width 6 → scales spanning 2^-3 .. 2^+2.
    sf = (torch.randint(124, 130, (N, K // block_k), dtype=torch.int32, generator=g)
          .to(torch.uint8))
    return w_packed, sf


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    am, bm = a - a.mean(), b - b.mean()
    num = (am * bm).sum()
    den = (am.norm() * bm.norm()).clamp_min(1e-12)
    return (num / den).item()


def _test_shape(device, K: int, N: int, M: int = 32, threshold: float = 0.999):
    print(f"\n[shape] M={M} K={K} N={N}")

    torch.manual_seed(0)
    # Activation: M valid rows, padded to TILE if M < TILE.
    M_pad = max(M, TILE)
    x_pad = torch.zeros(M_pad, K, dtype=torch.bfloat16)
    x_pad[:M] = torch.randn(M, K, dtype=torch.bfloat16) * 0.5

    # Random fp4 weight + e8m0 scale (matmul-order: we need [K, N], inference.py
    # stores [N, K]; we'll generate [N, K/2] then transpose for matmul order).
    w_fp4_nk_packed, w_sf_nk = _make_random_fp4_weight(N=N, K=K, block_k=FP4_BLOCK_K, seed=1)

    # --- Reference: lossless fp4 dequant (bf16) → matmul ---
    w_bf16_nk = fp4_bytes_to_fp4_dequant_bf16(w_fp4_nk_packed, w_sf_nk)  # [N, K]
    w_bf16_kn = w_bf16_nk.transpose(0, 1).contiguous()                   # [K, N]
    y_ref = (x_pad[:M].float() @ w_bf16_kn.float()).to(torch.bfloat16)   # [M, N]

    # --- Device: bfp4 storage + remap + scale ---
    # Build the bf16 lattice tensor that, when stored as bfloat4_b, encodes
    # our fp4 nibbles bit-for-bit.
    w_lat_nk = fp4_bytes_to_bfp4_lattice_bf16(w_fp4_nk_packed)   # [N, K] bf16, lattice values
    w_lat_kn = w_lat_nk.transpose(0, 1).contiguous()              # [K, N]

    # Per-K-block scale, expanded to [K, N] for tile-aligned multiply.
    scale_kn = expand_e8m0_scale_to_bf16_kn(w_sf_nk, K=K, N=N)   # [K, N] bf16

    common = dict(
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, **common)
    w_tt = ttnn.from_torch(w_lat_kn, dtype=ttnn.bfloat4_b, **common)
    sf_tt = ttnn.from_torch(scale_kn, dtype=ttnn.bfloat16, **common)

    y_tt = fp4_gemm_via_bfp4(x_tt, w_tt, sf_tt)
    y_back = ttnn.to_torch(y_tt)[:M].to(torch.bfloat16)

    p = _pcc(y_ref, y_back)
    diff = (y_ref.float() - y_back.float()).abs()
    print(
        f"  pcc={p:.6f}  max_abs_diff={diff.max().item():.4e}  "
        f"mean_abs_diff={diff.mean().item():.4e}"
    )
    if p < threshold:
        raise AssertionError(f"pcc {p:.6f} < threshold {threshold}")


def _test_remap_only(device):
    """Sanity: remap a bf16 tile of every lattice value, check fp4 magnitudes."""
    print("\n[remap-only] every bfp4 lattice value -> expected fp4 magnitude")
    signed_bfp4 = torch.tensor(
        list(BFP4_MAGS) + [-v for v in BFP4_MAGS], dtype=torch.bfloat16
    )
    expected_fp4 = torch.tensor(
        list(FP4_MAGS) + [-v for v in FP4_MAGS], dtype=torch.bfloat16
    )
    # Pad to a 32x32 tile.
    pad = torch.zeros(32, 32, dtype=torch.bfloat16)
    pad[0, :16] = signed_bfp4
    tt_in = ttnn.from_torch(
        pad.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = _remap_bfp4_lattice_to_fp4_mags(tt_in)
    out = ttnn.to_torch(tt_out).squeeze(0).squeeze(0)[0, :16]
    diff = (out.float() - expected_fp4.float()).abs().max().item()
    print(f"  in:       {signed_bfp4.tolist()}")
    print(f"  expected: {expected_fp4.tolist()}")
    print(f"  got:      {out.tolist()}")
    print(f"  max abs diff: {diff:.4e}")
    if diff > 1e-3:
        raise AssertionError(f"remap-only mismatch: max diff {diff:.4e}")


def main():
    device = ttnn.open_device(device_id=0)
    try:
        _test_remap_only(device)

        # V4-Flash routed-expert shapes (matmul-order).
        # w1/w3: [out=inter_dim=2048, in=dim=4096] -> matmul (K=4096, N=2048)
        # w2:    [out=dim=4096,       in=inter_dim=2048] -> matmul (K=2048, N=4096)
        SHAPES = [
            (4096, 2048),    # w1 / w3
            (2048, 4096),    # w2
            (256, 128),      # smaller sanity
        ]
        for K, N in SHAPES:
            _test_shape(device, K=K, N=N, M=32, threshold=0.999)

        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
