"""Lk-D-topk PCC test: bucketed pad+mask topk on the indexer score.

Reference covers `DeviceAttention._indexer_topk_body`:
- ttnn.copy(score, _indexer_score_in_tt)  (staging step)
- slice(score_in, [0,0,0], [B,S,bucket])
- lt(ramp_int, t_active) -> mask_bool
- typecast bool->bf16, subtract 1, multiply 1e4 -> additive mask
- add(score_slice, mask_add) -> masked
- topk(masked, k=k_fixed)
- lt(vals, -1000) -> invalid; correction:
    idxs_winned = idxs + win
    cmp_idxs = idxs_winned - (idxs_winned + 1) * invalid_int

Boundaries: input is the score tensor produced by Lk-D-idx-score (no CCL
between them); output feeds Lk-Dsparse directly (no CCL after).

This zone has no current tt-lang lowering: every op is blocked. The
kernel path mirrors the reference's ttnn chain so the test still gates
the op chain compiles + runs end-to-end. Once the tt-lang primitives
land, the body collapses into a single fused dispatch.

ttnn glue (TODO: mega):
  - lt(ramp_int, t_active) -> int32 compare (no tt-lang int compare)
  - typecast bool->bf16 (no tt-lang bool cast)
  - subtract/multiply/add for mask combine (could lower once the bf16
    inputs exist, but gating the whole zone on the int/bool blockers
    isn't worth the staging overhead today)
  - topk (no tt-lang topk primitive)
  - the idxs correction (typecast bool->int32, int add/multiply/subtract)
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl  # noqa: F401  (kept so the test compiles in the same env as the others)

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0


T_PAD = 128
BUCKET = 128       # smallest bucket from _INDEXER_TOPK_BUCKETS
K_FIXED = 64       # arbitrary k <= bucket (real model uses min(index_topk, bucket))
WIN = 128
MASK_AMP = 1e4
B, S = 1, 1


def _indexer_topk_body(score_in_tt, ramp_int_tt, t_active_tt):
    """The shared op chain. Used by both the reference and kernel paths
    today; once tt-lang gains topk + int compares + bool casts, the
    kernel path will diverge into a fused dispatch."""
    score_slice = ttnn.slice(score_in_tt, [0, 0, 0], [B, S, BUCKET])
    mask_bool = ttnn.lt(ramp_int_tt, t_active_tt)
    mask_bf16 = ttnn.typecast(mask_bool, dtype=ttnn.bfloat16)
    mask_minus_1 = ttnn.subtract(mask_bf16, 1.0)
    mask_add = ttnn.multiply(mask_minus_1, MASK_AMP)
    masked_tt = ttnn.add(score_slice, mask_add)
    vals_tt, idxs_tt = ttnn.topk(
        masked_tt, k=K_FIXED, dim=-1, largest=True, sorted=True)
    invalid_bf16 = ttnn.lt(vals_tt, -1000.0)
    invalid_int = ttnn.typecast(invalid_bf16, dtype=ttnn.int32)
    idxs_int = ttnn.typecast(idxs_tt, dtype=ttnn.int32)
    idxs_winned = ttnn.add(idxs_int, WIN)
    idxs_plus_1 = ttnn.add(idxs_winned, 1)
    correction = ttnn.multiply(idxs_plus_1, invalid_int)
    return ttnn.subtract(idxs_winned, correction)


def make_lk_d_topk_kernel(mesh):
    """Mega kernel for Lk-D-topk.

    Today the kernel path is identical to the reference because every op
    in the zone is blocked. Wrapped in a closure that matches the
    `kernel(in..., out)` calling convention used by sibling tests.
    """

    def lk_d_topk_kernel(score_in_tt, ramp_int_tt, t_active_tt, cmp_idxs_out):
        cmp_idxs_int = _indexer_topk_body(score_in_tt, ramp_int_tt, t_active_tt)
        ttnn.copy(cmp_idxs_int, cmp_idxs_out)

    return lk_d_topk_kernel


def reference(mesh, score_in_tt, ramp_int_tt, t_active_tt):
    return _indexer_topk_body(score_in_tt, ramp_int_tt, t_active_tt)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        # Score values randomized; only the first T_active positions are valid.
        score = torch.randn(1, 1, T_PAD, dtype=torch.bfloat16)
        t_active_val = 50
        ramp = torch.arange(BUCKET, dtype=torch.int32).view(1, 1, BUCKET)
        t_active = torch.full((1, 1, BUCKET), t_active_val, dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        score_in_tt = ttnn.as_tensor(score.contiguous(), dtype=ttnn.bfloat16, **rep)
        ramp_int_tt = ttnn.as_tensor(ramp.contiguous(), dtype=ttnn.int32, **rep)
        t_active_tt = ttnn.as_tensor(t_active.contiguous(), dtype=ttnn.int32, **rep)

        ref_out_tt = reference(mesh, score_in_tt, ramp_int_tt, t_active_tt)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_d_topk_kernel(mesh)
        cmp_idxs_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, K_FIXED, dtype=torch.int32),
            dtype=ttnn.int32, **rep)
        kernel(score_in_tt, ramp_int_tt, t_active_tt, cmp_idxs_out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, cmp_idxs_out_tt)

        # cmp_idxs is int32; PCC degenerates if many entries collide on -1.
        # Compare elementwise (exact match expected when math is preserved).
        if ref_host.shape != kernel_host.shape:
            print(f"[Lk-D-topk] FAIL shape mismatch: ref={tuple(ref_host.shape)} "
                  f"kernel={tuple(kernel_host.shape)}")
            sys.exit(1)
        diff = (ref_host.to(torch.int64) - kernel_host.to(torch.int64)).abs()
        max_diff = int(diff.max().item())
        n_mismatch = int((diff != 0).sum().item())
        ok = max_diff == 0
        status = "PASS" if ok else "FAIL"
        print(f"[Lk-D-topk] {status} max_diff={max_diff} n_mismatch={n_mismatch} "
              f"of {ref_host.numel()}")
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
