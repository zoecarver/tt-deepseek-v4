"""Lk-D-topk PCC test: bucketed pad+mask topk on the indexer score.

Reference covers `DeviceAttention._indexer_topk_body`:
- ttnn.copy(score, _indexer_score_in_tt)  (staging step)
- slice(score_in, [0,0,0], [B,S,bucket])
- lt(ramp_int, t_active) → mask_bool
- typecast bool→bf16, subtract 1, multiply 1e4 → additive mask
- add(score_slice, mask_add) → masked
- topk(masked, k=k_fixed)
- lt(vals, -1000) → invalid; correction:
    idxs_winned = idxs + win
    cmp_idxs = idxs_winned - (idxs_winned + 1) * invalid_int

Boundaries: input is the score tensor produced by Lk-D-idx-score (no CCL
between them); output feeds Lk-Dsparse directly (no CCL after).
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0


T_PAD = 128
BUCKET = 128       # smallest bucket from _INDEXER_TOPK_BUCKETS
K_FIXED = 64       # arbitrary k <= bucket (real model uses min(index_topk, bucket))
WIN = 128
B, S = 1, 1


def make_lk_d_topk_kernel():
    """Placeholder mega kernel for Lk-D-topk.

    Inputs:
      score_in:  [1, 1, T_pad] bf16  — staged via ttnn.copy from Lk-D-idx-score
      ramp_int:  [1, 1, bucket] int32  — persistent [0..bucket-1] ramp
      t_active:  [1, 1, bucket] int32  — staged outside the trace
      bucket, k_fixed, win: compile-time constants
    Output:
      cmp_idxs:  [1, 1, k_fixed] int32  (with -1 sentinels for masked positions)
    """
    @ttl.operation(grid="auto")
    def lk_d_topk_kernel(score_in, ramp_int, t_active, cmp_idxs):
        @ttl.compute()
        def compute():
            pass

    return lk_d_topk_kernel


def reference(mesh, score_in_tt, ramp_int_tt, t_active_tt):
    score_slice = ttnn.slice(score_in_tt, [0, 0, 0], [B, S, BUCKET])
    mask_bool = ttnn.lt(ramp_int_tt, t_active_tt)
    mask_bf16 = ttnn.typecast(mask_bool, dtype=ttnn.bfloat16)
    mask_minus_1 = ttnn.subtract(mask_bf16, 1.0)
    mask_add = ttnn.multiply(mask_minus_1, 1e4)
    masked_tt = ttnn.add(score_slice, mask_add)
    vals_tt, idxs_tt = ttnn.topk(masked_tt, k=K_FIXED, dim=-1, largest=True, sorted=True)
    invalid_bf16 = ttnn.lt(vals_tt, -1000.0)
    invalid_int = ttnn.typecast(invalid_bf16, dtype=ttnn.int32)
    idxs_int = ttnn.typecast(idxs_tt, dtype=ttnn.int32)
    idxs_winned = ttnn.add(idxs_int, WIN)
    idxs_plus_1 = ttnn.add(idxs_winned, 1)
    correction = ttnn.multiply(idxs_plus_1, invalid_int)
    cmp_idxs_int = ttnn.subtract(idxs_winned, correction)
    return cmp_idxs_int


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

        kernel = make_lk_d_topk_kernel()
        cmp_idxs_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, K_FIXED, dtype=torch.int32),
            dtype=ttnn.int32, **rep)
        kernel(score_in_tt, ramp_int_tt, t_active_tt, cmp_idxs_out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, cmp_idxs_out_tt)

        ok = report_pcc("Lk-D-topk", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
