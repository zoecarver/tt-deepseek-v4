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
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark


T_PAD = 128
BUCKET = 128       # smallest bucket from _INDEXER_TOPK_BUCKETS
K_FIXED = 64       # arbitrary k <= bucket (real model uses min(index_topk, bucket))
WIN = 128
MASK_AMP = 1e4
TILE = 32
B, S = 1, 1


def _make_mask_build_kernel(num_h_tiles: int):
    """Fused mask build via bf16 sign trick.

    Replaces the ttnn lt(int32) + typecast(bool->bf16) + sub(1) + mul(amp) + add
    chain with one tt-lang kernel. Inputs ramp/t_active are bf16. Output:
        masked = score + (sign(t_active - ramp - 0.5) - 1) * (MASK_AMP / 2)
    Validates as +0 for ramp < t_active (valid lane) and -MASK_AMP for invalid.
    """
    half_amp = MASK_AMP / 2.0

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def mask_build(score, ramp, t_active, masked_out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-num_h_tiles // total_cores)

        score_dfb = ttl.make_dataflow_buffer_like(
            score, shape=(1, 1), block_count=2)
        ramp_dfb = ttl.make_dataflow_buffer_like(
            ramp, shape=(1, 1), block_count=2)
        ta_dfb = ttl.make_dataflow_buffer_like(
            t_active, shape=(1, 1), block_count=2)
        # Scratch holds the additive mask term; combining it with `score` in
        # the same store mis-compiles when mixed with the multi-fill chain
        # (collapses the scaled term to ~0). Stage via a CB instead.
        mask_dfb = ttl.make_dataflow_buffer_like(
            masked_out, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(
            masked_out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < num_h_tiles:
                    sc = score_dfb.wait()
                    rp = ramp_dfb.wait()
                    ta = ta_dfb.wait()
                    mask_dfb.reserve().store(
                        (ttl.math.sign(ta - rp - ttl.math.fill(rp, 0.5))
                         - ttl.math.fill(ta, 1.0))
                        * ttl.math.fill(rp, half_amp))
                    m = mask_dfb.wait()
                    out_dfb.reserve().store(sc + m)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < num_h_tiles:
                    ttl.copy(score[0, global_w], score_dfb.reserve()).wait()
                    ttl.copy(ramp[0, global_w], ramp_dfb.reserve()).wait()
                    ttl.copy(t_active[0, global_w], ta_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < num_h_tiles:
                    ttl.copy(out_dfb.wait(), masked_out[0, global_w]).wait()

    return mask_build


def _indexer_topk_body_ref(score_in_tt, ramp_int_tt, t_active_tt):
    """Reference op chain, kept verbatim so the kernel path can diverge."""
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

    Mask construction (lt + bool typecast + sub + mul + add) is now a single
    tt-lang kernel using the bf16 sign trick. topk and the post-correction
    int-math chain remain ttnn (TODO: mega bundled with B1 topk lowering).
    """
    h_tiles = BUCKET // TILE
    mask_build = _make_mask_build_kernel(num_h_tiles=h_tiles)

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    # Pre-stage ramp as bf16 [TILE, BUCKET]; only row 0 is used downstream
    # but the kernel processes all rows identically (rows 1+ are don't-cares).
    ramp_bf16_cpu = torch.zeros(TILE, BUCKET, dtype=torch.bfloat16)
    ramp_bf16_cpu[0, :] = torch.arange(BUCKET, dtype=torch.bfloat16)
    ramp_bf16_tt = ttnn.from_torch(ramp_bf16_cpu, dtype=ttnn.bfloat16, **rep)

    state: dict = {"ramp_bf16": ramp_bf16_tt}

    def lk_d_topk_kernel(score_in_tt, ramp_int_tt, t_active_tt, cmp_idxs_out):
        if "init" not in state:
            state["masked_padded"] = ttnn.from_torch(
                torch.zeros(TILE, BUCKET, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["init"] = True

        # Stage score [1, 1, T_PAD] -> [TILE, BUCKET].
        # TODO: mega fusion blocked: ttnn.slice/reshape sub-tile.
        score_slice = ttnn.slice(score_in_tt, [0, 0, 0], [B, S, BUCKET])
        score_2d = ttnn.reshape(score_slice, [B, BUCKET])
        score_padded = ttnn.pad(score_2d, padding=[(0, TILE - B), (0, 0)],
                                value=0.0)

        # Convert t_active int32 -> bf16; pad to [TILE, BUCKET].
        # TODO: mega fusion blocked: ttnn.typecast int32->bf16.
        t_active_bf16 = ttnn.typecast(t_active_tt, dtype=ttnn.bfloat16)
        t_active_2d = ttnn.reshape(t_active_bf16, [B, BUCKET])
        t_active_padded = ttnn.pad(t_active_2d, padding=[(0, TILE - B), (0, 0)],
                                   value=0.0)

        # tt-lang fused mask build + score add.
        mask_build(score_padded, state["ramp_bf16"],
                   t_active_padded, state["masked_padded"])

        # Slice back to [1, 1, BUCKET] for topk consumption.
        # TODO: mega fusion blocked: ttnn.slice/reshape sub-tile.
        masked_2d = ttnn.slice(state["masked_padded"], [0, 0], [B, BUCKET])
        masked_tt = ttnn.reshape(masked_2d, [1, B, BUCKET])

        # topk + post-correction stay in ttnn until B1 lands.
        # TODO: mega fusion blocked: ttnn.topk + the int post-correction chain.
        vals_tt, idxs_tt = ttnn.topk(
            masked_tt, k=K_FIXED, dim=-1, largest=True, sorted=True)
        invalid_bf16 = ttnn.lt(vals_tt, -1000.0)
        invalid_int = ttnn.typecast(invalid_bf16, dtype=ttnn.int32)
        idxs_int = ttnn.typecast(idxs_tt, dtype=ttnn.int32)
        idxs_winned = ttnn.add(idxs_int, WIN)
        idxs_plus_1 = ttnn.add(idxs_winned, 1)
        correction = ttnn.multiply(idxs_plus_1, invalid_int)
        cmp_idxs_int = ttnn.subtract(idxs_winned, correction)
        ttnn.copy(cmp_idxs_int, cmp_idxs_out)

    return lk_d_topk_kernel


def reference(mesh, score_in_tt, ramp_int_tt, t_active_tt):
    return _indexer_topk_body_ref(score_in_tt, ramp_int_tt, t_active_tt)


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

        benchmark("Lk-D-topk ref",
                  lambda: reference(mesh, score_in_tt, ramp_int_tt, t_active_tt),
                  mesh)
        benchmark("Lk-D-topk ttl",
                  lambda: kernel(score_in_tt, ramp_int_tt, t_active_tt,
                                 cmp_idxs_out_tt),
                  mesh)

        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
