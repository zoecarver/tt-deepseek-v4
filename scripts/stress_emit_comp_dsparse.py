"""Stress test: lk_d_idx_emit -> lk_d_comp -> _hybrid_dsparse repro.

stress_comp_dsparse.py (comp + dsparse only) PASSED 1000 iters.
The real run hangs at lk_dsparse on emit step. Both lk_d_idx_emit and
lk_d_comp are emit-only kernels. This harness adds lk_d_idx_emit upstream
of comp to capture the full emit-only chain before dsparse.

If this hangs in 1000 iters with no sync, we have a 3-kernel reproducer.

Pre-stage and run on galaxy:
  copy-file.sh scripts/stress_emit_comp_dsparse.py /tmp/stress_emit_comp_dsparse.py
  copy-file.sh tt-lang-kernels/mega/test_lk_d_idx_emit.py /tmp/test_lk_d_idx_emit.py
  copy-file.sh tt-lang-kernels/mega/test_lk_d_comp.py /tmp/test_lk_d_comp.py
  copy-file.sh tt-lang-kernels/mega/_refs.py /tmp/_refs.py
  copy-file.sh tt-lang-kernels/mega/run_mega.py /tmp/run_mega.py
  copy-file.sh inference.py /tmp/inference.py
  run-test.sh --hw scripts/stress_emit_comp_dsparse.py
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    "/tmp",
    os.path.join(_HERE, "..", "tt-lang-kernels", "mega"),
):
    if os.path.exists(os.path.join(_cand, "test_lk_d_comp.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
for _cand in (
    "/tmp",
    os.path.join(_HERE, ".."),
):
    if os.path.exists(os.path.join(_cand, "inference.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

import torch
import ttnn

from test_lk_d_comp import (
    make_lk_d_comp_kernel,
    DIM, CDIM, HEAD_DIM, ROPE_HEAD_DIM, MAX_SEQ_LEN,
    RATIO, RATIO_PAD, T_PAD, B as COMP_B, _RMS_TILE,
)
from test_lk_d_idx_emit import (
    make_lk_d_idx_emit_kernel,
    INDEX_HEAD_DIM,
)
from inference import (
    DeviceSparseAttn,
    _device_apply_rotary_interleaved,
    _compressor_softmax_sum_norm_masks,
    _compressor_shift_matrix,
    _sylvester_hadamard,
)


N_ITERS = 1000
LOG_EVERY = 25
SYNC_BETWEEN = False

DS_DIM = 4096
N_HEADS = 64
WINDOW_SIZE = 128
O_GROUPS = 8
O_LORA_RANK = 1024
KV_CACHE_SIZE_PAD = 128
PER_GROUP = (N_HEADS * HEAD_DIM) // O_GROUPS
B = 1
S = 1
TILE = 32


def main():
    torch.manual_seed(0)

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
    )
    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(4, 8),
        trace_region_size=200_000_000,
        worker_l1_size=default_l1 - 128 * 1024,
    )
    try:
        rep_dram = dict(
            device=mesh, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        rep_l1 = dict(
            device=mesh, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        def up(t, dt=ttnn.bfloat16):
            return ttnn.as_tensor(t.contiguous(), dtype=dt, **rep_dram)

        # ============== shared rotary tables (compressor lane) ===========
        cos_compressor_cpu = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2,
                                         dtype=torch.bfloat16) * 0.5
        sin_compressor_cpu = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2,
                                         dtype=torch.bfloat16) * 0.5
        mf, mb, mp = _compressor_softmax_sum_norm_masks(RATIO)
        scaler_cpu = torch.ones(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16)
        shift_P_cpu = _compressor_shift_matrix(RATIO, RATIO_PAD)
        start_pos_cpu = torch.tensor([[RATIO - 1]], dtype=torch.int32)
        emit_slot_cpu = torch.tensor([0], dtype=torch.int32)

        cos_compressor_tt = up(cos_compressor_cpu)
        sin_compressor_tt = up(sin_compressor_cpu)
        mf_tt = up(mf); mb_tt = up(mb); mp_tt = up(mp)
        scaler_tt = up(scaler_cpu)
        shift_P_tt = up(shift_P_cpu)
        start_pos_tt = ttnn.from_torch(
            start_pos_cpu, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        emit_slot_tt = ttnn.from_torch(
            emit_slot_cpu, device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        # ============== lk_d_idx_emit (indexer side, d=128) ==============
        idx_d = INDEX_HEAD_DIM
        idx_kv_sf_cpu = torch.randn(RATIO_PAD, idx_d,
                                    dtype=torch.bfloat16) * 0.1
        idx_kv_sb_cpu = torch.randn(RATIO_PAD, idx_d,
                                    dtype=torch.bfloat16) * 0.1
        idx_sc_sf_cpu = torch.randn(RATIO_PAD, idx_d,
                                    dtype=torch.bfloat16) * 0.1
        idx_sc_sb_cpu = torch.randn(RATIO_PAD, idx_d,
                                    dtype=torch.bfloat16) * 0.1
        idx_sc_sf_cpu[2 * RATIO:, :] = float("-inf")
        idx_sc_sb_cpu[2 * RATIO:, :] = float("-inf")
        idx_gamma_cpu = torch.ones(_RMS_TILE, idx_d, dtype=torch.bfloat16)
        idx_H_cpu = (_sylvester_hadamard(idx_d) *
                     (idx_d ** -0.5)).to(torch.bfloat16)
        idx_kv_cache_cpu = torch.zeros(1, 1, T_PAD, idx_d,
                                       dtype=torch.bfloat16)

        idx_kv_sf_tt = up(idx_kv_sf_cpu)
        idx_kv_sb_tt = up(idx_kv_sb_cpu)
        idx_sc_sf_tt = up(idx_sc_sf_cpu)
        idx_sc_sb_tt = up(idx_sc_sb_cpu)
        idx_gamma_tt = up(idx_gamma_cpu)
        idx_H_tt = up(idx_H_cpu)
        idx_kv_cache_tt = up(idx_kv_cache_cpu)
        idx_zero_pad = torch.zeros(RATIO_PAD, idx_d, dtype=torch.bfloat16)
        idx_ninf_pad = torch.full_like(idx_zero_pad, float("-inf"))
        idx_kv_sf_scratch_tt = up(idx_zero_pad)
        idx_kv_sb_scratch_tt = up(idx_zero_pad)
        idx_sc_sf_scratch_tt = up(idx_ninf_pad)
        idx_sc_sb_scratch_tt = up(idx_ninf_pad)
        idx_kv_normed_out_tt = up(torch.zeros(1, 1, idx_d,
                                              dtype=torch.bfloat16))

        sharded_memcfg_idx = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                    ttnn.CoreCoord(0, 0))}),
                (32, idx_d),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        emit_kernel = make_lk_d_idx_emit_kernel(
            mesh, cos_compressor_cpu, sin_compressor_cpu, sharded_memcfg_idx)

        # ============== lk_d_comp (attn side, d=512) =====================
        x_cpu = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        wkv_w_cpu = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        wgate_w_cpu = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        ape_padded_cpu = torch.randn(MAX_SEQ_LEN, CDIM,
                                     dtype=torch.bfloat16) * 0.05

        kv_state_front_cpu = torch.randn(
            COMP_B, 1, RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_state_back_cpu = torch.randn(
            COMP_B, 1, RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_front_cpu = torch.randn(
            COMP_B, 1, RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_back_cpu = torch.randn(
            COMP_B, 1, RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_front_cpu[:, :, 2 * RATIO:, :] = float("-inf")
        score_state_back_cpu[:, :, 2 * RATIO:, :] = float("-inf")
        comp_gamma_cpu = torch.ones(_RMS_TILE, HEAD_DIM,
                                    dtype=torch.bfloat16)

        x_tt = up(x_cpu)
        wkv_w_tt = up(wkv_w_cpu)
        wgate_w_tt = up(wgate_w_cpu)
        ape_padded_tt = up(ape_padded_cpu)
        comp_gamma_tt = up(comp_gamma_cpu)
        state_slot_cpu = torch.tensor([RATIO + (RATIO - 1) % RATIO],
                                      dtype=torch.int32)
        state_slot_tt = ttnn.from_torch(
            state_slot_cpu, device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        kv_sf_4d_tt = up(kv_state_front_cpu)
        kv_sb_4d_tt = up(kv_state_back_cpu)
        sc_sf_4d_tt = up(score_state_front_cpu)
        sc_sb_4d_tt = up(score_state_back_cpu)
        kv_sf_2d_tt = ttnn.reshape(kv_sf_4d_tt, [RATIO_PAD, HEAD_DIM])
        kv_sb_2d_tt = ttnn.reshape(kv_sb_4d_tt, [RATIO_PAD, HEAD_DIM])
        sc_sf_2d_tt = ttnn.reshape(sc_sf_4d_tt, [RATIO_PAD, HEAD_DIM])
        sc_sb_2d_tt = ttnn.reshape(sc_sb_4d_tt, [RATIO_PAD, HEAD_DIM])

        zero_pad = torch.zeros(RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16)
        ninf_pad = torch.full_like(zero_pad, float("-inf"))
        kv_sf_scratch_tt = up(zero_pad)
        kv_sb_scratch_tt = up(zero_pad)
        sc_sf_scratch_tt = up(ninf_pad)
        sc_sb_scratch_tt = up(ninf_pad)
        cssn_out_tt = up(torch.zeros(_RMS_TILE, HEAD_DIM,
                                     dtype=torch.bfloat16))
        kv_normed_out_tt = up(torch.zeros(1, 1, HEAD_DIM,
                                          dtype=torch.bfloat16))
        kv_cache_comp_tt = up(torch.zeros(1, 1, T_PAD, HEAD_DIM,
                                          dtype=torch.bfloat16))

        sharded_memcfg_comp = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                    ttnn.CoreCoord(0, 0))}),
                (32, HEAD_DIM),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        comp_kernel = make_lk_d_comp_kernel(
            mesh, cos_compressor_cpu, sin_compressor_cpu,
            sharded_memcfg_comp)

        # ============== _hybrid_dsparse =================================
        kv_rot = torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_rot_tt = ttnn.from_torch(kv_rot.contiguous(),
                                    dtype=ttnn.bfloat16, **rep_l1)
        kv_cache_ds_tt = ttnn.from_torch(
            torch.zeros(1, 1, KV_CACHE_SIZE_PAD, HEAD_DIM,
                        dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep_dram)
        kv_slot_tt = ttnn.from_torch(
            torch.tensor([7], dtype=torch.int32),
            device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        ds_sharded_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                    ttnn.CoreCoord(0, 0))}),
                (TILE, HEAD_DIM),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        q_rot = torch.randn(1, 1, N_HEADS, HEAD_DIM,
                            dtype=torch.bfloat16) * 0.1
        q_rot_tt = ttnn.from_torch(q_rot.contiguous(),
                                   dtype=ttnn.bfloat16, **rep_l1)
        K = WINDOW_SIZE
        topk_t = torch.randint(0, KV_CACHE_SIZE_PAD,
                               (B, S, K), dtype=torch.int32)
        topk_t[..., -3:] = -1
        topk_idxs_tt = ttnn.from_torch(topk_t.contiguous(),
                                       dtype=ttnn.int32, **rep_dram)
        attn_sink_cpu = torch.zeros(N_HEADS, dtype=torch.bfloat16)

        cos_full = torch.randn(2048, ROPE_HEAD_DIM // 2,
                               dtype=torch.bfloat16) * 0.1
        sin_full = torch.randn(2048, ROPE_HEAD_DIM // 2,
                               dtype=torch.bfloat16) * 0.1
        cos_full_tt = ttnn.as_tensor(
            cos_full.contiguous(), device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        sin_full_tt = ttnn.as_tensor(
            sin_full.contiguous(), device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        ds_start_pos_tt = ttnn.from_torch(
            torch.tensor([[7]], dtype=torch.int32),
            device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        wo_a = torch.randn(O_GROUPS, PER_GROUP, O_LORA_RANK,
                           dtype=torch.bfloat16) * 0.02
        wo_b = torch.randn(O_GROUPS * O_LORA_RANK, DS_DIM,
                           dtype=torch.bfloat16) * 0.02
        wo_a_tt = ttnn.as_tensor(wo_a.contiguous(),
                                 dtype=ttnn.bfloat16, **rep_dram)
        wo_b_tt = ttnn.as_tensor(wo_b.contiguous(),
                                 dtype=ttnn.bfloat16, **rep_dram)
        attn_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, DS_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep_l1)

        softmax_scale = float(HEAD_DIM ** -0.5)
        dsa = DeviceSparseAttn(
            mesh=mesh, attn_sink=attn_sink_cpu,
            softmax_scale=softmax_scale)

        H = N_HEADS
        D = HEAD_DIM
        rd = ROPE_HEAD_DIM
        rd_half = rd // 2
        n_groups = O_GROUPS
        per_group = PER_GROUP
        o_lora_rank = O_LORA_RANK

        def call_emit():
            emit_kernel(
                idx_kv_sf_tt, idx_kv_sb_tt, idx_sc_sf_tt, idx_sc_sb_tt,
                mf_tt, mb_tt, mp_tt,
                idx_gamma_tt, scaler_tt,
                cos_compressor_tt, sin_compressor_tt, start_pos_tt,
                idx_H_tt, idx_kv_cache_tt, emit_slot_tt, shift_P_tt,
                idx_kv_sf_scratch_tt, idx_kv_sb_scratch_tt,
                idx_sc_sf_scratch_tt, idx_sc_sb_scratch_tt,
                idx_kv_normed_out_tt,
            )

        def call_comp():
            comp_kernel(
                x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt,
                cos_compressor_tt, sin_compressor_tt, start_pos_tt,
                state_slot_tt, emit_slot_tt,
                kv_sf_2d_tt, kv_sb_2d_tt, sc_sf_2d_tt, sc_sb_2d_tt,
                kv_sf_4d_tt, kv_sb_4d_tt, sc_sf_4d_tt, sc_sb_4d_tt,
                mf_tt, mb_tt, mp_tt, comp_gamma_tt, scaler_tt,
                kv_cache_comp_tt, shift_P_tt,
                kv_sf_scratch_tt, kv_sb_scratch_tt,
                sc_sf_scratch_tt, sc_sb_scratch_tt,
                cssn_out_tt, kv_normed_out_tt,
            )

        def call_dsparse():
            kv_4d = ttnn.reshape(kv_rot_tt, [1, B, 1, D])
            kv_4d_sharded = ttnn.to_memory_config(
                kv_4d, memory_config=ds_sharded_memcfg)
            ttnn.experimental.paged_update_cache(
                kv_cache_ds_tt, kv_4d_sharded,
                update_idxs_tensor=kv_slot_tt)

            idxs_tt, valid_tt = dsa._idxs_int_tile_to_idxs_and_mask(
                topk_idxs_tt, B, S, K)
            kv_full_tt = ttnn.reshape(kv_cache_ds_tt,
                                       [KV_CACHE_SIZE_PAD, D])
            o_tt = dsa.forward_device(q_rot_tt, kv_full_tt, idxs_tt,
                                      valid_tt, S, K)

            cos = ttnn.embedding(ds_start_pos_tt, cos_full_tt,
                                 layout=ttnn.TILE_LAYOUT)
            sin = ttnn.embedding(ds_start_pos_tt, sin_full_tt,
                                 layout=ttnn.TILE_LAYOUT)
            cos = ttnn.reshape(cos, [1, S, 1, rd_half])
            sin = ttnn.reshape(sin, [1, S, 1, rd_half])
            o_nope = ttnn.slice(o_tt, [0, 0, 0, 0], [B, S, H, D - rd])
            o_rope = ttnn.slice(o_tt, [0, 0, 0, D - rd], [B, S, H, D])
            o_rope = _device_apply_rotary_interleaved(
                ttnn, o_rope, cos, sin, inverse=True)
            o_tt = ttnn.concat([o_nope, o_rope], dim=-1)

            o_perm = ttnn.reshape(o_tt, [B, S, n_groups, per_group])
            o_perm = ttnn.permute(o_perm, [2, 0, 1, 3])
            o_g = ttnn.reshape(o_perm, [n_groups, B * S, per_group])
            o_wo_a_g = ttnn.matmul(
                o_g, wo_a_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o_wo_a_rs = ttnn.reshape(
                o_wo_a_g, [n_groups, B, S, o_lora_rank])
            o_wo_a = ttnn.permute(o_wo_a_rs, [1, 2, 0, 3])
            o_flat = ttnn.reshape(o_wo_a, [B, S, n_groups * o_lora_rank])

            out_tt = ttnn.matmul(
                o_flat, wo_b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(out_tt, attn_out_tt)

        print("[stress] warmup ...", flush=True)
        call_emit()
        call_comp()
        call_dsparse()
        ttnn.synchronize_device(mesh)
        print(f"[stress] warmup done; running {N_ITERS} iters "
              f"sync_between={SYNC_BETWEEN}", flush=True)

        t0 = time.time()
        last_log = t0
        for it in range(N_ITERS):
            call_emit()
            if SYNC_BETWEEN:
                ttnn.synchronize_device(mesh)
            call_comp()
            if SYNC_BETWEEN:
                ttnn.synchronize_device(mesh)
            call_dsparse()
            if (it + 1) % LOG_EVERY == 0:
                now = time.time()
                print(f"[stress] iter {it+1}/{N_ITERS} dispatched "
                      f"(+{now - last_log:.1f}s, {now - t0:.1f}s total)",
                      flush=True)
                last_log = now
        ttnn.synchronize_device(mesh)
        elapsed = time.time() - t0
        print(f"[stress] PASS {N_ITERS} iters in {elapsed:.1f}s "
              f"({N_ITERS / elapsed:.2f} iter/s)", flush=True)
        sys.exit(0)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
