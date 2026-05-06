"""Stress test: lk_d_idx_cmp → _hybrid_dsparse boundary repro.

stress_dsparse.py showed that _hybrid_dsparse alone runs cleanly for 1000
iters with no sync. stress_cmp_score.py showed that lk_d_idx_cmp's SUMMA
matmul vs ttnn.matmul flips the hang on the cmp→score boundary.

This harness combines the two: each iter calls lk_d_idx_cmp (paged_update_cache
to idx_kv_cache pinned to (0,0)) followed immediately by the _hybrid_dsparse
body (paged_update_cache to kv_cache pinned to (0,0) + DeviceSparseAttn).
Both kernels write to L1 sharded (0,0) buffers. If they hang together
without sync, we have a minimal cmp→dsparse repro decoupled from the
indexer score/topk/comp pipeline.

Pre-stage and run on galaxy:
  copy-file.sh scripts/stress_cmp_dsparse.py /tmp/stress_cmp_dsparse.py
  copy-file.sh tt-lang-kernels/mega/test_lk_d_idx_cmp.py /tmp/test_lk_d_idx_cmp.py
  copy-file.sh tt-lang-kernels/mega/_refs.py /tmp/_refs.py
  copy-file.sh tt-lang-kernels/mega/run_mega.py /tmp/run_mega.py  # _refs imports it
  copy-file.sh inference.py /tmp/inference.py
  run-test.sh --hw scripts/stress_cmp_dsparse.py
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
    if os.path.exists(os.path.join(_cand, "test_lk_d_idx_cmp.py")):
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

from test_lk_d_idx_cmp import (
    make_lk_d_idx_cmp_kernel,
    DIM, CDIM, INDEX_HEAD_DIM, RATIO_PAD, MAX_SEQ_LEN, B as CMP_B,
)
from inference import DeviceSparseAttn, _device_apply_rotary_interleaved


N_ITERS = 1000
LOG_EVERY = 25
SYNC_BETWEEN = False

# DSparse config (DeepSeek-V4-Flash defaults).
DS_DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
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

        # ===== cmp inputs (mirrors stress_cmp_score.py) ===================
        x = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        wkv = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        wgate = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        wkv_gate = torch.cat([wkv, wgate], dim=-1).contiguous()
        ape_padded = torch.randn(MAX_SEQ_LEN, CDIM,
                                 dtype=torch.bfloat16) * 0.05
        kv_state_front = torch.zeros(
            CMP_B, 1, RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        kv_state_back = torch.zeros(
            CMP_B, 1, RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        score_state_front = torch.full(
            (CMP_B, 1, RATIO_PAD, INDEX_HEAD_DIM), float("-inf"),
            dtype=torch.bfloat16)
        score_state_back = torch.full(
            (CMP_B, 1, RATIO_PAD, INDEX_HEAD_DIM), float("-inf"),
            dtype=torch.bfloat16)
        cmp_start_pos = torch.tensor([[3]], dtype=torch.int32)
        cmp_state_slot = torch.tensor([7], dtype=torch.int32)

        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep_dram)
        wkv_gate_tt = ttnn.as_tensor(
            wkv_gate, dtype=ttnn.bfloat16, **rep_dram)
        ape_padded_tt = ttnn.as_tensor(
            ape_padded.contiguous(), dtype=ttnn.bfloat16, **rep_dram)
        kv_state_front_tt = ttnn.as_tensor(
            kv_state_front.contiguous(), dtype=ttnn.bfloat16, **rep_dram)
        kv_state_back_tt = ttnn.as_tensor(
            kv_state_back.contiguous(), dtype=ttnn.bfloat16, **rep_dram)
        score_state_front_tt = ttnn.as_tensor(
            score_state_front.contiguous(), dtype=ttnn.bfloat16, **rep_dram)
        score_state_back_tt = ttnn.as_tensor(
            score_state_back.contiguous(), dtype=ttnn.bfloat16, **rep_dram)
        cmp_start_pos_tt = ttnn.from_torch(
            cmp_start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        cmp_state_slot_tt = ttnn.from_torch(
            cmp_state_slot, device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        cmp_sharded_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                    ttnn.CoreCoord(0, 0))}),
                (32, INDEX_HEAD_DIM),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        idx_cmp_kv_tt = ttnn.from_torch(
            torch.zeros(1, 1, 2 * INDEX_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep_dram)
        idx_cmp_score_tt = ttnn.from_torch(
            torch.zeros(1, 1, 2 * INDEX_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep_dram)

        cmp_kernel = make_lk_d_idx_cmp_kernel(
            mesh, sharded_input_memcfg=cmp_sharded_memcfg)

        # ===== dsparse inputs (mirrors _hybrid_dsparse + run_mega) =======
        kv_rot = torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_rot_tt = ttnn.from_torch(kv_rot.contiguous(),
                                    dtype=ttnn.bfloat16, **rep_l1)
        kv_cache_tt = ttnn.from_torch(
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

        def call_cmp():
            cmp_kernel(
                x_tt, wkv_gate_tt, ape_padded_tt,
                cmp_start_pos_tt, cmp_state_slot_tt,
                kv_state_front_tt, kv_state_back_tt,
                score_state_front_tt, score_state_back_tt,
                idx_cmp_kv_tt, idx_cmp_score_tt,
            )

        def call_dsparse():
            kv_4d = ttnn.reshape(kv_rot_tt, [1, B, 1, D])
            kv_4d_sharded = ttnn.to_memory_config(
                kv_4d, memory_config=ds_sharded_memcfg)
            ttnn.experimental.paged_update_cache(
                kv_cache_tt, kv_4d_sharded,
                update_idxs_tensor=kv_slot_tt)

            idxs_tt, valid_tt = dsa._idxs_int_tile_to_idxs_and_mask(
                topk_idxs_tt, B, S, K)
            kv_full_tt = ttnn.reshape(kv_cache_tt, [KV_CACHE_SIZE_PAD, D])
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
        call_cmp()
        call_dsparse()
        ttnn.synchronize_device(mesh)
        print(f"[stress] warmup done; running {N_ITERS} iters "
              f"sync_between={SYNC_BETWEEN}", flush=True)

        t0 = time.time()
        last_log = t0
        for it in range(N_ITERS):
            call_cmp()
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
