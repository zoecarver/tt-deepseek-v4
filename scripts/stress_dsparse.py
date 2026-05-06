"""Stress test: _hybrid_dsparse repro for run_mega's pos=7 hang.

In run_mega.py, `_hybrid_dsparse` is the legacy ttnn-only attn block:
paged_update_cache (sharded L1 (0,0)) + DeviceSparseAttn.forward_device
(embedding + matmul + softmax + matmul) + inverse rotary + wo_a matmul +
wo_b matmul + copy. Phase prints traced the 50-tok run hang to the first
op inside this block on L32 of pos=7.

This harness drops the hybrid body into a 1000-iter loop with no sync,
matching run_mega.py's mesh + memcfg + tensor layouts. If it hangs in
isolation we have a 200-line repro decoupled from MoE / attn / lm_head.
If it does NOT hang, the bug needs context from earlier in the layer
(lk_d_idx_cmp, lk_d_comp) and we'll escalate.

Pre-stage and run on galaxy:
  copy-file.sh scripts/stress_dsparse.py /tmp/stress_dsparse.py
  copy-file.sh inference.py /tmp/inference.py
  run-test.sh --hw scripts/stress_dsparse.py
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
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

from inference import DeviceSparseAttn, _device_apply_rotary_interleaved


N_ITERS = 1000
LOG_EVERY = 25
SYNC_BETWEEN = False  # set True to insert sync between iters

# Match run_mega config (DeepSeek-V4-Flash defaults from inference.py).
DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
WINDOW_SIZE = 128
O_GROUPS = 8
O_LORA_RANK = 1024
KV_CACHE_SIZE_PAD = 128
PER_GROUP = (N_HEADS * HEAD_DIM) // O_GROUPS  # 4096
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

        # paged_update_cache inputs ----------------------------------------
        # sb.kv_rotated: [1, 1, head_dim] bf16 L1 replicated.
        kv_rot = torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_rot_tt = ttnn.from_torch(kv_rot.contiguous(),
                                    dtype=ttnn.bfloat16, **rep_l1)
        # kv_cache: [1, 1, T, head_dim] bf16 DRAM replicated.
        kv_cache_tt = ttnn.from_torch(
            torch.zeros(1, 1, KV_CACHE_SIZE_PAD, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep_dram)
        # kv_slot: scalar int32 [1] DRAM replicated.
        kv_slot_tt = ttnn.from_torch(
            torch.tensor([7], dtype=torch.int32),
            device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Sharded L1 (0,0) staging memcfg matching run_mega exactly.
        sharded_memcfg = ttnn.MemoryConfig(
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

        # DeviceSparseAttn body inputs -------------------------------------
        # q_rotated: [1, 1, n_heads, head_dim] bf16 L1.
        q_rot = torch.randn(1, 1, N_HEADS, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        q_rot_tt = ttnn.from_torch(q_rot.contiguous(),
                                   dtype=ttnn.bfloat16, **rep_l1)
        # topk_idxs: [B, S, K] int32 TILE; values in [0, T) with some -1.
        K = WINDOW_SIZE
        topk_t = torch.randint(0, KV_CACHE_SIZE_PAD,
                               (B, S, K), dtype=torch.int32)
        topk_t[..., -3:] = -1  # exercise the -1 sentinel branch
        topk_idxs_tt = ttnn.from_torch(topk_t.contiguous(),
                                       dtype=ttnn.int32, **rep_dram)
        attn_sink_cpu = torch.zeros(N_HEADS, dtype=torch.bfloat16)

        # Inverse-rotary tables. Match run_mega max_seq_len rope tables.
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
        start_pos_tt = ttnn.from_torch(
            torch.tensor([[7]], dtype=torch.int32),
            device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        # wo_a / wo_b weights ----------------------------------------------
        wo_a = torch.randn(O_GROUPS, PER_GROUP, O_LORA_RANK,
                           dtype=torch.bfloat16) * 0.02
        wo_b = torch.randn(O_GROUPS * O_LORA_RANK, DIM,
                           dtype=torch.bfloat16) * 0.02
        wo_a_tt = ttnn.as_tensor(wo_a.contiguous(),
                                 dtype=ttnn.bfloat16, **rep_dram)
        wo_b_tt = ttnn.as_tensor(wo_b.contiguous(),
                                 dtype=ttnn.bfloat16, **rep_dram)
        # attn_out destination: [1, 1, dim] bf16 L1.
        attn_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, DIM, dtype=torch.bfloat16),
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

        def call_dsparse():
            kv_4d = ttnn.reshape(kv_rot_tt, [1, B, 1, D])
            kv_4d_sharded = ttnn.to_memory_config(
                kv_4d, memory_config=sharded_memcfg)
            ttnn.experimental.paged_update_cache(
                kv_cache_tt, kv_4d_sharded,
                update_idxs_tensor=kv_slot_tt)

            idxs_tt, valid_tt = dsa._idxs_int_tile_to_idxs_and_mask(
                topk_idxs_tt, B, S, K)
            kv_full_tt = ttnn.reshape(kv_cache_tt, [KV_CACHE_SIZE_PAD, D])
            o_tt = dsa.forward_device(q_rot_tt, kv_full_tt, idxs_tt,
                                      valid_tt, S, K)

            cos = ttnn.embedding(start_pos_tt, cos_full_tt,
                                 layout=ttnn.TILE_LAYOUT)
            sin = ttnn.embedding(start_pos_tt, sin_full_tt,
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
        call_dsparse()
        ttnn.synchronize_device(mesh)
        print(f"[stress] warmup done; running {N_ITERS} iters "
              f"sync_between={SYNC_BETWEEN}", flush=True)

        t0 = time.time()
        last_log = t0
        for it in range(N_ITERS):
            call_dsparse()
            if SYNC_BETWEEN:
                ttnn.synchronize_device(mesh)
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
