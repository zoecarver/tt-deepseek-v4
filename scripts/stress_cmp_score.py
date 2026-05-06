"""Stress test: cmp→score boundary, repro for run_mega cmp→score hang.

In run_mega.py the indexer's `lk_d_idx_cmp` (paged_update_cache writes
pinned to CoreCoord(0,0)) is followed by `lk_d_idx_score` (fused
score+reduce with a PipeNet whose reduce dst-endpoints include (0,0)).
Without `ttnn.synchronize_device` between them the device hangs every
4-5 decode steps. With sync, eager runs reliably but trace capture is
illegal (sync inside a capture is not allowed).

This harness drops the two kernels into a 1000-iter loop with no sync,
matching run_mega.py's mesh + memcfg + tensor layouts exactly. If it
hangs without sync, we have a 100-line repro decoupled from MoE, attn,
lm_head, and the 5-min weight-load cycle.

Pre-stage and run on galaxy:
  copy-file.sh scripts/stress_cmp_score.py /tmp/stress_cmp_score.py
  copy-file.sh tt-lang-kernels/mega/test_lk_d_idx_cmp.py /tmp/test_lk_d_idx_cmp.py
  copy-file.sh tt-lang-kernels/mega/test_lk_d_idx_score.py /tmp/test_lk_d_idx_score.py
  copy-file.sh tt-lang-kernels/mega/_refs.py /tmp/_refs.py
  copy-file.sh tt-lang-kernels/mega/inference.py /tmp/inference.py  # _refs imports it
  run-test.sh --hw scripts/stress_cmp_score.py
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

import torch
import ttnn

from test_lk_d_idx_cmp import (
    make_lk_d_idx_cmp_kernel,
    DIM, CDIM, INDEX_HEAD_DIM, RATIO_PAD, MAX_SEQ_LEN, B,
)
from test_lk_d_idx_score import (
    make_lk_d_idx_score_kernel, INDEX_N_HEADS, T_PAD,
)


N_ITERS = 1000
LOG_EVERY = 25
# Set True to insert ttnn.synchronize_device between cmp and score (the
# workaround that's currently load-bearing in run_mega). Default False so
# we attempt the repro.
SYNC_BETWEEN = False


def main():
    torch.manual_seed(0)

    # Mirror run_mega.py mesh setup so dispatch behavior matches.
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
        rep = dict(
            device=mesh, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # cmp inputs --------------------------------------------------------
        x = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        wkv = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        wgate = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        wkv_gate = torch.cat([wkv, wgate], dim=-1).contiguous()
        ape_padded = torch.randn(
            MAX_SEQ_LEN, CDIM, dtype=torch.bfloat16) * 0.05
        kv_state_front = torch.zeros(
            B, 1, RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        kv_state_back = torch.zeros(
            B, 1, RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        score_state_front = torch.full(
            (B, 1, RATIO_PAD, INDEX_HEAD_DIM), float("-inf"),
            dtype=torch.bfloat16)
        score_state_back = torch.full(
            (B, 1, RATIO_PAD, INDEX_HEAD_DIM), float("-inf"),
            dtype=torch.bfloat16)
        start_pos = torch.tensor([[3]], dtype=torch.int32)
        state_slot = torch.tensor([7], dtype=torch.int32)

        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep)
        wkv_gate_tt = ttnn.as_tensor(wkv_gate, dtype=ttnn.bfloat16, **rep)
        ape_padded_tt = ttnn.as_tensor(
            ape_padded.contiguous(), dtype=ttnn.bfloat16, **rep)
        kv_state_front_tt = ttnn.as_tensor(
            kv_state_front.contiguous(), dtype=ttnn.bfloat16, **rep)
        kv_state_back_tt = ttnn.as_tensor(
            kv_state_back.contiguous(), dtype=ttnn.bfloat16, **rep)
        score_state_front_tt = ttnn.as_tensor(
            score_state_front.contiguous(), dtype=ttnn.bfloat16, **rep)
        score_state_back_tt = ttnn.as_tensor(
            score_state_back.contiguous(), dtype=ttnn.bfloat16, **rep)
        start_pos_tt = ttnn.from_torch(
            start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        state_slot_tt = ttnn.from_torch(
            state_slot, device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        sharded_memcfg = ttnn.MemoryConfig(
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
            dtype=ttnn.bfloat16, **rep)
        idx_cmp_score_tt = ttnn.from_torch(
            torch.zeros(1, 1, 2 * INDEX_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)

        # score inputs ------------------------------------------------------
        wproj = torch.randn(
            DIM, INDEX_N_HEADS, dtype=torch.bfloat16) * 0.02
        scale = float(INDEX_HEAD_DIM ** -0.5) * float(INDEX_N_HEADS ** -0.5)
        wproj_scaled = (wproj.to(torch.float32) * scale).to(torch.bfloat16)
        q_idx = torch.randn(
            1, 1, INDEX_N_HEADS, INDEX_HEAD_DIM,
            dtype=torch.bfloat16) * 0.1
        idx_kv_cache = torch.randn(
            1, 1, T_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1

        wproj_scaled_tt = ttnn.as_tensor(
            wproj_scaled.contiguous(), dtype=ttnn.bfloat16, **rep)
        q_idx_tt = ttnn.as_tensor(
            q_idx.contiguous(), dtype=ttnn.bfloat16, **rep)
        idx_kv_cache_tt = ttnn.as_tensor(
            idx_kv_cache.contiguous(), dtype=ttnn.bfloat16, **rep)
        idx_score_tt = ttnn.from_torch(
            torch.zeros(1, 1, T_PAD, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)

        cmp_kernel = make_lk_d_idx_cmp_kernel(
            mesh, sharded_input_memcfg=sharded_memcfg)
        score_kernel = make_lk_d_idx_score_kernel(mesh)

        def call_cmp():
            cmp_kernel(
                x_tt, wkv_gate_tt, ape_padded_tt,
                start_pos_tt, state_slot_tt,
                kv_state_front_tt, kv_state_back_tt,
                score_state_front_tt, score_state_back_tt,
                idx_cmp_kv_tt, idx_cmp_score_tt,
            )

        def call_score():
            score_kernel(
                x_tt, wproj_scaled_tt, q_idx_tt,
                idx_kv_cache_tt, idx_score_tt,
            )

        print("[stress] warmup ...", flush=True)
        call_cmp()
        call_score()
        ttnn.synchronize_device(mesh)
        print(f"[stress] warmup done; running {N_ITERS} iters "
              f"sync_between={SYNC_BETWEEN}", flush=True)

        t0 = time.time()
        last_log = t0
        for it in range(N_ITERS):
            call_cmp()
            if SYNC_BETWEEN:
                ttnn.synchronize_device(mesh)
            call_score()
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
