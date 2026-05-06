"""Lk-D-idx-cmp PCC test: indexer compressor wkv + wgate + APE + state updates.

Combined zone for the indexer's compressor's two linears + APE + state
buffer writes. In inference.py this is `DeviceCompressor.forward_device`
(non-emit branch path), called from `DeviceIndexer.forward_device_score`
via `self.dc.forward_device(x_tt, B, start_pos, start_pos_tt=...)`.

Reference covers (from `DeviceCompressor.forward_device`, sans CCLs):
- ttnn.matmul(x, wkv)
- ttnn.matmul(x, wgate)
- embedding(start_pos, ape_padded), reshape, add(score, ape_slot)
- 4× slice front/back (overlap=True)
- 4× paged_update_cache to state_front/back, score_front/back

The wkv and wgate all_gathers between them are excluded.

Kernel inlines the SUMMA matmul and reuses it for both linears. APE
(embedding + add) and the four `paged_update_cache` calls remain as
ttnn glue — see TODO below.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark


DIM = 4096
INDEX_HEAD_DIM = 128
COFF = 2                                # overlap=True
CDIM = COFF * INDEX_HEAD_DIM            # 256
RATIO = 4
RATIO_PAD = 32                          # _RMS_TILE
MAX_SEQ_LEN = 512
B = 1
TILE = 32

# TODO: mega the embedding(start_pos, ape_padded) + reshape + add and the
# four paged_update_cache calls remain ttnn here. Lowering them to
# tt-lang requires:
#   * embedding -> a runtime-indexed ttl.copy from ape_padded[start_pos]
#     into a 1-tile buffer (start_pos is a [1] uint32 device tensor).
#   * paged_update_cache -> a runtime-indexed ttl.copy that writes the
#     newly produced kv/score row into cache[state_slot]. state_slot is
#     also a device-side scalar.
# Both need data-movement-time index reads. Punt for now; once we have a
# `ttl.copy_indexed` (or equivalent) primitive these all collapse into
# one mega-kernel dispatch.


def _make_summa_matmul_kernel(M: int, K: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    """Pure-SUMMA matmul kernel (same shape as Lk-A/Lk-B/Lk-D-idx-q)."""
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Kp != 1:
        raise ValueError(f"K_parts must be 1, got {Kp}")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape (tiles): Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})")
    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Mb % Mp or Nb % Np:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} Mp={Mp} Np={Np}")
    M_BPN = Mb // Mp
    N_BPN = Nb // Np

    @ttl.operation(grid=(Np, Mp), fp32_dest_acc_en=fp32_dest_acc_en)
    def summa_matmul(a, w, out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p))
                   for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp)))
                   for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = out_cb.reserve()
                    for _ in range(Kb):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        p += a_blk @ b_blk

        @ttl.datamovement()
        def dm_read():
            _, row_c = ttl.node(dims=2)
            for local_mb in range(M_BPN):
                mb = row_c * M_BPN + local_mb
                mr = mb * bm
                for _ in range(N_BPN):
                    for kb in range(Kb):
                        kc = kb * bk
                        a_blk = a_cb.reserve()

                        def read_a(pipe):
                            ttl.copy(a[mr:mr + bm, kc:kc + bk], a_blk).wait()
                            ttl.copy(a_blk, pipe).wait()

                        mcast_a_net.if_src(read_a)
                        mcast_a_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            for local_mb in range(M_BPN):
                mb = row_c * M_BPN + local_mb
                mr = mb * bm
                for local_nb in range(N_BPN):
                    nb = col_c * N_BPN + local_nb
                    nc = nb * bn
                    for kb in range(Kb):
                        kc = kb * bk
                        b_blk = b_cb.reserve()

                        def read_b(pipe):
                            ttl.copy(w[kc:kc + bk, nc:nc + bn], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    o = out_cb.wait()
                    ttl.copy(o, out[mr:mr + bm, nc:nc + bn]).wait()

    return summa_matmul


def make_lk_d_idx_cmp_kernel(mesh, sharded_input_memcfg):
    """Mega kernel for Lk-D-idx-cmp.

    Single fused SUMMA matmul against `cat([wkv, wgate], dim=-1)` so that
    x is mcast once (was mcast twice when wkv and wgate were separate
    matmuls). Output is sliced into kv and score halves. APE +
    paged_update_cache remain ttnn glue (see TODO above).
    """
    d = INDEX_HEAD_DIM
    c = CDIM
    M_PAD = TILE
    NCAT = 2 * CDIM
    state: dict = {}

    # Triage: replaced fused SUMMA ttl matmul with ttnn.matmul to bisect
    # the cmp→score isolated-stress hang. Restore once root-cause is known.

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def lk_d_idx_cmp_kernel(x_tt, wkv_gate_w_tt, ape_padded_tt,
                            start_pos_tt, state_slot_tt,
                            kv_state_front_tt, kv_state_back_tt,
                            score_state_front_tt, score_state_back_tt,
                            kv_out, score_out):
        # ttnn.matmul replacement for the SUMMA kernel. x_tt is [1, 1, DIM]
        # bf16 replicated; wkv_gate_w_tt is [DIM, NCAT]. Output is
        # [1, 1, NCAT] which we slice into kv | score halves.
        kv_gate_out = ttnn.matmul(
            x_tt, wkv_gate_w_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Slice halves -> [B, 1, CDIM] each.
        kv_3d = ttnn.slice(kv_gate_out, [0, 0, 0], [B, 1, CDIM])
        score_3d = ttnn.slice(kv_gate_out, [0, 0, CDIM], [B, 1, NCAT])

        # APE add (TODO: mega).
        ape_slot = ttnn.embedding(
            start_pos_tt, ape_padded_tt, layout=ttnn.TILE_LAYOUT)
        score_3d = ttnn.add(score_3d, ttnn.reshape(ape_slot, [1, 1, c]))

        # paged_update_cache writes.
        # TODO: mega fusion blocked (bucket #1 — unwired): element_write
        # available; lower the 4 slot-writes into the fused kernel. C10.
        kv_front = ttnn.slice(kv_3d, [0, 0, 0], [B, 1, d])
        kv_back = ttnn.slice(kv_3d, [0, 0, d], [B, 1, c])
        score_front = ttnn.slice(score_3d, [0, 0, 0], [B, 1, d])
        score_back = ttnn.slice(score_3d, [0, 0, d], [B, 1, c])

        def pug(cache_tt, x_3d_tt):
            x_4d = ttnn.reshape(x_3d_tt, [1, B, 1, d])
            x_sharded = ttnn.to_memory_config(
                x_4d, memory_config=sharded_input_memcfg)
            ttnn.experimental.paged_update_cache(
                cache_tt, x_sharded, update_idxs_tensor=state_slot_tt)

        pug(kv_state_front_tt, kv_front)
        pug(kv_state_back_tt, kv_back)
        pug(score_state_front_tt, score_front)
        pug(score_state_back_tt, score_back)

        ttnn.copy(kv_3d, kv_out)
        ttnn.copy(score_3d, score_out)

    return lk_d_idx_cmp_kernel


def reference(mesh, x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt, start_pos_tt,
              state_slot_tt,
              kv_state_front_tt, kv_state_back_tt,
              score_state_front_tt, score_state_back_tt,
              sharded_input_memcfg):
    d = INDEX_HEAD_DIM
    c = CDIM

    # Linears (no CCLs in test).
    kv_tt = ttnn.matmul(x_tt, wkv_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    score_tt = ttnn.matmul(x_tt, wgate_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # APE add.
    ape_slot = ttnn.embedding(
        start_pos_tt, ape_padded_tt, layout=ttnn.TILE_LAYOUT)
    score_tt = ttnn.add(score_tt, ttnn.reshape(ape_slot, [1, 1, c]))

    # overlap=True slices + paged_update_cache calls.
    kv_front = ttnn.slice(kv_tt, [0, 0, 0], [B, 1, d])
    kv_back = ttnn.slice(kv_tt, [0, 0, d], [B, 1, c])
    score_front = ttnn.slice(score_tt, [0, 0, 0], [B, 1, d])
    score_back = ttnn.slice(score_tt, [0, 0, d], [B, 1, c])

    def pug(cache_tt, x_3d_tt):
        x_4d = ttnn.reshape(x_3d_tt, [1, B, 1, d])
        x_sharded = ttnn.to_memory_config(x_4d, memory_config=sharded_input_memcfg)
        ttnn.experimental.paged_update_cache(
            cache_tt, x_sharded, update_idxs_tensor=state_slot_tt)

    pug(kv_state_front_tt, kv_front)
    pug(kv_state_back_tt, kv_back)
    pug(score_state_front_tt, score_front)
    pug(score_state_back_tt, score_back)

    return kv_tt, score_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        x = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        wkv_w = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        wgate_w = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        ape_padded = torch.randn(MAX_SEQ_LEN, CDIM, dtype=torch.bfloat16) * 0.05
        kv_state_front = torch.zeros(B, 1, RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        kv_state_back = torch.zeros(B, 1, RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        score_state_front = torch.full(
            (B, 1, RATIO_PAD, INDEX_HEAD_DIM), float("-inf"), dtype=torch.bfloat16)
        score_state_back = torch.full(
            (B, 1, RATIO_PAD, INDEX_HEAD_DIM), float("-inf"), dtype=torch.bfloat16)
        start_pos = torch.tensor([[3]], dtype=torch.int32)
        state_slot = torch.tensor([RATIO + 3 % RATIO], dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep)
        wkv_w_tt = ttnn.as_tensor(wkv_w.contiguous(), dtype=ttnn.bfloat16, **rep)
        wgate_w_tt = ttnn.as_tensor(wgate_w.contiguous(), dtype=ttnn.bfloat16, **rep)
        # Pre-concat for the fused kernel (offline, would happen at offload time).
        wkv_gate_w = torch.cat([wkv_w, wgate_w], dim=-1).contiguous()
        wkv_gate_w_tt = ttnn.as_tensor(wkv_gate_w, dtype=ttnn.bfloat16, **rep)
        ape_padded_tt = ttnn.as_tensor(ape_padded.contiguous(), dtype=ttnn.bfloat16, **rep)
        kv_state_front_tt = ttnn.as_tensor(kv_state_front.contiguous(),
                                           dtype=ttnn.bfloat16, **rep)
        kv_state_back_tt = ttnn.as_tensor(kv_state_back.contiguous(),
                                          dtype=ttnn.bfloat16, **rep)
        score_state_front_tt = ttnn.as_tensor(score_state_front.contiguous(),
                                              dtype=ttnn.bfloat16, **rep)
        score_state_back_tt = ttnn.as_tensor(score_state_back.contiguous(),
                                             dtype=ttnn.bfloat16, **rep)
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
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                (32, INDEX_HEAD_DIM),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        ref_kv_tt, ref_score_tt = reference(
            mesh, x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt,
            start_pos_tt, state_slot_tt,
            kv_state_front_tt, kv_state_back_tt,
            score_state_front_tt, score_state_back_tt,
            sharded_memcfg)
        ref_kv_host = download_chip0(mesh, mesh_shape, ref_kv_tt)
        ref_score_host = download_chip0(mesh, mesh_shape, ref_score_tt)

        kernel = make_lk_d_idx_cmp_kernel(mesh, sharded_memcfg)
        kv_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, CDIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        score_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, CDIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(x_tt, wkv_gate_w_tt, ape_padded_tt, start_pos_tt,
               state_slot_tt,
               kv_state_front_tt, kv_state_back_tt,
               score_state_front_tt, score_state_back_tt,
               kv_out_tt, score_out_tt)
        kernel_kv_host = download_chip0(mesh, mesh_shape, kv_out_tt)
        kernel_score_host = download_chip0(mesh, mesh_shape, score_out_tt)

        ok_kv = report_pcc("Lk-D-idx-cmp/kv", ref_kv_host, kernel_kv_host)
        ok_sc = report_pcc("Lk-D-idx-cmp/score", ref_score_host, kernel_score_host)

        benchmark("Lk-D-idx-cmp ref",
                  lambda: reference(mesh, x_tt, wkv_w_tt, wgate_w_tt,
                                    ape_padded_tt, start_pos_tt, state_slot_tt,
                                    kv_state_front_tt, kv_state_back_tt,
                                    score_state_front_tt, score_state_back_tt,
                                    sharded_memcfg),
                  mesh)
        benchmark("Lk-D-idx-cmp ttl",
                  lambda: kernel(x_tt, wkv_gate_w_tt, ape_padded_tt,
                                 start_pos_tt, state_slot_tt,
                                 kv_state_front_tt, kv_state_back_tt,
                                 score_state_front_tt, score_state_back_tt,
                                 kv_out_tt, score_out_tt),
                  mesh)

        sys.exit(0 if (ok_kv and ok_sc) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
