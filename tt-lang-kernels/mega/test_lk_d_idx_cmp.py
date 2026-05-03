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
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0


DIM = 4096
INDEX_HEAD_DIM = 128
COFF = 2                                # overlap=True
CDIM = COFF * INDEX_HEAD_DIM            # 256
RATIO = 4
RATIO_PAD = 32                          # _RMS_TILE
MAX_SEQ_LEN = 512
B = 1


def make_lk_d_idx_cmp_kernel():
    """Placeholder mega kernel for indexer compressor non-emit path.

    Inputs:
      x:              [1, 1, dim] bf16  — replicated activation
      wkv_w:          [dim, coff*head_dim] bf16
      wgate_w:        [dim, coff*head_dim] bf16
      ape_padded:     [max_seq_len, coff*head_dim] bf16
      start_pos:      [1, 1] uint32
      state_slot:     [1] int32
      state buffers (in/out): kv_state_front/back, score_state_front/back
                              each [1, 1, ratio_pad, head_dim] bf16
    Output:
      kv_out:         [1, 1, cdim] bf16  — pre-wkv-all_gather wkv result
      score_out:      [1, 1, cdim] bf16  — pre-wgate-all_gather + APE result
    """
    @ttl.operation(grid="auto")
    def lk_d_idx_cmp_kernel(x, wkv_w, wgate_w, ape_padded, start_pos,
                            state_slot,
                            kv_state_front, kv_state_back,
                            score_state_front, score_state_back,
                            kv_out, score_out):
        @ttl.compute()
        def compute():
            pass

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

        kernel = make_lk_d_idx_cmp_kernel()
        kv_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, CDIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        score_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, CDIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt, start_pos_tt,
               state_slot_tt,
               kv_state_front_tt, kv_state_back_tt,
               score_state_front_tt, score_state_back_tt,
               kv_out_tt, score_out_tt)
        kernel_kv_host = download_chip0(mesh, mesh_shape, kv_out_tt)
        kernel_score_host = download_chip0(mesh, mesh_shape, score_out_tt)

        ok_kv = report_pcc("Lk-D-idx-cmp/kv", ref_kv_host, kernel_kv_host)
        ok_sc = report_pcc("Lk-D-idx-cmp/score", ref_score_host, kernel_score_host)
        sys.exit(0 if (ok_kv and ok_sc) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
