"""Lk-Dsparse PCC test: sparse_attn body + inverse rotary + wo_a + wo_b.

Reference covers everything from the topk indices ready → wo_b matmul,
sans the wo_b all_gather. Mirrors the attn.kv_update + attn.sparse +
attn.o phases of `DeviceAttention.forward_device`:
- paged_update_cache(kv_cache, kv_tt, kv_slot)
- _idxs_int_tile_to_idxs_and_mask(topk_idxs)
- DeviceSparseAttn.forward_device: gather + score matmul + masked + sink
  concat + softmax + drop sink + output matmul
- inverse rotary on o[..., -rd:]
- group reshape + block-diag wo_a matmul (replicated weight, no CCL)
- reshape + permute + reshape
- ttnn.matmul(o_flat, wo_b)  (pre-all_gather)

Boundaries: pre-CCL is whatever produced the topk indices (Lk-D-topk
output, or no CCL if compress is absent); post-CCL is wo_b all_gather.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    DeviceSparseAttn, _device_apply_rotary_interleaved,
)


DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
N_GROUPS = 8
O_LORA_RANK = 1024
KV_CACHE_SIZE_PAD = 128
WIN = 128
K = WIN                            # window-only topk for the test (no compressor)
MAX_SEQ_LEN = 512
B, S = 1, 1


def make_lk_dsparse_kernel():
    """Placeholder mega kernel for Lk-Dsparse.

    Inputs:
      q:                  [1, 1, n_heads, head_dim] bf16
      kv:                 [1, 1, head_dim] bf16  — current step's kv (already act_quant'd)
      kv_cache:           [1, 1, kv_cache_size_pad, head_dim] bf16  (in place)
      kv_slot:            [1] int32
      topk_idxs:          [1, 1, K] int32   (with -1 sentinels)
      attn_sink:          [1, 1, n_heads, 1] bf16  (per-head sink)
      cos_full, sin_full: [max_seq_len, rope_head_dim/2] bf16
      start_pos:          [1, 1] uint32
      wo_a_w:             [n_groups, in_per_group, o_lora_rank] bf16  (replicated)
      wo_b_w:             [n_groups*o_lora_rank, dim] bf16
      softmax_scale:      [TILE, TILE] bf16  (broadcast scalar tile)
    Output:
      out:                [1, 1, dim] bf16  — pre-wo_b-all_gather
    """
    @ttl.operation(grid="auto")
    def lk_dsparse_kernel(
        q, kv, kv_cache, kv_slot, topk_idxs,
        attn_sink, cos_full, sin_full, start_pos,
        wo_a_w, wo_b_w, softmax_scale,
        out,
    ):
        @ttl.compute()
        def compute():
            pass

    return lk_dsparse_kernel


def reference(mesh, q_tt, kv_tt, kv_cache_tt, kv_slot_tt, topk_idxs_tt,
              attn_sink_cpu, cos_full_tt, sin_full_tt, start_pos_tt,
              wo_a_w_tt, wo_b_w_tt, softmax_scale_value,
              sharded_input_memcfg):
    H = N_HEADS
    D = HEAD_DIM
    rd = ROPE_HEAD_DIM
    rd_half = rd // 2

    # kv_cache update at window slot (paged_update_cache).
    kv_4d = ttnn.reshape(kv_tt, [1, B, 1, D])
    kv_4d_sharded = ttnn.to_memory_config(kv_4d, memory_config=sharded_input_memcfg)
    ttnn.experimental.paged_update_cache(
        kv_cache_tt, kv_4d_sharded, update_idxs_tensor=kv_slot_tt)

    # Build sparse_attn.
    sparse = DeviceSparseAttn(
        mesh=mesh, attn_sink=attn_sink_cpu,
        softmax_scale=softmax_scale_value)

    # Convert int32 topk indices into (uint32 row-major idxs, bf16 valid mask).
    idxs_tt, valid_tt = sparse._idxs_int_tile_to_idxs_and_mask(
        topk_idxs_tt, B, S, K)

    # Reshape kv_cache for the gather op.
    kv_full_tt = ttnn.reshape(kv_cache_tt, [KV_CACHE_SIZE_PAD, D])

    # Sparse attention body.
    o_tt = sparse.forward_device(q_tt, kv_full_tt, idxs_tt, valid_tt, S, K)
    # o_tt: [B, S, H, D].

    # Inverse rotary on o[..., -rd:].
    cos = ttnn.embedding(start_pos_tt, cos_full_tt, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(start_pos_tt, sin_full_tt, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, S, 1, rd_half])
    sin = ttnn.reshape(sin, [1, S, 1, rd_half])
    o_nope = ttnn.slice(o_tt, [0, 0, 0, 0], [B, S, H, D - rd])
    o_rope = ttnn.slice(o_tt, [0, 0, 0, D - rd], [B, S, H, D])
    o_rope = _device_apply_rotary_interleaved(ttnn, o_rope, cos, sin, inverse=True)
    o_tt = ttnn.concat([o_nope, o_rope], dim=-1)

    # Group reshape + block-diag wo_a (replicated weight).
    per_group = (H * D) // N_GROUPS
    o_perm = ttnn.reshape(o_tt, [B, S, N_GROUPS, per_group])
    o_perm = ttnn.permute(o_perm, [2, 0, 1, 3])
    o_g = ttnn.reshape(o_perm, [N_GROUPS, B * S, per_group])
    o_wo_a_g = ttnn.matmul(o_g, wo_a_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    o_wo_a_rs = ttnn.reshape(o_wo_a_g, [N_GROUPS, B, S, O_LORA_RANK])
    o_wo_a = ttnn.permute(o_wo_a_rs, [1, 2, 0, 3])
    o_flat = ttnn.reshape(o_wo_a, [B, S, N_GROUPS * O_LORA_RANK])

    # wo_b matmul (no all_gather; weight replicated for the test).
    return ttnn.matmul(o_flat, wo_b_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        q = torch.randn(B, S, N_HEADS, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_cache = torch.randn(1, 1, KV_CACHE_SIZE_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        # Random valid topk indices in [0, KV_CACHE_SIZE_PAD).
        topk_idxs = torch.randint(0, KV_CACHE_SIZE_PAD, (B, S, K), dtype=torch.int32)
        attn_sink = torch.randn(N_HEADS, dtype=torch.float32) * 0.1
        cos_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        sin_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        in_per_group = (N_HEADS * HEAD_DIM) // N_GROUPS
        wo_a_w = torch.randn(N_GROUPS, in_per_group, O_LORA_RANK, dtype=torch.bfloat16) * 0.02
        wo_b_w = torch.randn(N_GROUPS * O_LORA_RANK, DIM, dtype=torch.bfloat16) * 0.02
        kv_slot = torch.tensor([1], dtype=torch.int32)
        start_pos = torch.tensor([[1]], dtype=torch.int32)
        softmax_scale = float(HEAD_DIM ** -0.5)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        q_tt = ttnn.as_tensor(q.contiguous(), dtype=ttnn.bfloat16, **rep)
        kv_tt = ttnn.as_tensor(kv.contiguous(), dtype=ttnn.bfloat16, **rep)
        kv_cache_tt = ttnn.as_tensor(kv_cache.contiguous(),
                                     dtype=ttnn.bfloat16, **rep)
        topk_idxs_tt = ttnn.as_tensor(topk_idxs.contiguous(),
                                      dtype=ttnn.int32, **rep)
        cos_full_tt = ttnn.as_tensor(cos_full.contiguous(),
                                     dtype=ttnn.bfloat16, **rep)
        sin_full_tt = ttnn.as_tensor(sin_full.contiguous(),
                                     dtype=ttnn.bfloat16, **rep)
        wo_a_w_tt = ttnn.as_tensor(wo_a_w.contiguous(),
                                   dtype=ttnn.bfloat16, **rep)
        wo_b_w_tt = ttnn.as_tensor(wo_b_w.contiguous(),
                                   dtype=ttnn.bfloat16, **rep)
        kv_slot_tt = ttnn.from_torch(
            kv_slot, device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        start_pos_tt = ttnn.from_torch(
            start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        scale_tile = ttnn.as_tensor(
            torch.full((32, 32), softmax_scale, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        sink_4d_tt = ttnn.as_tensor(
            torch.zeros(1, 1, N_HEADS, 1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)

        sharded_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                (32, HEAD_DIM),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        ref_out_tt = reference(
            mesh, q_tt, kv_tt, kv_cache_tt, kv_slot_tt, topk_idxs_tt,
            attn_sink, cos_full_tt, sin_full_tt, start_pos_tt,
            wo_a_w_tt, wo_b_w_tt, softmax_scale, sharded_memcfg)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_dsparse_kernel()
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(q_tt, kv_tt, kv_cache_tt, kv_slot_tt, topk_idxs_tt,
               sink_4d_tt, cos_full_tt, sin_full_tt, start_pos_tt,
               wo_a_w_tt, wo_b_w_tt, scale_tile, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-Dsparse", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
