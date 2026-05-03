"""Lk-D-idx-emit PCC test: indexer compressor emit branch.

Reference covers `DeviceCompressor._emit_body` (overlap=True, rotate=True
for the indexer's compressor). Specifically (from inference.py):
- compressor_softmax_sum_norm (existing TTL kernel) → kv_normed via slice + reshape
- pick cos/sin via embedding(start_pos, cos_compressor); reshape
- slice nope/rope, rotary on rope, concat
- _device_rotate_activation (Walsh-Hadamard, rotate=True)
- paged_update_cache to indexer kv_cache at emit_slot
- 4× compressor_slot_shift kernel (existing TTL) + 4× ttnn.copy

Boundaries: pre-CCL is indexer.compressor.wgate all_gather; post-CCL is
indexer.weights_proj all_gather.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    _device_apply_rotary_interleaved, _device_rotate_activation,
    _sylvester_hadamard,
    _get_ttl_compressor_softmax_sum_norm_kernel,
    _get_ttl_compressor_slot_shift_kernel,
    _compressor_softmax_sum_norm_masks, _compressor_shift_matrix,
    _RMS_TILE,
)


INDEX_HEAD_DIM = 128
ROPE_HEAD_DIM = 64
RATIO = 4
RATIO_PAD = _RMS_TILE  # 32
T_PAD = 128            # placeholder; in inference this is max_seq_len/ratio padded to TILE
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B = 1


def make_lk_d_idx_emit_kernel():
    """Placeholder mega kernel for Lk-D-idx-emit.

    Inputs:
      kv_state_front_2d:    [ratio_pad, head_dim] bf16
      kv_state_back_2d:     [ratio_pad, head_dim] bf16
      score_state_front_2d: [ratio_pad, head_dim] bf16
      score_state_back_2d:  [ratio_pad, head_dim] bf16
      cssn_mask_front/back/pad: [TILE, TILE] bf16
      norm_gamma:           [TILE, head_dim] bf16  (packed gamma)
      cos_compressor:       [max_seq_len, rope_head_dim/2] bf16
      sin_compressor:       [max_seq_len, rope_head_dim/2] bf16
      start_pos:            [1, 1] uint32
      H:                    [head_dim, head_dim] bf16
      kv_cache:              [1, 1, T_pad, head_dim] bf16  (in place)
      emit_slot:            [1] int32
      shift_P:              [ratio_pad, ratio_pad] bf16
    Outputs:
      kv_normed:            [1, 1, head_dim] bf16  — emitted compressed kv (rotated)
      (state buffers updated in place)
    """
    @ttl.operation(grid="auto")
    def lk_d_idx_emit_kernel(
        kv_state_front_2d, kv_state_back_2d,
        score_state_front_2d, score_state_back_2d,
        cssn_mask_front, cssn_mask_back, cssn_mask_pad,
        norm_gamma, scaler,
        cos_compressor, sin_compressor, start_pos,
        H, kv_cache, emit_slot, shift_P,
        kv_normed_out,
    ):
        @ttl.compute()
        def compute():
            pass

    return lk_d_idx_emit_kernel


def reference(mesh,
              kv_state_front_2d, kv_state_back_2d,
              score_state_front_2d, score_state_back_2d,
              cssn_mask_front, cssn_mask_back, cssn_mask_pad,
              norm_gamma_tt, scaler_tt,
              cos_compressor_tt, sin_compressor_tt, start_pos_tt,
              H_tt, kv_cache_tt, emit_slot_tt, shift_P_tt,
              cssn_out_tt,
              kv_state_front_out_2d, kv_state_back_out_2d,
              score_state_front_out_2d, score_state_back_out_2d,
              sharded_input_memcfg):
    d = INDEX_HEAD_DIM
    rd = ROPE_HEAD_DIM
    rd_half = rd // 2

    cssn = _get_ttl_compressor_softmax_sum_norm_kernel(RATIO, RATIO_PAD, d, NORM_EPS)
    slot_shift = _get_ttl_compressor_slot_shift_kernel(1, RATIO_PAD, d)

    # Fused softmax + sum + norm.
    cssn(
        kv_state_front_2d, kv_state_back_2d,
        score_state_front_2d, score_state_back_2d,
        cssn_mask_front, cssn_mask_back, cssn_mask_pad,
        norm_gamma_tt, scaler_tt, cssn_out_tt,
    )
    kv_2d = ttnn.slice(cssn_out_tt, [0, 0], [B, d])
    kv_normed = ttnn.reshape(kv_2d, [B, 1, d])

    # Rotary on rope half (cos/sin from compressor variants).
    cos = ttnn.embedding(start_pos_tt, cos_compressor_tt, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(start_pos_tt, sin_compressor_tt, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, 1, rd_half])
    sin = ttnn.reshape(sin, [1, 1, rd_half])
    kv_nope = ttnn.slice(kv_normed, [0, 0, 0], [B, 1, d - rd])
    kv_rope = ttnn.slice(kv_normed, [0, 0, d - rd], [B, 1, d])
    kv_rope = _device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)
    kv_normed = ttnn.concat([kv_nope, kv_rope], dim=-1)

    # rotate=True for the indexer's compressor.
    kv_normed = _device_rotate_activation(ttnn, kv_normed, H_tt)

    # Write compressed slot into kv_cache.
    kv_4d = ttnn.reshape(kv_normed, [1, B, 1, d])
    kv_sharded = ttnn.to_memory_config(kv_4d, memory_config=sharded_input_memcfg)
    ttnn.experimental.paged_update_cache(
        kv_cache_tt, kv_sharded, update_idxs_tensor=emit_slot_tt)

    # 4× slot-shift + ttnn.copy.
    slot_shift(kv_state_front_2d, shift_P_tt, kv_state_front_out_2d)
    ttnn.copy(kv_state_front_out_2d, kv_state_front_2d)
    slot_shift(kv_state_back_2d, shift_P_tt, kv_state_back_out_2d)
    ttnn.copy(kv_state_back_out_2d, kv_state_back_2d)
    slot_shift(score_state_front_2d, shift_P_tt, score_state_front_out_2d)
    ttnn.copy(score_state_front_out_2d, score_state_front_2d)
    slot_shift(score_state_back_2d, shift_P_tt, score_state_back_out_2d)
    ttnn.copy(score_state_back_out_2d, score_state_back_2d)

    return kv_normed


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        # State buffers — randomized so the cssn output is non-trivial.
        kv_state_front = torch.randn(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_state_back = torch.randn(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_front = torch.randn(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_back = torch.randn(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        # Mask the padding rows of the score buffers to -inf (matches inference.py).
        score_state_front[2 * RATIO:, :] = float("-inf")
        score_state_back[2 * RATIO:, :] = float("-inf")

        mf, mb, mp = _compressor_softmax_sum_norm_masks(RATIO)
        gamma = torch.ones(_RMS_TILE, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        scaler = torch.ones(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16)
        cos_compressor = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        sin_compressor = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        H_mat = (_sylvester_hadamard(INDEX_HEAD_DIM) *
                 (INDEX_HEAD_DIM ** -0.5)).to(torch.bfloat16)
        kv_cache_init = torch.zeros(1, 1, T_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        shift_P = _compressor_shift_matrix(RATIO, RATIO_PAD)
        start_pos = torch.tensor([[RATIO - 1]], dtype=torch.int32)
        emit_slot = torch.tensor([0], dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def upload_2d(t, dt=ttnn.bfloat16):
            return ttnn.as_tensor(t.contiguous(), dtype=dt, **rep)

        kv_state_front_tt = upload_2d(kv_state_front)
        kv_state_back_tt = upload_2d(kv_state_back)
        score_state_front_tt = upload_2d(score_state_front)
        score_state_back_tt = upload_2d(score_state_back)
        mf_tt = upload_2d(mf)
        mb_tt = upload_2d(mb)
        mp_tt = upload_2d(mp)
        gamma_tt = upload_2d(gamma)
        scaler_tt = upload_2d(scaler)
        cos_compressor_tt = upload_2d(cos_compressor)
        sin_compressor_tt = upload_2d(sin_compressor)
        H_tt = upload_2d(H_mat)
        kv_cache_tt = upload_2d(kv_cache_init)
        shift_P_tt = upload_2d(shift_P)
        start_pos_tt = ttnn.from_torch(
            start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        emit_slot_tt = ttnn.from_torch(
            emit_slot, device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        cssn_out_tt = ttnn.from_torch(
            torch.zeros(_RMS_TILE, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)

        # Slot-shift double buffers (zeros for kv, -inf for score).
        zero_pad = torch.zeros(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        ninf_pad = torch.full_like(zero_pad, float("-inf"))
        kv_state_front_out_tt = upload_2d(zero_pad)
        kv_state_back_out_tt = upload_2d(zero_pad)
        score_state_front_out_tt = upload_2d(ninf_pad)
        score_state_back_out_tt = upload_2d(ninf_pad)

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

        ref_kv_normed_tt = reference(
            mesh,
            kv_state_front_tt, kv_state_back_tt,
            score_state_front_tt, score_state_back_tt,
            mf_tt, mb_tt, mp_tt,
            gamma_tt, scaler_tt,
            cos_compressor_tt, sin_compressor_tt, start_pos_tt,
            H_tt, kv_cache_tt, emit_slot_tt, shift_P_tt,
            cssn_out_tt,
            kv_state_front_out_tt, kv_state_back_out_tt,
            score_state_front_out_tt, score_state_back_out_tt,
            sharded_memcfg)
        ref_host = download_chip0(mesh, mesh_shape, ref_kv_normed_tt)

        kernel = make_lk_d_idx_emit_kernel()
        kv_normed_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(
            kv_state_front_tt, kv_state_back_tt,
            score_state_front_tt, score_state_back_tt,
            mf_tt, mb_tt, mp_tt,
            gamma_tt, scaler_tt,
            cos_compressor_tt, sin_compressor_tt, start_pos_tt,
            H_tt, kv_cache_tt, emit_slot_tt, shift_P_tt,
            kv_normed_out_tt,
        )
        kernel_host = download_chip0(mesh, mesh_shape, kv_normed_out_tt)

        ok = report_pcc("Lk-D-idx-emit", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
