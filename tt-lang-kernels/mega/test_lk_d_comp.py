"""Lk-D-comp PCC test: attn-side compressor body (overlap=True path).

Same structure as Lk-D-idx-cmp + Lk-D-idx-emit, but for the attn's own
compressor (head_dim=512, rotate=False). Reference covers the entire
`DeviceCompressor.forward_device` body for a non-rotated overlap=True
compressor on an emit step.

For a non-emit step the body is just the linear matmuls + APE +
state-front updates (no _emit_body). This test exercises the emit path;
the non-emit path is structurally a subset.

Boundaries: pre-CCL is wkv all_gather (Lk-D1's tail) or compressor.wkv
all_gather; post-CCL is whichever attn linear comes next.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    _device_apply_rotary_interleaved,
    _get_ttl_compressor_softmax_sum_norm_kernel,
    _get_ttl_compressor_slot_shift_kernel,
    _compressor_softmax_sum_norm_masks, _compressor_shift_matrix,
    _RMS_TILE,
)


DIM = 4096
HEAD_DIM = 512                        # attn head_dim
ROPE_HEAD_DIM = 64
COFF = 2
CDIM = COFF * HEAD_DIM                # 1024
RATIO = 4
RATIO_PAD = _RMS_TILE                 # 32
T_PAD = 128
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B = 1


def make_lk_d_comp_kernel():
    """Placeholder mega kernel for the attn-side compressor (emit path).

    Inputs (analogous to Lk-D-idx-cmp + Lk-D-idx-emit but at d=head_dim=512,
    rotate=False — no Walsh-Hadamard step):
      x:                       [1, 1, dim] bf16
      wkv_w, wgate_w:          [dim, coff*head_dim] bf16
      ape_padded:              [max_seq_len, coff*head_dim] bf16
      cos_compressor:          [max_seq_len, rope_head_dim/2] bf16
      sin_compressor:          [max_seq_len, rope_head_dim/2] bf16
      start_pos:               [1, 1] uint32
      state_slot, emit_slot:   [1] int32
      kv_state_front/back_2d:  [ratio_pad, head_dim] bf16
      score_state_front/back_2d: [ratio_pad, head_dim] bf16
      cssn_mask_front/back/pad: [TILE, TILE] bf16
      norm_gamma:              [TILE, head_dim] bf16
      kv_cache:                 [1, 1, T_pad, head_dim] bf16
      shift_P:                 [ratio_pad, ratio_pad] bf16
    Outputs:
      kv_normed:               [1, 1, head_dim] bf16
      (state buffers updated in place)
    """
    @ttl.operation(grid="auto")
    def lk_d_comp_kernel(
        x, wkv_w, wgate_w, ape_padded,
        cos_compressor, sin_compressor, start_pos,
        state_slot, emit_slot,
        kv_state_front_2d, kv_state_back_2d,
        score_state_front_2d, score_state_back_2d,
        cssn_mask_front, cssn_mask_back, cssn_mask_pad,
        norm_gamma, scaler,
        kv_cache, shift_P,
        kv_normed_out,
    ):
        @ttl.compute()
        def compute():
            pass

    return lk_d_comp_kernel


def reference(mesh,
              x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt,
              cos_compressor_tt, sin_compressor_tt, start_pos_tt,
              state_slot_tt, emit_slot_tt,
              kv_state_front_2d_tt, kv_state_back_2d_tt,
              score_state_front_2d_tt, score_state_back_2d_tt,
              kv_state_front_4d_tt, kv_state_back_4d_tt,
              score_state_front_4d_tt, score_state_back_4d_tt,
              mf_tt, mb_tt, mp_tt,
              norm_gamma_tt, scaler_tt,
              kv_cache_tt, shift_P_tt,
              cssn_out_tt,
              kv_state_front_out_2d_tt, kv_state_back_out_2d_tt,
              score_state_front_out_2d_tt, score_state_back_out_2d_tt,
              sharded_input_memcfg):
    d = HEAD_DIM
    c = CDIM
    rd = ROPE_HEAD_DIM
    rd_half = rd // 2

    # --- non-emit body: matmuls + APE + 4 paged_update_cache calls ---
    kv_tt = ttnn.matmul(x_tt, wkv_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    score_tt = ttnn.matmul(x_tt, wgate_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ape_slot = ttnn.embedding(start_pos_tt, ape_padded_tt, layout=ttnn.TILE_LAYOUT)
    score_tt = ttnn.add(score_tt, ttnn.reshape(ape_slot, [1, 1, c]))

    kv_front = ttnn.slice(kv_tt, [0, 0, 0], [B, 1, d])
    kv_back = ttnn.slice(kv_tt, [0, 0, d], [B, 1, c])
    score_front = ttnn.slice(score_tt, [0, 0, 0], [B, 1, d])
    score_back = ttnn.slice(score_tt, [0, 0, d], [B, 1, c])

    def pug(cache_4d_tt, x_3d_tt):
        x_4d = ttnn.reshape(x_3d_tt, [1, B, 1, d])
        x_sharded = ttnn.to_memory_config(x_4d, memory_config=sharded_input_memcfg)
        ttnn.experimental.paged_update_cache(
            cache_4d_tt, x_sharded, update_idxs_tensor=state_slot_tt)

    pug(kv_state_front_4d_tt, kv_front)
    pug(kv_state_back_4d_tt, kv_back)
    pug(score_state_front_4d_tt, score_front)
    pug(score_state_back_4d_tt, score_back)

    # --- emit body ---
    cssn = _get_ttl_compressor_softmax_sum_norm_kernel(RATIO, RATIO_PAD, d, NORM_EPS)
    slot_shift = _get_ttl_compressor_slot_shift_kernel(1, RATIO_PAD, d)

    cssn(
        kv_state_front_2d_tt, kv_state_back_2d_tt,
        score_state_front_2d_tt, score_state_back_2d_tt,
        mf_tt, mb_tt, mp_tt,
        norm_gamma_tt, scaler_tt, cssn_out_tt,
    )
    kv_2d = ttnn.slice(cssn_out_tt, [0, 0], [B, d])
    kv_normed = ttnn.reshape(kv_2d, [B, 1, d])

    cos = ttnn.embedding(start_pos_tt, cos_compressor_tt, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(start_pos_tt, sin_compressor_tt, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, 1, rd_half])
    sin = ttnn.reshape(sin, [1, 1, rd_half])
    kv_nope = ttnn.slice(kv_normed, [0, 0, 0], [B, 1, d - rd])
    kv_rope = ttnn.slice(kv_normed, [0, 0, d - rd], [B, 1, d])
    kv_rope = _device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)
    kv_normed = ttnn.concat([kv_nope, kv_rope], dim=-1)

    # rotate=False for the attn-side compressor (no Walsh-Hadamard).

    kv_4d = ttnn.reshape(kv_normed, [1, B, 1, d])
    kv_sharded = ttnn.to_memory_config(kv_4d, memory_config=sharded_input_memcfg)
    ttnn.experimental.paged_update_cache(
        kv_cache_tt, kv_sharded, update_idxs_tensor=emit_slot_tt)

    slot_shift(kv_state_front_2d_tt, shift_P_tt, kv_state_front_out_2d_tt)
    ttnn.copy(kv_state_front_out_2d_tt, kv_state_front_2d_tt)
    slot_shift(kv_state_back_2d_tt, shift_P_tt, kv_state_back_out_2d_tt)
    ttnn.copy(kv_state_back_out_2d_tt, kv_state_back_2d_tt)
    slot_shift(score_state_front_2d_tt, shift_P_tt, score_state_front_out_2d_tt)
    ttnn.copy(score_state_front_out_2d_tt, score_state_front_2d_tt)
    slot_shift(score_state_back_2d_tt, shift_P_tt, score_state_back_out_2d_tt)
    ttnn.copy(score_state_back_out_2d_tt, score_state_back_2d_tt)

    return kv_normed


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        x = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        wkv_w = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        wgate_w = torch.randn(DIM, CDIM, dtype=torch.bfloat16) * 0.02
        ape_padded = torch.randn(MAX_SEQ_LEN, CDIM, dtype=torch.bfloat16) * 0.05
        cos_compressor = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        sin_compressor = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        kv_state_front = torch.randn(B, 1, RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_state_back = torch.randn(B, 1, RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_front = torch.randn(B, 1, RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_back = torch.randn(B, 1, RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_front[:, :, 2 * RATIO:, :] = float("-inf")
        score_state_back[:, :, 2 * RATIO:, :] = float("-inf")
        mf, mb, mp = _compressor_softmax_sum_norm_masks(RATIO)
        gamma = torch.ones(_RMS_TILE, HEAD_DIM, dtype=torch.bfloat16)
        scaler = torch.ones(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16)
        kv_cache_init = torch.zeros(1, 1, T_PAD, HEAD_DIM, dtype=torch.bfloat16)
        shift_P = _compressor_shift_matrix(RATIO, RATIO_PAD)
        start_pos = torch.tensor([[RATIO - 1]], dtype=torch.int32)
        state_slot = torch.tensor([RATIO + (RATIO - 1) % RATIO], dtype=torch.int32)
        emit_slot = torch.tensor([0], dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def up(t, dt=ttnn.bfloat16):
            return ttnn.as_tensor(t.contiguous(), dtype=dt, **rep)

        x_tt = up(x)
        wkv_w_tt = up(wkv_w)
        wgate_w_tt = up(wgate_w)
        ape_padded_tt = up(ape_padded)
        cos_compressor_tt = up(cos_compressor)
        sin_compressor_tt = up(sin_compressor)

        # 4D state buffers (for paged_update_cache) and 2D views (for the
        # cssn/slot_shift kernels).
        kv_state_front_4d_tt = up(kv_state_front)
        kv_state_back_4d_tt = up(kv_state_back)
        score_state_front_4d_tt = up(score_state_front)
        score_state_back_4d_tt = up(score_state_back)
        kv_state_front_2d_tt = ttnn.reshape(kv_state_front_4d_tt, [RATIO_PAD, HEAD_DIM])
        kv_state_back_2d_tt = ttnn.reshape(kv_state_back_4d_tt, [RATIO_PAD, HEAD_DIM])
        score_state_front_2d_tt = ttnn.reshape(score_state_front_4d_tt, [RATIO_PAD, HEAD_DIM])
        score_state_back_2d_tt = ttnn.reshape(score_state_back_4d_tt, [RATIO_PAD, HEAD_DIM])

        mf_tt = up(mf)
        mb_tt = up(mb)
        mp_tt = up(mp)
        gamma_tt = up(gamma)
        scaler_tt = up(scaler)
        kv_cache_tt = up(kv_cache_init)
        shift_P_tt = up(shift_P)
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
        emit_slot_tt = ttnn.from_torch(
            emit_slot, device=mesh, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        cssn_out_tt = ttnn.from_torch(
            torch.zeros(_RMS_TILE, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        zero_pad = torch.zeros(RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16)
        ninf_pad = torch.full_like(zero_pad, float("-inf"))
        kv_state_front_out_2d_tt = up(zero_pad)
        kv_state_back_out_2d_tt = up(zero_pad)
        score_state_front_out_2d_tt = up(ninf_pad)
        score_state_back_out_2d_tt = up(ninf_pad)

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

        ref_kv_normed_tt = reference(
            mesh, x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt,
            cos_compressor_tt, sin_compressor_tt, start_pos_tt,
            state_slot_tt, emit_slot_tt,
            kv_state_front_2d_tt, kv_state_back_2d_tt,
            score_state_front_2d_tt, score_state_back_2d_tt,
            kv_state_front_4d_tt, kv_state_back_4d_tt,
            score_state_front_4d_tt, score_state_back_4d_tt,
            mf_tt, mb_tt, mp_tt, gamma_tt, scaler_tt,
            kv_cache_tt, shift_P_tt,
            cssn_out_tt,
            kv_state_front_out_2d_tt, kv_state_back_out_2d_tt,
            score_state_front_out_2d_tt, score_state_back_out_2d_tt,
            sharded_memcfg)
        ref_host = download_chip0(mesh, mesh_shape, ref_kv_normed_tt)

        kernel = make_lk_d_comp_kernel()
        kv_normed_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(
            x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt,
            cos_compressor_tt, sin_compressor_tt, start_pos_tt,
            state_slot_tt, emit_slot_tt,
            kv_state_front_2d_tt, kv_state_back_2d_tt,
            score_state_front_2d_tt, score_state_back_2d_tt,
            mf_tt, mb_tt, mp_tt, gamma_tt, scaler_tt,
            kv_cache_tt, shift_P_tt,
            kv_normed_out_tt,
        )
        kernel_host = download_chip0(mesh, mesh_shape, kv_normed_out_tt)

        ok = report_pcc("Lk-D-comp", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
