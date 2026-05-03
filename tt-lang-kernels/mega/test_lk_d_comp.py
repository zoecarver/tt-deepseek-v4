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

Inlined tt-lang kernels:
  - SUMMA matmul (reused for wkv and wgate at M=TILE, K=4096, N=1024)
  - cssn (compressor_softmax_sum_norm) at d=512, ratio=4
  - slot_shift at d=512, ratio_pad=32
  - rotary swap-SUMMA + rotary-combine on the rope half (Lk-C/Lk-D1
    pattern; cos/sin tables are pre-replicated across TILE rows)

ttnn glue (TODO: mega): embedding for APE/cos/sin (depends on a device
uint32 index, no tt-lang gather primitive) and the four
paged_update_cache writes to state buffers + the kv_cache emit. The
rotary math itself is now in tt-lang.
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
    _compressor_softmax_sum_norm_masks,
    _compressor_shift_matrix,
    _RMS_TILE,
)


DIM = 4096
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
COFF = 2
CDIM = COFF * HEAD_DIM             # 1024
RATIO = 4
RATIO_PAD = _RMS_TILE              # 32
T_PAD = 128
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B = 1
TILE = _RMS_TILE


# TODO: mega fusion blocked: ttnn used for APE embedding+add and the 5
# paged_update_cache writes (4 state buffers + 1 kv_cache emit).
# Lowering them needs ttl.copy_indexed (runtime-indexed read of APE
# table, runtime write to state buffers / kv_cache). The rotary itself
# is now lowered (see _make_rotary_combine_kernel + swap-SUMMA below).


def _make_cssn_kernel(ratio: int, ratio_pad: int, d: int, rms_eps: float):
    """Inlined compressor_softmax_sum_norm kernel.

    Same body as the d=128 indexer cssn (see test_lk_d_idx_emit.py) but
    n_tiles = d // TILE scales with head_dim.
    """
    if ratio_pad != _RMS_TILE:
        raise ValueError(f"ratio_pad={ratio_pad} != TILE={_RMS_TILE} unsupported")
    if d % _RMS_TILE != 0:
        raise ValueError(f"d={d} not multiple of TILE={_RMS_TILE}")
    if 2 * ratio > ratio_pad:
        raise ValueError(f"2*ratio={2 * ratio} > ratio_pad={ratio_pad}")

    n_tiles = d // _RMS_TILE
    inv_d = 1.0 / d

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
    def cssn_kernel(kv_front, kv_back, sc_front, sc_back,
                    mask_front, mask_back, mask_pad,
                    gamma, scaler, out):
        kvf_dfb = ttl.make_dataflow_buffer_like(kv_front, shape=(1, 1), block_count=2)
        kvb_dfb = ttl.make_dataflow_buffer_like(kv_back, shape=(1, 1), block_count=2)
        scf_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        scb_dfb = ttl.make_dataflow_buffer_like(sc_back, shape=(1, 1), block_count=2)
        mf_dfb = ttl.make_dataflow_buffer_like(mask_front, shape=(1, 1), block_count=1)
        mb_dfb = ttl.make_dataflow_buffer_like(mask_back, shape=(1, 1), block_count=1)
        mp_dfb = ttl.make_dataflow_buffer_like(mask_pad, shape=(1, 1), block_count=1)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        gamma_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        sv_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        kv_dfb = ttl.make_dataflow_buffer_like(kv_front, shape=(1, 1), block_count=2)
        max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        max_bc_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        exp_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        invsum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        invsum_bc_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        sm_dfb = ttl.make_dataflow_buffer_like(sc_front, shape=(1, 1), block_count=2)
        weighted_dfb = ttl.make_dataflow_buffer_like(kv_front, shape=(1, 1), block_count=2)
        kv_sum_partial_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        kv_sum_stash_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=n_tiles)
        ks_sq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        ssq_step_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        ssq_acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rms_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rms_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            sc = sc_dfb.wait()
            mf = mf_dfb.wait()
            mb = mb_dfb.wait()
            mp = mp_dfb.wait()

            for ct in range(n_tiles):
                kvf = kvf_dfb.wait()
                kvb = kvb_dfb.wait()
                scf = scf_dfb.wait()
                scb = scb_dfb.wait()

                sv_dfb.reserve().store(mf * scf + mb * scb + mp * scf)
                sv = sv_dfb.wait()

                kv_dfb.reserve().store(mf * kvf + mb * kvb)
                kvv = kv_dfb.wait()

                max_dfb.reserve().store(ttl.math.reduce_max(sv, sc, dims=[0]))
                row_max = max_dfb.wait()
                max_bc = max_bc_dfb.reserve()
                max_bc.store(ttl.math.broadcast(row_max, max_bc, dims=[0]))
                row_max_bc = max_bc_dfb.wait()

                exp_dfb.reserve().store(ttl.math.exp(sv - row_max_bc))
                exp_view = exp_dfb.wait()

                sum_dfb.reserve().store(ttl.math.reduce_sum(exp_view, sc, dims=[0]))
                row_sum = sum_dfb.wait()
                invsum_dfb.reserve().store(ttl.math.recip(row_sum))
                row_invsum = invsum_dfb.wait()
                invsum_bc = invsum_bc_dfb.reserve()
                invsum_bc.store(ttl.math.broadcast(row_invsum, invsum_bc, dims=[0]))
                row_invsum_bc = invsum_bc_dfb.wait()

                sm_dfb.reserve().store(exp_view * row_invsum_bc)
                sm = sm_dfb.wait()

                weighted_dfb.reserve().store(kvv * sm)
                w = weighted_dfb.wait()

                kv_sum_partial_dfb.reserve().store(
                    ttl.math.reduce_sum(w, sc, dims=[0])
                )
                ks = kv_sum_partial_dfb.wait()
                kv_sum_stash_dfb.reserve().store(ks)

                ks_sq_dfb.reserve().store(ks * ks)
                ssq_step_dfb.reserve().store(
                    ttl.math.reduce_sum(ks_sq_dfb.wait(), sc, dims=[0, 1])
                )
                step = ssq_step_dfb.wait()
                if ct == 0:
                    ssq_acc_dfb.reserve().store(step)
                else:
                    prev = ssq_acc_dfb.wait()
                    ssq_acc_dfb.reserve().store(prev + step)

            ssq = ssq_acc_dfb.wait()
            rms_dfb.reserve().store(
                ttl.math.rsqrt(
                    ssq * ttl.math.fill(ssq, inv_d) + ttl.math.fill(ssq, rms_eps)
                )
            )
            rms_scalar = rms_dfb.wait()
            rms_bc = rms_bc_dfb.reserve()
            rms_bc.store(ttl.math.broadcast(rms_scalar, rms_bc, dims=[0, 1]))
            rms_full = rms_bc_dfb.wait()

            for ct in range(n_tiles):
                ks = kv_sum_stash_dfb.wait()
                g = gamma_dfb.wait()
                out_dfb.reserve().store(ks * g * rms_full)

        @ttl.datamovement()
        def dm_read():
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            ttl.copy(mask_front[0, 0], mf_dfb.reserve()).wait()
            ttl.copy(mask_back[0, 0], mb_dfb.reserve()).wait()
            ttl.copy(mask_pad[0, 0], mp_dfb.reserve()).wait()

            for ct in range(n_tiles):
                ttl.copy(kv_front[0, ct], kvf_dfb.reserve()).wait()
                ttl.copy(kv_back[0, ct], kvb_dfb.reserve()).wait()
                ttl.copy(sc_front[0, ct], scf_dfb.reserve()).wait()
                ttl.copy(sc_back[0, ct], scb_dfb.reserve()).wait()

            for ct in range(n_tiles):
                ttl.copy(gamma[0, ct], gamma_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            for ct in range(n_tiles):
                ttl.copy(out_dfb.wait(), out[0, ct]).wait()

    return cssn_kernel


def _make_slot_shift_kernel(num_buffers: int, ratio_pad: int, d: int):
    """Inlined compressor_slot_shift kernel.

    out = P @ buf where P is the within-tile shift matrix.
    """
    if ratio_pad != _RMS_TILE:
        raise ValueError(f"ratio_pad={ratio_pad} != TILE={_RMS_TILE} unsupported")
    if d % _RMS_TILE != 0:
        raise ValueError(f"d={d} not multiple of TILE={_RMS_TILE}")

    M_tiles = num_buffers
    N_tiles = d // _RMS_TILE
    total_work = M_tiles * N_tiles

    @ttl.operation(grid="auto")
    def slot_shift_kernel(buf, P, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        buf_dfb = ttl.make_dataflow_buffer_like(buf, shape=(1, 1), block_count=2)
        P_dfb = ttl.make_dataflow_buffer_like(P, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            P_tile = P_dfb.wait()

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    buf_tile = buf_dfb.wait()
                    out_dfb.reserve().store(P_tile @ buf_tile)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(P[0, 0], P_dfb.reserve()).wait()
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    m = global_w // N_tiles
                    n = global_w % N_tiles
                    ttl.copy(buf[m, n], buf_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    m = global_w // N_tiles
                    n = global_w % N_tiles
                    ttl.copy(out_dfb.wait(), out[m, n]).wait()

    return slot_shift_kernel


def _make_summa_matmul_kernel(M: int, K: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
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


def _make_rotary_combine_kernel(num_row_tiles: int, num_h_tiles: int):
    """out = x * cos + x_swap * sin; cos/sin tile-replicated."""

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
    def rotary_combine(x, x_swap, cos, sin, out):
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        xs_dfb = ttl.make_dataflow_buffer_like(x_swap, shape=(1, 1), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(cos, shape=(1, 1), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(sin, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(num_row_tiles):
                for _ in range(num_h_tiles):
                    xt = x_dfb.wait()
                    xst = xs_dfb.wait()
                    ct = c_dfb.wait()
                    st = s_dfb.wait()
                    out_dfb.reserve().store(xt * ct + xst * st)

        @ttl.datamovement()
        def dm_read():
            for t in range(num_row_tiles):
                for h in range(num_h_tiles):
                    ttl.copy(x[t, h], x_dfb.reserve()).wait()
                    ttl.copy(x_swap[t, h], xs_dfb.reserve()).wait()
                    ttl.copy(cos[0, h], c_dfb.reserve()).wait()
                    ttl.copy(sin[0, h], s_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            for t in range(num_row_tiles):
                for h in range(num_h_tiles):
                    ttl.copy(out_dfb.wait(), out[t, h]).wait()

    return rotary_combine


def _build_rotary_tables(cos_full_cpu: torch.Tensor, sin_full_cpu: torch.Tensor,
                         inverse: bool):
    max_seq_len, rd_half = cos_full_cpu.shape
    rd = 2 * rd_half
    if rd % TILE != 0:
        raise ValueError(f"rd={rd} not multiple of TILE={TILE}")
    cos_extended = cos_full_cpu.repeat_interleave(2, dim=-1)
    sign = torch.ones(rd, dtype=cos_full_cpu.dtype)
    if inverse:
        sign[1::2] = -1
    else:
        sign[0::2] = -1
    sin_signed = sin_full_cpu.repeat_interleave(2, dim=-1) * sign
    cos_extended_packed = cos_extended.unsqueeze(1).expand(
        max_seq_len, TILE, rd).reshape(max_seq_len, TILE * rd).contiguous()
    sin_signed_packed = sin_signed.unsqueeze(1).expand(
        max_seq_len, TILE, rd).reshape(max_seq_len, TILE * rd).contiguous()
    return cos_extended_packed, sin_signed_packed


def _build_swap_matrix(rd: int) -> torch.Tensor:
    if rd % 2:
        raise ValueError(f"rd={rd} must be even")
    P = torch.zeros(rd, rd, dtype=torch.bfloat16)
    for k in range(rd // 2):
        P[2 * k, 2 * k + 1] = 1.0
        P[2 * k + 1, 2 * k] = 1.0
    return P


def make_lk_d_comp_kernel(mesh, cos_compressor_cpu, sin_compressor_cpu,
                          sharded_input_memcfg):
    """Mega kernel for Lk-D-comp (attn-side compressor, emit step).

    Pipeline (rotate=False):
      Pad x to [TILE, dim].
      SUMMA wkv: x @ wkv -> kv [TILE, CDIM]
      SUMMA wgate: x @ wgate -> score [TILE, CDIM]
      APE add (TODO: mega).
      slice/reshape -> kv_front, kv_back, score_front, score_back [B, 1, d]
      4x paged_update_cache to state_front/back (TODO: mega).
      cssn over (now updated) front/back state buffers -> kv_normed [TILE, d]
      slice nope/rope, swap-SUMMA + rotary-combine on rope, concat.
      paged_update_cache to kv_cache @ emit_slot (TODO: mega).
      4x slot_shift + 4x ttnn.copy.
    """
    d = HEAD_DIM
    c = CDIM
    rd = ROPE_HEAD_DIM

    # SUMMA: M=TILE, K=4096, N=1024. Mt=1, Kt=128, Nt=32.
    # block=(1, 4, 4) part=(1, 8, 1) -> 8 cores (Nb=8, Mb=1, Kb=32).
    matmul_kernel = _make_summa_matmul_kernel(
        M=TILE, K=DIM, N=CDIM,
        block_cfg=(1, 4, 4), part_cfg=(1, 8, 1))

    cssn_kernel = _make_cssn_kernel(RATIO, RATIO_PAD, d, NORM_EPS)
    slot_shift_kernel = _make_slot_shift_kernel(1, RATIO_PAD, d)

    # Rotary swap SUMMA: M=TILE=32, K=N=rd=64. Mt=1, Kt=Nt=2.
    # block=(1,1,2), part=(1,2,1) -> 2 cores.
    swap_kernel = _make_summa_matmul_kernel(
        M=TILE, K=rd, N=rd,
        block_cfg=(1, 1, 2), part_cfg=(1, 2, 1))
    rotary_combine_kernel = _make_rotary_combine_kernel(
        num_row_tiles=TILE // TILE, num_h_tiles=rd // TILE)

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    cos_ext_packed, sin_signed_packed = _build_rotary_tables(
        cos_compressor_cpu, sin_compressor_cpu, inverse=False)
    cos_ext_tt = ttnn.as_tensor(cos_ext_packed, dtype=ttnn.bfloat16, **rep)
    sin_signed_tt = ttnn.as_tensor(sin_signed_packed, dtype=ttnn.bfloat16, **rep)
    P_cpu = _build_swap_matrix(rd)
    P_tt = ttnn.as_tensor(P_cpu.contiguous(), dtype=ttnn.bfloat16, **rep)

    state: dict = {}

    def lk_d_comp_kernel(
        x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt,
        cos_compressor_tt, sin_compressor_tt, start_pos_tt,
        state_slot_tt, emit_slot_tt,
        kv_state_front_2d_tt, kv_state_back_2d_tt,
        score_state_front_2d_tt, score_state_back_2d_tt,
        kv_state_front_4d_tt, kv_state_back_4d_tt,
        score_state_front_4d_tt, score_state_back_4d_tt,
        cssn_mask_front_tt, cssn_mask_back_tt, cssn_mask_pad_tt,
        norm_gamma_tt, scaler_tt,
        kv_cache_tt, shift_P_tt,
        kv_state_front_scratch_tt, kv_state_back_scratch_tt,
        score_state_front_scratch_tt, score_state_back_scratch_tt,
        cssn_out_tt, kv_normed_out,
    ):
        if "scratch" not in state:
            state["kv_padded"] = ttnn.from_torch(
                torch.zeros(TILE, CDIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["score_padded"] = ttnn.from_torch(
                torch.zeros(TILE, CDIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["rope_swap"] = ttnn.from_torch(
                torch.zeros(TILE, ROPE_HEAD_DIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["rope_rot"] = ttnn.from_torch(
                torch.zeros(TILE, ROPE_HEAD_DIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["scratch"] = True

        x_2d = ttnn.reshape(x_tt, [B, DIM])
        x_padded = ttnn.pad(x_2d, padding=[(0, TILE - B), (0, 0)], value=0.0)

        # SUMMA wkv.
        matmul_kernel(x_padded, wkv_w_tt, state["kv_padded"])
        # SUMMA wgate.
        matmul_kernel(x_padded, wgate_w_tt, state["score_padded"])

        kv_row = ttnn.slice(state["kv_padded"], [0, 0], [B, c])
        kv_3d = ttnn.reshape(kv_row, [B, 1, c])
        score_row = ttnn.slice(state["score_padded"], [0, 0], [B, c])
        score_3d = ttnn.reshape(score_row, [B, 1, c])

        # APE add (TODO: mega).
        ape_slot = ttnn.embedding(start_pos_tt, ape_padded_tt, layout=ttnn.TILE_LAYOUT)
        score_3d = ttnn.add(score_3d, ttnn.reshape(ape_slot, [1, 1, c]))

        # 4x paged_update_cache to state buffers (TODO: mega).
        kv_front = ttnn.slice(kv_3d, [0, 0, 0], [B, 1, d])
        kv_back = ttnn.slice(kv_3d, [0, 0, d], [B, 1, c])
        score_front = ttnn.slice(score_3d, [0, 0, 0], [B, 1, d])
        score_back = ttnn.slice(score_3d, [0, 0, d], [B, 1, c])

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
        cssn_kernel(
            kv_state_front_2d_tt, kv_state_back_2d_tt,
            score_state_front_2d_tt, score_state_back_2d_tt,
            cssn_mask_front_tt, cssn_mask_back_tt, cssn_mask_pad_tt,
            norm_gamma_tt, scaler_tt, cssn_out_tt,
        )
        # rotary on rope half via swap-SUMMA + rotary-combine.
        # Slice rope tiles directly off cssn_out_tt (TILE rows: row 0 valid,
        # others zero from gamma=0). Multiplying zeros by cos/sin still
        # gives zeros so the kernel-side TILE-row layout is fine.
        # TODO: mega fusion blocked: ttnn used for embedding(start_pos, ...).
        cos_b_2d = ttnn.embedding(start_pos_tt, cos_ext_tt, layout=ttnn.TILE_LAYOUT)
        sin_b_2d = ttnn.embedding(start_pos_tt, sin_signed_tt, layout=ttnn.TILE_LAYOUT)
        cos_b = ttnn.reshape(cos_b_2d, [TILE, rd])
        sin_b = ttnn.reshape(sin_b_2d, [TILE, rd])
        cssn_nope_2d = ttnn.slice(cssn_out_tt, [0, 0], [TILE, d - rd])
        cssn_rope_2d = ttnn.slice(cssn_out_tt, [0, d - rd], [TILE, d])
        swap_kernel(cssn_rope_2d, P_tt, state["rope_swap"])
        rotary_combine_kernel(
            cssn_rope_2d, state["rope_swap"], cos_b, sin_b, state["rope_rot"])
        kv_full_2d = ttnn.concat([cssn_nope_2d, state["rope_rot"]], dim=-1)
        kv_2d = ttnn.slice(kv_full_2d, [0, 0], [B, d])
        kv_normed = ttnn.reshape(kv_2d, [B, 1, d])

        # paged_update_cache to kv_cache (TODO: mega).
        kv_4d = ttnn.reshape(kv_normed, [1, B, 1, d])
        kv_sharded = ttnn.to_memory_config(kv_4d, memory_config=sharded_input_memcfg)
        ttnn.experimental.paged_update_cache(
            kv_cache_tt, kv_sharded, update_idxs_tensor=emit_slot_tt)

        # 4x slot_shift + 4x ttnn.copy.
        slot_shift_kernel(kv_state_front_2d_tt, shift_P_tt, kv_state_front_scratch_tt)
        ttnn.copy(kv_state_front_scratch_tt, kv_state_front_2d_tt)
        slot_shift_kernel(kv_state_back_2d_tt, shift_P_tt, kv_state_back_scratch_tt)
        ttnn.copy(kv_state_back_scratch_tt, kv_state_back_2d_tt)
        slot_shift_kernel(score_state_front_2d_tt, shift_P_tt, score_state_front_scratch_tt)
        ttnn.copy(score_state_front_scratch_tt, score_state_front_2d_tt)
        slot_shift_kernel(score_state_back_2d_tt, shift_P_tt, score_state_back_scratch_tt)
        ttnn.copy(score_state_back_scratch_tt, score_state_back_2d_tt)

        ttnn.copy(kv_normed, kv_normed_out)

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


def _build_inputs(mesh, rep):
    """Generate torch inputs + upload helpers, returning every tensor the
    caller needs. Kept out of main() so we can re-upload between ref and
    kernel runs without code duplication.
    """
    return None  # placeholder; main() builds inline below.


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

        def fresh_state_uploads():
            kv_sf_4d = up(kv_state_front)
            kv_sb_4d = up(kv_state_back)
            sc_sf_4d = up(score_state_front)
            sc_sb_4d = up(score_state_back)
            kv_sf_2d = ttnn.reshape(kv_sf_4d, [RATIO_PAD, HEAD_DIM])
            kv_sb_2d = ttnn.reshape(kv_sb_4d, [RATIO_PAD, HEAD_DIM])
            sc_sf_2d = ttnn.reshape(sc_sf_4d, [RATIO_PAD, HEAD_DIM])
            sc_sb_2d = ttnn.reshape(sc_sb_4d, [RATIO_PAD, HEAD_DIM])
            return (kv_sf_4d, kv_sb_4d, sc_sf_4d, sc_sb_4d,
                    kv_sf_2d, kv_sb_2d, sc_sf_2d, sc_sb_2d)

        x_tt = up(x)
        wkv_w_tt = up(wkv_w)
        wgate_w_tt = up(wgate_w)
        ape_padded_tt = up(ape_padded)
        cos_compressor_tt = up(cos_compressor)
        sin_compressor_tt = up(sin_compressor)
        mf_tt = up(mf)
        mb_tt = up(mb)
        mp_tt = up(mp)
        gamma_tt = up(gamma)
        scaler_tt = up(scaler)
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

        zero_pad = torch.zeros(RATIO_PAD, HEAD_DIM, dtype=torch.bfloat16)
        ninf_pad = torch.full_like(zero_pad, float("-inf"))

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

        # ----- reference run -----
        (kv_sf_4d, kv_sb_4d, sc_sf_4d, sc_sb_4d,
         kv_sf_2d, kv_sb_2d, sc_sf_2d, sc_sb_2d) = fresh_state_uploads()
        kv_cache_tt = up(kv_cache_init)
        cssn_out_tt = up(torch.zeros(_RMS_TILE, HEAD_DIM, dtype=torch.bfloat16))
        kv_sf_scratch = up(zero_pad)
        kv_sb_scratch = up(zero_pad)
        sc_sf_scratch = up(ninf_pad)
        sc_sb_scratch = up(ninf_pad)

        ref_kv_normed_tt = reference(
            mesh, x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt,
            cos_compressor_tt, sin_compressor_tt, start_pos_tt,
            state_slot_tt, emit_slot_tt,
            kv_sf_2d, kv_sb_2d, sc_sf_2d, sc_sb_2d,
            kv_sf_4d, kv_sb_4d, sc_sf_4d, sc_sb_4d,
            mf_tt, mb_tt, mp_tt, gamma_tt, scaler_tt,
            kv_cache_tt, shift_P_tt,
            cssn_out_tt,
            kv_sf_scratch, kv_sb_scratch, sc_sf_scratch, sc_sb_scratch,
            sharded_memcfg)
        ref_host = download_chip0(mesh, mesh_shape, ref_kv_normed_tt)

        # ----- kernel run (fresh state) -----
        (kv_sf_4d2, kv_sb_4d2, sc_sf_4d2, sc_sb_4d2,
         kv_sf_2d2, kv_sb_2d2, sc_sf_2d2, sc_sb_2d2) = fresh_state_uploads()
        kv_cache_tt2 = up(kv_cache_init)
        cssn_out_tt2 = up(torch.zeros(_RMS_TILE, HEAD_DIM, dtype=torch.bfloat16))
        kv_sf_scratch2 = up(zero_pad)
        kv_sb_scratch2 = up(zero_pad)
        sc_sf_scratch2 = up(ninf_pad)
        sc_sb_scratch2 = up(ninf_pad)

        kernel = make_lk_d_comp_kernel(
            mesh, cos_compressor, sin_compressor, sharded_memcfg)
        kv_normed_out_tt = up(torch.zeros(1, 1, HEAD_DIM, dtype=torch.bfloat16))
        kernel(
            x_tt, wkv_w_tt, wgate_w_tt, ape_padded_tt,
            cos_compressor_tt, sin_compressor_tt, start_pos_tt,
            state_slot_tt, emit_slot_tt,
            kv_sf_2d2, kv_sb_2d2, sc_sf_2d2, sc_sb_2d2,
            kv_sf_4d2, kv_sb_4d2, sc_sf_4d2, sc_sb_4d2,
            mf_tt, mb_tt, mp_tt, gamma_tt, scaler_tt,
            kv_cache_tt2, shift_P_tt,
            kv_sf_scratch2, kv_sb_scratch2, sc_sf_scratch2, sc_sb_scratch2,
            cssn_out_tt2, kv_normed_out_tt,
        )
        kernel_host = download_chip0(mesh, mesh_shape, kv_normed_out_tt)

        ok = report_pcc("Lk-D-comp", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
