"""Lk-D-idx-emit PCC test: indexer compressor emit branch.

Reference covers `DeviceCompressor._emit_body` (overlap=True, rotate=True
for the indexer's compressor). Specifically:
- compressor_softmax_sum_norm (cssn) over front/back state buffers + masks
- slice/reshape -> kv_normed [1, 1, d]
- pick cos/sin via embedding(start_pos, cos_compressor); reshape
- slice nope/rope, rotary on rope, concat
- _device_rotate_activation (Walsh-Hadamard, rotate=True) -> matmul(x, H)
- paged_update_cache to indexer kv_cache at emit_slot
- 4x compressor_slot_shift kernel + 4x ttnn.copy

Boundaries: pre-CCL is indexer.compressor.wgate all_gather; post-CCL is
indexer.weights_proj all_gather.

This file inlines:
  - cssn kernel factory (_make_cssn_kernel)
  - slot-shift kernel factory (_make_slot_shift_kernel)
  - SUMMA matmul factory used for Walsh-Hadamard rotation
  - rotary swap-SUMMA + rotary-combine (same lowering as Lk-C/Lk-D1)

ttnn glue (with TODO markers) covers embedding(start_pos, ...) for the
cos/sin lookup, slice/reshape/concat around the rotary, and
paged_update_cache to indexer kv_cache.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark

from inference import (
    _device_apply_rotary_interleaved,
    _device_rotate_activation,
    _sylvester_hadamard,
    _get_ttl_compressor_softmax_sum_norm_kernel,
    _get_ttl_compressor_slot_shift_kernel,
    _compressor_softmax_sum_norm_masks,
    _compressor_shift_matrix,
    _RMS_TILE,
)


INDEX_HEAD_DIM = 128
ROPE_HEAD_DIM = 64
RATIO = 4
RATIO_PAD = _RMS_TILE       # 32
T_PAD = 128
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B = 1
TILE = _RMS_TILE


# TODO: mega fusion blocked: ttnn used for embedding(start_pos, cos/sin)
# (depends on a device uint32 index, no tt-lang gather primitive) and for
# paged_update_cache(kv_cache, emit_slot) (runtime-indexed write). The
# rotary math itself is now in tt-lang (see _make_swap_combine_kernel
# below).


def _make_cssn_kernel(ratio: int, ratio_pad: int, d: int, rms_eps: float):
    """Inlined compressor_softmax_sum_norm kernel.

    Fuses slice+concat view + softmax + weighted-sum + RMSNorm for the
    compressor emit branch (overlap=True). One core per d-tile: each
    core does the local softmax+kv_sum+ssq-partial for its tile, partials
    are reduced to col 0 via PipeNet, root finalizes rsqrt and broadcasts
    back, then every core multiplies by its gamma tile and writes.

    Math:
      score_view = mf*sc_front + mb*sc_back + mp*sc_front  # padding rows -inf
      kv_view    = mf*kv_front + mb*kv_back                # padding rows zero
      sm         = softmax(score_view, dim=0)
      kv_sum     = sum(kv_view * sm, dim=0)                # row 0 of [TILE, d]
      rms        = rsqrt(mean(kv_sum^2) + rms_eps)
      out        = kv_sum * gamma * rms                    # row 0 only

    See _compile_compressor_softmax_sum_norm_kernel in inference.py for
    the production source.
    """
    if ratio_pad != _RMS_TILE:
        raise ValueError(f"ratio_pad={ratio_pad} != TILE={_RMS_TILE} unsupported")
    if d % _RMS_TILE != 0:
        raise ValueError(f"d={d} not multiple of TILE={_RMS_TILE}")
    if 2 * ratio > ratio_pad:
        raise ValueError(f"2*ratio={2 * ratio} > ratio_pad={ratio_pad}")

    n_tiles = d // _RMS_TILE
    inv_d = 1.0 / d
    # Lay out cores in a 1D row that fits the device compute grid
    # (sterling BH max X = 11). Each core handles TPC consecutive
    # d-tiles. n_tiles=4 -> 4x1 (TPC=1), n_tiles=16 -> 8x1 (TPC=2).
    COL = min(n_tiles, 8)
    if n_tiles % COL:
        raise ValueError(f"n_tiles={n_tiles} not divisible by COL={COL}")
    TPC = n_tiles // COL
    SSQ_RECV_BC = max(2, COL - 1)

    @ttl.operation(grid=(COL, 1), fp32_dest_acc_en=True)
    def cssn_kernel(kv_front, kv_back, sc_front, sc_back,
                    mask_front, mask_back, mask_pad,
                    gamma, scaler, out):
        ssq_reduce_pipes = [ttl.Pipe(src=(col, 0), dst=(0, 0))
                            for col in range(1, COL)]
        ssq_reduce_net = ttl.PipeNet(ssq_reduce_pipes)
        rms_bcast_pipes = [ttl.Pipe(src=(0, 0), dst=(slice(0, COL), 0))]
        rms_bcast_net = ttl.PipeNet(rms_bcast_pipes)

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
        kv_sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        kv_sum_stash_dfb = ttl.make_dataflow_buffer_like(
            scaler, shape=(1, 1), block_count=max(2, TPC))
        ks_sq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        ssq_step_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        ssq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        ssq_recv_dfb = ttl.make_dataflow_buffer_like(
            scaler, shape=(1, 1), block_count=SSQ_RECV_BC)
        rms_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rms_recv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rms_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            c, _ = ttl.node(dims=2)
            sc = sc_dfb.wait()
            mf = mf_dfb.wait()
            mb = mb_dfb.wait()
            mp = mp_dfb.wait()

            for t in range(TPC):
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

                kv_sum_dfb.reserve().store(
                    ttl.math.reduce_sum(w, sc, dims=[0])
                )
                ks = kv_sum_dfb.wait()
                kv_sum_stash_dfb.reserve().store(ks)

                ks_sq_dfb.reserve().store(ks * ks)
                ssq_step_dfb.reserve().store(
                    ttl.math.reduce_sum(ks_sq_dfb.wait(), sc, dims=[0, 1])
                )
                step = ssq_step_dfb.wait()
                if t == 0:
                    ssq_dfb.reserve().store(step)
                else:
                    prev = ssq_dfb.wait()
                    ssq_dfb.reserve().store(prev + step)

            if c == 0:
                for _ in range(COL - 1):
                    prev = ssq_dfb.wait()
                    r = ssq_recv_dfb.wait()
                    ssq_dfb.reserve().store(prev + r)
                ssq_total = ssq_dfb.wait()
                rms_dfb.reserve().store(
                    ttl.math.rsqrt(
                        ssq_total * ttl.math.fill(ssq_total, inv_d)
                        + ttl.math.fill(ssq_total, rms_eps)
                    )
                )

            rms_scalar = rms_recv_dfb.wait()
            rms_bc = rms_bc_dfb.reserve()
            rms_bc.store(ttl.math.broadcast(rms_scalar, rms_bc, dims=[0, 1]))
            rms_full = rms_bc_dfb.wait()

            for t in range(TPC):
                ks_stashed = kv_sum_stash_dfb.wait()
                g = gamma_dfb.wait()
                out_dfb.reserve().store(ks_stashed * g * rms_full)

        @ttl.datamovement()
        def dm_read():
            c, _ = ttl.node(dims=2)
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            ttl.copy(mask_front[0, 0], mf_dfb.reserve()).wait()
            ttl.copy(mask_back[0, 0], mb_dfb.reserve()).wait()
            ttl.copy(mask_pad[0, 0], mp_dfb.reserve()).wait()

            for t in range(TPC):
                ti = c * TPC + t
                ttl.copy(kv_front[0, ti], kvf_dfb.reserve()).wait()
                ttl.copy(kv_back[0, ti], kvb_dfb.reserve()).wait()
                ttl.copy(sc_front[0, ti], scf_dfb.reserve()).wait()
                ttl.copy(sc_back[0, ti], scb_dfb.reserve()).wait()

            for t in range(TPC):
                ti = c * TPC + t
                ttl.copy(gamma[0, ti], gamma_dfb.reserve()).wait()

            if c == 0:
                def recv_ssq(pipe):
                    blk = ssq_recv_dfb.reserve()
                    ttl.copy(pipe, blk).wait()
                ssq_reduce_net.if_dst(recv_ssq)
            else:
                ssq_local = ssq_dfb.wait()

                def send_ssq(pipe):
                    ttl.copy(ssq_local, pipe).wait()
                ssq_reduce_net.if_src(send_ssq)

            if c == 0:
                rms_local = rms_dfb.wait()

                def send_rms(pipe):
                    ttl.copy(rms_local, pipe).wait()
                rms_bcast_net.if_src(send_rms)

            def recv_rms(pipe):
                blk = rms_recv_dfb.reserve()
                ttl.copy(pipe, blk).wait()
            rms_bcast_net.if_dst(recv_rms)

        @ttl.datamovement()
        def dm_write():
            c, _ = ttl.node(dims=2)
            for t in range(TPC):
                ti = c * TPC + t
                ttl.copy(out_dfb.wait(), out[0, ti]).wait()

    return cssn_kernel


def _make_slot_shift_kernel(num_buffers: int, ratio_pad: int, d: int):
    """Inlined compressor_slot_shift kernel.

    Math: out[i, :] = buf[i + ratio, :] for i < ratio
          out[i, :] = buf[i, :]         for i >= ratio
    Expressed as a tile matmul: out = P @ buf where P is the [ratio_pad,
    ratio_pad] shift matrix. ratio_pad must equal TILE (32) so P fits in
    one tile and the in-tile shift is a single matmul.

    See _compile_compressor_slot_shift_kernel in inference.py for the
    production source.
    """
    if ratio_pad != _RMS_TILE:
        raise ValueError(f"ratio_pad={ratio_pad} != TILE={_RMS_TILE} unsupported")
    if d % _RMS_TILE != 0:
        raise ValueError(f"d={d} not multiple of TILE={_RMS_TILE}")

    M_tiles = num_buffers
    N_tiles = d // _RMS_TILE
    total_work = M_tiles * N_tiles

    # Grid sized to total_work (one tile per core; P is broadcast). For
    # d=128, num_buffers=1 this is grid=(4, 1) = 4 cores, one tile per
    # core. For larger workloads grid_rows scales up.
    grid_cols = min(total_work, 8)
    grid_rows = -(-total_work // grid_cols)

    @ttl.operation(grid=(grid_cols, grid_rows))
    def slot_shift_kernel(buf, P, out):
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


def _make_swap_combine_kernel(M: int, K_dim: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    """SUMMA matmul fused with rotary combine.
      out[m, n] = x[m, n] * cos[0, n] + (x @ P)[m, n] * sin[0, n]
    cos, sin shape: [TILE, N] (only row 0 is consumed; downstream broadcasts).
    P shape: [K_dim, N] swap-pairs constant.
    """
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Kp != 1:
        raise ValueError(f"K_parts must be 1, got {Kp}")
    Mt, Nt, Kt = M // TILE, N // TILE, K_dim // TILE
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
    def swap_combine(x, P, cos, sin, out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p))
                   for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp)))
                   for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(x, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(P, shape=(bk, bn), block_count=2)
        x_diag_cb = ttl.make_dataflow_buffer_like(
            x, shape=(bm, bn), block_count=2)
        cos_cb = ttl.make_dataflow_buffer_like(
            cos, shape=(1, bn), block_count=2)
        sin_cb = ttl.make_dataflow_buffer_like(
            sin, shape=(1, bn), block_count=2)
        swap_cb = ttl.make_dataflow_buffer_like(
            out, shape=(bm, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(
            out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = swap_cb.reserve()
                    for _ in range(Kb):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        p += a_blk @ b_blk
                    s = swap_cb.wait()
                    xd = x_diag_cb.wait()
                    c = cos_cb.wait()
                    si = sin_cb.wait()
                    out_cb.reserve().store(xd * c + s * si)

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
                            ttl.copy(x[mr:mr + bm, kc:kc + bk], a_blk).wait()
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
                            ttl.copy(P[kc:kc + bk, nc:nc + bn], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    ttl.copy(x[mr:mr + bm, nc:nc + bn],
                             x_diag_cb.reserve()).wait()
                    ttl.copy(cos[0:1, nc:nc + bn],
                             cos_cb.reserve()).wait()
                    ttl.copy(sin[0:1, nc:nc + bn],
                             sin_cb.reserve()).wait()
                    ttl.copy(out_cb.wait(),
                             out[mr:mr + bm, nc:nc + bn]).wait()

    return swap_combine


def _build_rotary_tables(cos_full_cpu: torch.Tensor, sin_full_cpu: torch.Tensor,
                         inverse: bool):
    """Pre-replicated cos/sin tables; see Lk-C for the shape rationale."""
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


def make_lk_d_idx_emit_kernel(mesh, cos_compressor_cpu, sin_compressor_cpu,
                              sharded_input_memcfg):
    """Mega kernel for Lk-D-idx-emit.

    Pipeline:
      cssn -> [TILE, d] (row 0 valid)
      reshape -> [1, TILE, d]
      slice nope/rope, swap-SUMMA + rotary-combine on rope, concat
      reshape -> [TILE, d]
      SUMMA Walsh-Hadamard matmul -> [TILE, d]
      slice/reshape -> kv_normed [1, 1, d]
      paged_update_cache(kv_cache, emit_slot) (TODO: mega - ttnn glue)
      4x slot_shift + 4x ttnn.copy
    """
    d = INDEX_HEAD_DIM
    rd = ROPE_HEAD_DIM

    cssn_kernel = _make_cssn_kernel(RATIO, RATIO_PAD, d, NORM_EPS)
    slot_shift_kernel = _make_slot_shift_kernel(1, RATIO_PAD, d)

    # Walsh-Hadamard SUMMA: M=TILE=32 (cssn out shape), K=d=128, N=d=128.
    # Mt=1, Kt=4, Nt=4. block=(1, 2, 4) part=(1, 2, 1) -> 2 cores, M_BPN=N_BPN=1.
    matmul_hada_kernel = _make_summa_matmul_kernel(
        M=TILE, K=d, N=d,
        block_cfg=(1, 2, 4), part_cfg=(1, 2, 1))

    # Fused swap SUMMA + rotary combine: M=TILE=32, K=N=rd=64.
    # Mt=1, Kt=Nt=2. block=(1,1,2), part=(1,2,1) -> 2 cores.
    swap_combine_kernel = _make_swap_combine_kernel(
        M=TILE, K_dim=rd, N=rd,
        block_cfg=(1, 1, 2), part_cfg=(1, 2, 1))

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

    def lk_d_idx_emit_kernel(
        kv_state_front_2d, kv_state_back_2d,
        score_state_front_2d, score_state_back_2d,
        cssn_mask_front, cssn_mask_back, cssn_mask_pad,
        norm_gamma, scaler,
        cos_compressor, sin_compressor, start_pos,
        H, kv_cache, emit_slot, shift_P,
        kv_state_front_scratch, kv_state_back_scratch,
        score_state_front_scratch, score_state_back_scratch,
        kv_normed_out,
    ):
        if "scratch" not in state:
            state["cssn_out"] = ttnn.from_torch(
                torch.zeros(TILE, d, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["hada_out"] = ttnn.from_torch(
                torch.zeros(TILE, d, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["rope_rot"] = ttnn.from_torch(
                torch.zeros(TILE, rd, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["scratch"] = True

        # cssn fused softmax+sum+norm.
        cssn_kernel(
            kv_state_front_2d, kv_state_back_2d,
            score_state_front_2d, score_state_back_2d,
            cssn_mask_front, cssn_mask_back, cssn_mask_pad,
            norm_gamma, scaler, state["cssn_out"],
        )

        # Reshape to 3D for rotary path. Operate on the full TILE-padded
        # layout; only row 0 is valid, others are zero from cssn (gamma * 0).
        kv_3d = ttnn.reshape(state["cssn_out"], [1, TILE, d])

        # Rotary on the rope half via swap-SUMMA + rotary-combine.
        # TODO: mega fusion blocked: ttnn used for embedding(start_pos, ...).
        cos_b_2d = ttnn.embedding(start_pos, cos_ext_tt, layout=ttnn.TILE_LAYOUT)
        sin_b_2d = ttnn.embedding(start_pos, sin_signed_tt, layout=ttnn.TILE_LAYOUT)
        cos_b = ttnn.reshape(cos_b_2d, [TILE, rd])
        sin_b = ttnn.reshape(sin_b_2d, [TILE, rd])
        kv_nope = ttnn.slice(kv_3d, [0, 0, 0], [1, TILE, d - rd])
        kv_rope = ttnn.slice(kv_3d, [0, 0, d - rd], [1, TILE, d])
        kv_rope_2d = ttnn.reshape(kv_rope, [TILE, rd])
        swap_combine_kernel(
            kv_rope_2d, P_tt, cos_b, sin_b, state["rope_rot"])
        kv_rope_rot_3d = ttnn.reshape(state["rope_rot"], [1, TILE, rd])
        kv_3d = ttnn.concat([kv_nope, kv_rope_rot_3d], dim=-1)

        # Walsh-Hadamard rotation via inline SUMMA matmul.
        kv_2d = ttnn.reshape(kv_3d, [TILE, d])
        matmul_hada_kernel(kv_2d, H, state["hada_out"])

        # Take row 0 (the valid output), reshape to [1, 1, d].
        kv_row = ttnn.slice(state["hada_out"], [0, 0], [B, d])
        kv_normed = ttnn.reshape(kv_row, [B, 1, d])

        # paged_update_cache to indexer kv_cache.
        # TODO: mega fusion blocked (bucket #1 — unwired): element_write
        # available; lower this slot-write into the fused kernel. C10.
        kv_4d = ttnn.reshape(kv_normed, [1, B, 1, d])
        kv_sharded = ttnn.to_memory_config(kv_4d, memory_config=sharded_input_memcfg)
        ttnn.experimental.paged_update_cache(
            kv_cache, kv_sharded, update_idxs_tensor=emit_slot)

        # 4x in-place slot-shift. out == buf is safe because each output tile
        # depends only on the same-coordinate input tile, so per-core tile
        # iteration reads then writes without cross-tile hazards. Eliminates
        # 4 ttnn.copy dispatches.
        slot_shift_kernel(kv_state_front_2d, shift_P, kv_state_front_2d)
        slot_shift_kernel(kv_state_back_2d, shift_P, kv_state_back_2d)
        slot_shift_kernel(score_state_front_2d, shift_P, score_state_front_2d)
        slot_shift_kernel(score_state_back_2d, shift_P, score_state_back_2d)

        ttnn.copy(kv_normed, kv_normed_out)

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

    cssn(
        kv_state_front_2d, kv_state_back_2d,
        score_state_front_2d, score_state_back_2d,
        cssn_mask_front, cssn_mask_back, cssn_mask_pad,
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

    kv_normed = _device_rotate_activation(ttnn, kv_normed, H_tt)

    kv_4d = ttnn.reshape(kv_normed, [1, B, 1, d])
    kv_sharded = ttnn.to_memory_config(kv_4d, memory_config=sharded_input_memcfg)
    ttnn.experimental.paged_update_cache(
        kv_cache_tt, kv_sharded, update_idxs_tensor=emit_slot_tt)

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
        kv_state_front = torch.randn(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        kv_state_back = torch.randn(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_front = torch.randn(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
        score_state_back = torch.randn(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16) * 0.1
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

        # Snapshot original state buffers so we can re-upload between ref and kernel.
        state_torch = dict(
            kv_state_front=kv_state_front, kv_state_back=kv_state_back,
            score_state_front=score_state_front, score_state_back=score_state_back,
        )

        def fresh_state_uploads():
            return (
                upload_2d(state_torch["kv_state_front"]),
                upload_2d(state_torch["kv_state_back"]),
                upload_2d(state_torch["score_state_front"]),
                upload_2d(state_torch["score_state_back"]),
            )

        kv_sf_tt, kv_sb_tt, sc_sf_tt, sc_sb_tt = fresh_state_uploads()

        mf_tt = upload_2d(mf)
        mb_tt = upload_2d(mb)
        mp_tt = upload_2d(mp)
        gamma_tt = upload_2d(gamma)
        scaler_tt = upload_2d(scaler)
        cos_tt = upload_2d(cos_compressor)
        sin_tt = upload_2d(sin_compressor)
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

        zero_pad = torch.zeros(RATIO_PAD, INDEX_HEAD_DIM, dtype=torch.bfloat16)
        ninf_pad = torch.full_like(zero_pad, float("-inf"))
        kv_sf_scratch = upload_2d(zero_pad)
        kv_sb_scratch = upload_2d(zero_pad)
        sc_sf_scratch = upload_2d(ninf_pad)
        sc_sb_scratch = upload_2d(ninf_pad)

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
            kv_sf_tt, kv_sb_tt, sc_sf_tt, sc_sb_tt,
            mf_tt, mb_tt, mp_tt,
            gamma_tt, scaler_tt,
            cos_tt, sin_tt, start_pos_tt,
            H_tt, kv_cache_tt, emit_slot_tt, shift_P_tt,
            cssn_out_tt,
            kv_sf_scratch, kv_sb_scratch, sc_sf_scratch, sc_sb_scratch,
            sharded_memcfg)
        ref_host = download_chip0(mesh, mesh_shape, ref_kv_normed_tt)

        # Reset state buffers + scratches + cache for the kernel run.
        kv_sf_tt2, kv_sb_tt2, sc_sf_tt2, sc_sb_tt2 = fresh_state_uploads()
        kv_sf_scratch2 = upload_2d(zero_pad)
        kv_sb_scratch2 = upload_2d(zero_pad)
        sc_sf_scratch2 = upload_2d(ninf_pad)
        sc_sb_scratch2 = upload_2d(ninf_pad)
        kv_cache_tt2 = upload_2d(kv_cache_init)

        kernel = make_lk_d_idx_emit_kernel(
            mesh, cos_compressor, sin_compressor, sharded_memcfg)
        kv_normed_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(
            kv_sf_tt2, kv_sb_tt2, sc_sf_tt2, sc_sb_tt2,
            mf_tt, mb_tt, mp_tt,
            gamma_tt, scaler_tt,
            cos_tt, sin_tt, start_pos_tt,
            H_tt, kv_cache_tt2, emit_slot_tt, shift_P_tt,
            kv_sf_scratch2, kv_sb_scratch2, sc_sf_scratch2, sc_sb_scratch2,
            kv_normed_out_tt,
        )
        kernel_host = download_chip0(mesh, mesh_shape, kv_normed_out_tt)

        ok = report_pcc("Lk-D-idx-emit", ref_host, kernel_host)

        benchmark("Lk-D-idx-emit ref",
                  lambda: reference(
                      mesh,
                      kv_sf_tt, kv_sb_tt, sc_sf_tt, sc_sb_tt,
                      mf_tt, mb_tt, mp_tt,
                      gamma_tt, scaler_tt,
                      cos_tt, sin_tt, start_pos_tt,
                      H_tt, kv_cache_tt, emit_slot_tt, shift_P_tt,
                      cssn_out_tt,
                      kv_sf_scratch, kv_sb_scratch, sc_sf_scratch, sc_sb_scratch,
                      sharded_memcfg),
                  mesh)
        benchmark("Lk-D-idx-emit ttl",
                  lambda: kernel(
                      kv_sf_tt2, kv_sb_tt2, sc_sf_tt2, sc_sb_tt2,
                      mf_tt, mb_tt, mp_tt,
                      gamma_tt, scaler_tt,
                      cos_tt, sin_tt, start_pos_tt,
                      H_tt, kv_cache_tt2, emit_slot_tt, shift_P_tt,
                      kv_sf_scratch2, kv_sb_scratch2,
                      sc_sf_scratch2, sc_sb_scratch2,
                      kv_normed_out_tt),
                  mesh)

        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
