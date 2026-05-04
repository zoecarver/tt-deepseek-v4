"""Lk-A PCC test: hc_pre_attn + attn_norm + wq_a (sans wq_a all_gather).

Reference is the exact ttnn / ttl op chain `inference.py` runs between
the previous layer's `wo_b all_gather` (or the embed all_gather for
layer 0) and the wq_a all_gather. Specifically: `DeviceMHC.hc_pre_device`
+ ttnn.typecast + `DeviceRMSNorm.forward_device` + ttnn.slice + reshape +
`ttnn.matmul(wq_a)`. The all_gather after the matmul is excluded.

All tt-lang kernel definitions are inlined in this file (mhc_norm_fn_ksplit
+ mhc_split_mixes + mhc_sinkhorn + mhc_apply_mix_h + rmsnorm + SUMMA matmul).
The reference path still calls into inference.py's wrappers so the
comparison is apples-to-apples.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401  (sets up sys.path for `inference` import)
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark

from inference import (
    DeviceMHC, DeviceRMSNorm, _MHC_TILE, _MHC_PAD_SENTINEL,
    _MHC_POST_MULT, _mhc_pack_fn, _mhc_build_split_constant_tiles,
    _mhc_sinkhorn_mask_tile, _mhc_sinkhorn_eps_mask_tile,
)


# V4-Flash decode shape.
DIM = 4096
MHC = 4
MHC_MULT3 = MHC * 2 + MHC * MHC                # 24
D = MHC * DIM                                   # 16384
Q_LORA_RANK = 1024
NORM_EPS = 1e-6
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20
NUM_TOKENS = 1
NUM_TOKENS_PAD = _MHC_TILE                      # 32
TILE = 32
NORM_FN_KP = 8


def _make_mhc_norm_fn_ksplit_kernel(K_tiles: int, Kp: int,
                                    rms_eps: float, inv_D: float,
                                    *, grid_cols: int = 8,
                                    bk: int = 2):
    """K-axis-parallel mhc norm_fn for num_out_tiles=1 (decode).

    a_cb shape=(1, bk), b_cb shape=(bk, 1): each core's K-loop runs
    K_NB=K_BPN/bk SUMMA-style accumulations of bk-tile products.
    bk=2 is the max valid block size; bk=4 hits PCC due to DST register
    pressure with fp32_dest_acc_en (a*a (1,bk)=(1,bk) tile + matmul DST).
    """
    if Kp < 2:
        raise ValueError(f"ksplit kernel requires Kp >= 2, got {Kp}")
    if K_tiles % Kp:
        raise ValueError(f"K_tiles={K_tiles} not divisible by Kp={Kp}")
    if Kp < grid_cols:
        grid_cols = Kp
    if Kp % grid_cols:
        raise ValueError(f"Kp={Kp} not divisible by grid_cols={grid_cols}")
    if Kp - 1 > 32:
        raise ValueError(f"Kp={Kp} requires block_count={Kp-1} > 32")

    K_BPN = K_tiles // Kp
    if K_BPN % bk:
        raise ValueError(f"K_BPN={K_BPN} not divisible by bk={bk}")
    K_NB = K_BPN // bk
    COL = grid_cols
    ROW = Kp // grid_cols

    @ttl.operation(
        grid=(COL, ROW),
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def norm_fn_ksplit_kernel(a, b, scaler, out):
        c_reduce_pipes = [
            ttl.Pipe(src=(col, row), dst=(0, 0))
            for row in range(ROW)
            for col in range(COL)
            if not (col == 0 and row == 0)
        ]
        c_reduce_net = ttl.PipeNet(c_reduce_pipes)

        sq_reduce_pipes = [
            ttl.Pipe(src=(col, row), dst=(0, 0))
            for row in range(ROW)
            for col in range(COL)
            if not (col == 0 and row == 0)
        ]
        sq_reduce_net = ttl.PipeNet(sq_reduce_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(b, shape=(bk, 1), block_count=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        c_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        sq_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_bc_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        asq_cb = ttl.make_dataflow_buffer_like(a, shape=(1, bk), block_count=2)
        red_step_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        recv_c_cb = ttl.make_dataflow_buffer_like(
            out, shape=(1, 1), block_count=max(2, Kp - 1))
        recv_sq_cb = ttl.make_dataflow_buffer_like(
            scaler, shape=(1, 1), block_count=max(2, Kp - 1))

        @ttl.compute()
        def compute():
            col_c, row_c = ttl.node(dims=2)
            sc = sc_cb.wait()

            a0 = a_cb.wait()
            b0 = b_cb.wait()
            c_cb.reserve().store(a0 @ b0)
            asq_cb.reserve().store(a0 * a0)
            sq_cb.reserve().store(
                ttl.math.reduce_sum(asq_cb.wait(), sc, dims=[1])
            )

            for _ in range(K_NB - 1):
                ak = a_cb.wait()
                bk_blk = b_cb.wait()
                prev_c = c_cb.wait()
                c_cb.reserve().store(prev_c + ak @ bk_blk)

                asq_cb.reserve().store(ak * ak)
                red_step_cb.reserve().store(
                    ttl.math.reduce_sum(asq_cb.wait(), sc, dims=[1])
                )
                prev_sq = sq_cb.wait()
                sq_cb.reserve().store(prev_sq + red_step_cb.wait())

            if col_c == 0 and row_c == 0:
                for _ in range(Kp - 1):
                    prev_c = c_cb.wait()
                    r = recv_c_cb.wait()
                    new_c = c_cb.reserve()
                    new_c.store(prev_c + r)
                for _ in range(Kp - 1):
                    prev_sq = sq_cb.wait()
                    r = recv_sq_cb.wait()
                    new_sq = sq_cb.reserve()
                    new_sq.store(prev_sq + r)

                sq_total = sq_cb.wait()
                inv_bc = inv_bc_cb.reserve()
                inv_bc.store(ttl.math.broadcast(
                    ttl.math.rsqrt(
                        sq_total * ttl.math.fill(sq_total, inv_D)
                        + ttl.math.fill(sq_total, rms_eps)
                    ),
                    inv_bc, dims=[1],
                ))
                c_total = c_cb.wait()
                out_cb.reserve().store(c_total * inv_bc_cb.wait())

        @ttl.datamovement()
        def dm_read():
            col_c, row_c = ttl.node(dims=2)
            k_p = row_c * COL + col_c

            ttl.copy(scaler[0, 0], sc_cb.reserve()).wait()

            for kb_local in range(K_NB):
                kc = (k_p * K_NB + kb_local) * bk
                ttl.copy(a[0:1, kc:kc + bk], a_cb.reserve()).wait()
                ttl.copy(b[kc:kc + bk, 0:1], b_cb.reserve()).wait()

            is_root = (col_c == 0 and row_c == 0)
            if is_root:
                def recv_c(pipe):
                    blk = recv_c_cb.reserve()
                    ttl.copy(pipe, blk).wait()
                c_reduce_net.if_dst(recv_c)

                def recv_sq(pipe):
                    blk = recv_sq_cb.reserve()
                    ttl.copy(pipe, blk).wait()
                sq_reduce_net.if_dst(recv_sq)
            else:
                p_c = c_cb.wait()

                def send_c(pipe):
                    ttl.copy(p_c, pipe).wait()
                c_reduce_net.if_src(send_c)

                p_sq = sq_cb.wait()

                def send_sq(pipe):
                    ttl.copy(p_sq, pipe).wait()
                sq_reduce_net.if_src(send_sq)

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            if col_c == 0 and row_c == 0:
                ttl.copy(out_cb.wait(), out[0, 0]).wait()

    return norm_fn_ksplit_kernel


def _make_mhc_split_mixes_kernel(num_tiles: int):
    """Per-token elementwise split of mixes into pre/post/comb sections.
    Inlined from inference.py / tt-lang-kernels/pre_split_mixes.py.
    """

    @ttl.operation(grid="auto")
    def split_mixes_kernel(
        input_mixes, scale_tile, base_tile,
        pre_mask, pre_eps_tile, post_mult_mask, comb_mask,
        pre_out, post_out, comb_out,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_tiles // total_cores)

        in_dfb = ttl.make_dataflow_buffer_like(input_mixes, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
        base_dfb = ttl.make_dataflow_buffer_like(base_tile, shape=(1, 1), block_count=1)
        prem_dfb = ttl.make_dataflow_buffer_like(pre_mask, shape=(1, 1), block_count=1)
        preeps_dfb = ttl.make_dataflow_buffer_like(pre_eps_tile, shape=(1, 1), block_count=1)
        postmm_dfb = ttl.make_dataflow_buffer_like(post_mult_mask, shape=(1, 1), block_count=1)
        combm_dfb = ttl.make_dataflow_buffer_like(comb_mask, shape=(1, 1), block_count=1)

        sig_dfb = ttl.make_dataflow_buffer_like(input_mixes, shape=(1, 1), block_count=2)

        pre_out_dfb = ttl.make_dataflow_buffer_like(pre_out, shape=(1, 1), block_count=2)
        post_out_dfb = ttl.make_dataflow_buffer_like(post_out, shape=(1, 1), block_count=2)
        comb_out_dfb = ttl.make_dataflow_buffer_like(comb_out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()
            base = base_dfb.wait()
            prem = prem_dfb.wait()
            preeps = preeps_dfb.wait()
            postmm = postmm_dfb.wait()
            combm = combm_dfb.wait()

            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    inp = in_dfb.wait()
                    sig_dfb.reserve().store(ttl.math.sigmoid(inp * sc + base))
                    comb_out_dfb.reserve().store((inp * sc + base) * combm)

                    sig = sig_dfb.wait()
                    pre_out_dfb.reserve().store(sig * prem + preeps)
                    post_out_dfb.reserve().store(sig * postmm)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scale_tile[0, 0], sc_dfb.reserve()).wait()
            ttl.copy(base_tile[0, 0], base_dfb.reserve()).wait()
            ttl.copy(pre_mask[0, 0], prem_dfb.reserve()).wait()
            ttl.copy(pre_eps_tile[0, 0], preeps_dfb.reserve()).wait()
            ttl.copy(post_mult_mask[0, 0], postmm_dfb.reserve()).wait()
            ttl.copy(comb_mask[0, 0], combm_dfb.reserve()).wait()
            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    ttl.copy(input_mixes[global_i, 0], in_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_i in range(tiles_per_core):
                global_i = core_idx * tiles_per_core + local_i
                if global_i < num_tiles:
                    ttl.copy(pre_out_dfb.wait(), pre_out[global_i, 0]).wait()
                    ttl.copy(post_out_dfb.wait(), post_out[global_i, 0]).wait()
                    ttl.copy(comb_out_dfb.wait(), comb_out[global_i, 0]).wait()

    return split_mixes_kernel


def _make_mhc_sinkhorn_kernel(num_slices: int, repeat: int, eps: float):
    """Per-slice iterative softmax + alternating row/col normalize.
    Inlined from inference.py / tt-lang-kernels/sinkhorn.py.
    """

    @ttl.operation(grid="auto", options="--no-ttl-reduce-full-fp32")
    def sinkhorn_kernel(x, mask, eps_mask, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        slices_per_core = -(-num_slices // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), block_count=1)
        em_dfb = ttl.make_dataflow_buffer_like(eps_mask, shape=(1, 1), block_count=1)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        state_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        state_copy_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        exp_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()
            m = m_dfb.wait()
            em = em_dfb.wait()

            for local_i in range(slices_per_core):
                global_i = core_idx * slices_per_core + local_i
                if global_i < num_slices:
                    # Row softmax: state := softmax(x, dim=-1)
                    x_in = x_dfb.wait()
                    red_dfb.reserve().store(ttl.math.reduce_max(x_in, sc, dims=[1]))
                    rmx = bc_dfb.reserve()
                    rmx.store(ttl.math.broadcast(red_dfb.wait(), rmx, dims=[1]))
                    exp_dfb.reserve().store(ttl.math.exp(x_in - bc_dfb.wait()))

                    ex = exp_dfb.wait()
                    red_dfb.reserve().store(ttl.math.reduce_sum(ex, sc, dims=[1]))
                    state_copy_dfb.reserve().store(ex)
                    rinv = bc_dfb.reserve()
                    rinv.store(ttl.math.broadcast(ttl.math.recip(red_dfb.wait()), rinv, dims=[1]))
                    state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                    # Mask + eps: state := state * mask + eps_mask
                    state_copy_dfb.reserve().store(state_dfb.wait() * m + em)
                    state_dfb.reserve().store(state_copy_dfb.wait())

                    # First col-normalize: state := state / (col_sum + eps)
                    s = state_dfb.wait()
                    red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[0]))
                    state_copy_dfb.reserve().store(s)
                    cinv = bc_dfb.reserve()
                    csum = red_dfb.wait()
                    cinv.store(ttl.math.broadcast(
                        ttl.math.recip(csum + ttl.math.fill(csum, eps)),
                        cinv, dims=[0]))
                    state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                    for _ in range(repeat - 1):
                        s = state_dfb.wait()
                        red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[1]))
                        state_copy_dfb.reserve().store(s)
                        rinv = bc_dfb.reserve()
                        rsum = red_dfb.wait()
                        rinv.store(ttl.math.broadcast(
                            ttl.math.recip(rsum + ttl.math.fill(rsum, eps)),
                            rinv, dims=[1]))
                        state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                        s = state_dfb.wait()
                        red_dfb.reserve().store(ttl.math.reduce_sum(s, sc, dims=[0]))
                        state_copy_dfb.reserve().store(s)
                        cinv = bc_dfb.reserve()
                        csum = red_dfb.wait()
                        cinv.store(ttl.math.broadcast(
                            ttl.math.recip(csum + ttl.math.fill(csum, eps)),
                            cinv, dims=[0]))
                        state_dfb.reserve().store(state_copy_dfb.wait() * bc_dfb.wait())

                    out_dfb.reserve().store(state_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(mask[0, 0], m_dfb.reserve()).wait()
            ttl.copy(eps_mask[0, 0], em_dfb.reserve()).wait()
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_i in range(slices_per_core):
                global_i = core_idx * slices_per_core + local_i
                if global_i < num_slices:
                    ttl.copy(x[global_i, 0], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_i in range(slices_per_core):
                global_i = core_idx * slices_per_core + local_i
                if global_i < num_slices:
                    ttl.copy(out_dfb.wait(), out[global_i, 0]).wait()

    return sinkhorn_kernel


def _make_mhc_apply_mix_h_kernel(num_tokens: int, h_tiles: int,
                                 *, h_block: int = 16):
    """h-axis-sharded apply_mix. Per-(token, h-tile): out[t, h] = sum_m x[t,m,h] * mix[t,m].

    DFBs hold shape=(1, h_block) tile blocks. Total work shrinks
    h_blocks=h_tiles/h_block; each work unit batches h_block consecutive
    h-tiles. mix stays (1, 1) (per-token scalar).
    """
    if h_tiles % h_block:
        raise ValueError(
            f"h_tiles={h_tiles} not divisible by h_block={h_block}")
    h_blocks = h_tiles // h_block
    total_work = num_tokens * h_blocks
    BH = h_block

    @ttl.operation(grid="auto")
    def apply_mix_h_kernel(x, mix, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)
        mix_dfb = ttl.make_dataflow_buffer_like(mix, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BH), block_count=2)

        mix_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)
        prod_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)
        red_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BH), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            sc = sc_dfb.wait()

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    mix_raw = mix_dfb.wait()
                    mx = mix_bc_dfb.reserve()
                    mx.store(ttl.math.broadcast(mix_raw, mx, dims=[1]))
                    mix_bc = mix_bc_dfb.wait()

                    x_blk = x_dfb.wait()
                    prod_dfb.reserve().store(x_blk * mix_bc)
                    red_dfb.reserve().store(
                        ttl.math.reduce_sum(prod_dfb.wait(), sc, dims=[0])
                    )
                    out_dfb.reserve().store(red_dfb.wait())

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    global_t = global_w // h_blocks
                    global_hb = global_w % h_blocks
                    h_start = global_hb * BH
                    ttl.copy(mix[global_t, 0], mix_dfb.reserve()).wait()
                    ttl.copy(
                        x[global_t:global_t + 1, h_start:h_start + BH],
                        x_dfb.reserve(),
                    ).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    global_t = global_w // h_blocks
                    global_hb = global_w % h_blocks
                    h_start = global_hb * BH
                    ttl.copy(
                        out_dfb.wait(),
                        out[global_t:global_t + 1, h_start:h_start + BH],
                    ).wait()

    return apply_mix_h_kernel


def _make_fused_rms_summa_kernel(M: int, K: int, N: int,
                                 block_cfg, part_cfg,
                                 rms_eps: float,
                                 fp32_dest_acc_en: bool = True):
    """Fused rmsnorm(x, gamma) @ Wg as a single ttl.operation (Kp=1 SUMMA).

    Wg = gamma[:, None] * W is pre-baked on host. Each (n_p, m_p) core
    sees the full K range of A via mcast (Kp=1, no K-split), accumulates
    matmul partial AND ssq partial across Kb K-tiles, then independently
    applies inv = rsqrt(ssq/D + eps) to its matmul output. ssq is computed
    redundantly across n_p (each core sees the same A and gets the same
    answer), so no cross-row reduce is needed.

    Mirrors `_make_summa_matmul_kernel` plus the ssq fusion logic.
    """
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Kp != 1:
        raise ValueError(f"fused rms+summa requires Kp=1, got {Kp}")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    M_BPN = Mb // Mp
    N_BPN = Nb // Np
    inv_D = 1.0 / float(K)

    @ttl.operation(
        grid=(Np, Mp),
        fp32_dest_acc_en=fp32_dest_acc_en,
        options="--no-ttl-reduce-full-fp32",
    )
    def fused_kernel(a, w_g, scaler, out):
        # A row-mcast across n_p; B unicast per (n_p, m_p).
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p))
                   for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w_g, shape=(bk, bn), block_count=2)
        partial_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        xsq_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        ssq_step_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, 1), block_count=2)
        ssq_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, 1), block_count=2)
        inv_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, 1), block_count=2)
        inv_bc_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            sc = sc_cb.wait()
            for m_idx in range(M_BPN):
                # First N-block: matmul + ssq fused over Kb. ssq is the
                # same regardless of n_idx, but only needed once.
                a0 = a_cb.wait()
                b0 = b_cb.wait()
                partial_cb.reserve().store(a0 @ b0)
                xsq_cb.reserve().store(a0 * a0)
                ssq_cb.reserve().store(
                    ttl.math.reduce_sum(xsq_cb.wait(), sc, dims=[1])
                )
                for _ in range(Kb - 1):
                    ak = a_cb.wait()
                    bk_blk = b_cb.wait()
                    prev_acc = partial_cb.wait()
                    partial_cb.reserve().store(prev_acc + ak @ bk_blk)
                    xsq_cb.reserve().store(ak * ak)
                    ssq_step_cb.reserve().store(
                        ttl.math.reduce_sum(xsq_cb.wait(), sc, dims=[1])
                    )
                    prev_ssq = ssq_cb.wait()
                    ssq_cb.reserve().store(prev_ssq + ssq_step_cb.wait())

                # Finalize inv_rms for this M-block.
                sq = ssq_cb.wait()
                inv_t = inv_cb.reserve()
                inv_t.store(ttl.math.broadcast(
                    ttl.math.rsqrt(
                        sq * ttl.math.fill(sq, inv_D)
                        + ttl.math.fill(sq, rms_eps)
                    ),
                    inv_t, dims=[1],
                ))
                inv_bc_t = inv_bc_cb.reserve()
                inv_bc_t.store(ttl.math.broadcast(
                    inv_cb.wait(), inv_bc_t, dims=[1]
                ))
                inv = inv_bc_cb.wait()  # held across the N loop below.

                # Apply inv to first N-block partial.
                acc_done = partial_cb.wait()
                out_cb.reserve().store(acc_done * inv)

                # Subsequent N-blocks for this M: matmul-only + apply inv.
                for _ in range(N_BPN - 1):
                    a0n = a_cb.wait()
                    b0n = b_cb.wait()
                    partial_cb.reserve().store(a0n @ b0n)
                    for _ in range(Kb - 1):
                        akn = a_cb.wait()
                        bkn = b_cb.wait()
                        prev_acc_n = partial_cb.wait()
                        partial_cb.reserve().store(prev_acc_n + akn @ bkn)
                    acc_n = partial_cb.wait()
                    out_cb.reserve().store(acc_n * inv)

        @ttl.datamovement()
        def dm_read():
            ttl.copy(scaler[0, 0], sc_cb.reserve()).wait()
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
                        ttl.copy(w_g[kc:kc + bk, nc:nc + bn], b_blk).wait()
                    o = out_cb.wait()
                    ttl.copy(o, out[mr:mr + bm, nc:nc + bn]).wait()

    return fused_kernel


def make_lk_a_kernel(mesh, hc_fn_cpu, hc_scale_cpu, hc_base_cpu, gamma_cpu):
    """Mega kernel for Lk-A = hc_pre + typecast + attn_norm + wq_a matmul.

    Composes 5 tt-lang dispatches all defined locally in this file:
    norm_fn_ksplit + split_mixes + sinkhorn + apply_mix_h +
    fused (rmsnorm + SUMMA wq_a matmul). Constants/scratch tensors are
    allocated up front from cpu-side weights.
    """
    K_tiles = D // TILE                                # 512
    h_tiles = DIM // TILE                              # 128
    norm_fn_kernel = _make_mhc_norm_fn_ksplit_kernel(
        K_tiles=K_tiles, Kp=NORM_FN_KP,
        rms_eps=NORM_EPS, inv_D=1.0 / D)
    split_mixes_kernel = _make_mhc_split_mixes_kernel(
        num_tiles=NUM_TOKENS_PAD // _MHC_TILE)         # 1
    sinkhorn_kernel = _make_mhc_sinkhorn_kernel(
        num_slices=NUM_TOKENS, repeat=HC_SINKHORN_ITERS, eps=HC_EPS)
    apply_mix_kernel = _make_mhc_apply_mix_h_kernel(
        num_tokens=NUM_TOKENS, h_tiles=h_tiles)
    fused_rms_matmul_kernel = _make_fused_rms_summa_kernel(
        M=TILE, K=DIM, N=Q_LORA_RANK,
        block_cfg=(1, 4, 8), part_cfg=(1, 8, 1),
        rms_eps=NORM_EPS)

    rep_fp32 = dict(device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    rep_bf16 = dict(device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    # Constant tensors (uploaded once at kernel construction).
    fn_packed = _mhc_pack_fn(
        hc_fn_cpu.detach().to(torch.float32).cpu(), MHC_MULT3)
    fn_tt = ttnn.as_tensor(fn_packed, **rep_fp32)
    sc_fp32_tt = ttnn.as_tensor(
        torch.ones((_MHC_TILE, _MHC_TILE), dtype=torch.float32), **rep_fp32)
    sc_bf16_tt = ttnn.as_tensor(
        torch.ones((TILE, TILE), dtype=torch.bfloat16), **rep_bf16)

    (scale_tile, base_tile, pre_mask_tile, pre_eps_tile,
     post_mult_mask_tile, comb_mask_tile) = _mhc_build_split_constant_tiles(
        hc_scale_cpu.detach().to(torch.float32).cpu(),
        hc_base_cpu.detach().to(torch.float32).cpu(),
        MHC, _MHC_POST_MULT, HC_EPS,
    )
    split_scale_tt = ttnn.as_tensor(scale_tile, **rep_fp32)
    split_base_tt = ttnn.as_tensor(base_tile, **rep_fp32)
    split_pre_mask_tt = ttnn.as_tensor(pre_mask_tile, **rep_fp32)
    split_pre_eps_tt = ttnn.as_tensor(pre_eps_tile, **rep_fp32)
    split_post_mult_mask_tt = ttnn.as_tensor(post_mult_mask_tile, **rep_fp32)
    split_comb_mask_tt = ttnn.as_tensor(comb_mask_tile, **rep_fp32)

    sk_mask_tt = ttnn.as_tensor(_mhc_sinkhorn_mask_tile(MHC), **rep_fp32)
    sk_eps_mask_tt = ttnn.as_tensor(
        _mhc_sinkhorn_eps_mask_tile(MHC, HC_EPS), **rep_fp32)

    state: dict = {}

    def _zeros_fp32(shape):
        return ttnn.zeros(
            shape=tuple(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _zeros_bf16(shape):
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def lk_a_kernel(a_tt, wq_a_w_tt, out_tt, x_pre_norm_bf16_out=None):
        if "scratch" not in state:
            # mhc scratch
            state["mixes_tt"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["pre_tt"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["post_tt"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["comb_tt"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["comb_sk_out_tt"] = _zeros_fp32((NUM_TOKENS * _MHC_TILE, _MHC_TILE))
            state["apply_mix_out_tt"] = _zeros_fp32((NUM_TOKENS * _MHC_TILE, DIM))
            state["a_sliced_scratch_tt"] = _zeros_fp32((NUM_TOKENS, MHC * DIM))
            # fused rms+matmul scratch
            state["rms_in_tt"] = _zeros_bf16((TILE, DIM))   # bf16 input post-typecast
            state["y_padded_tt"] = _zeros_bf16((TILE, Q_LORA_RANK))
            state["scratch"] = True

        mixes_tt = state["mixes_tt"]
        pre_tt = state["pre_tt"]
        post_tt = state["post_tt"]
        comb_tt = state["comb_tt"]
        comb_sk_out_tt = state["comb_sk_out_tt"]
        apply_mix_out_tt = state["apply_mix_out_tt"]
        a_sliced_scratch_tt = state["a_sliced_scratch_tt"]

        # 1. mhc.norm_fn (ksplit) on residual a_tt -> mixes_tt
        norm_fn_kernel(a_tt, fn_tt, sc_fp32_tt, mixes_tt)

        # 2. split_mixes -> pre/post/comb
        split_mixes_kernel(
            mixes_tt, split_scale_tt, split_base_tt,
            split_pre_mask_tt, split_pre_eps_tt,
            split_post_mult_mask_tt, split_comb_mask_tt,
            pre_tt, post_tt, comb_tt,
        )

        # 3. ttnn glue: pull comb's mhc*mhc tail, pad to [num_tokens, TILE, TILE].
        comb_sliced = ttnn.slice(
            comb_tt, [0, 2 * MHC], [NUM_TOKENS, 2 * MHC + MHC * MHC])
        comb_3d = ttnn.reshape(comb_sliced, [NUM_TOKENS, MHC, MHC])
        comb_padded = ttnn.pad(
            comb_3d,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, _MHC_TILE - MHC)],
            value=_MHC_PAD_SENTINEL,
        )
        comb_sk_in_tt = ttnn.reshape(
            comb_padded, [NUM_TOKENS * _MHC_TILE, _MHC_TILE])

        # 4. sinkhorn -> comb_sk_out_tt
        sinkhorn_kernel(
            comb_sk_in_tt, sk_mask_tt, sk_eps_mask_tt, sc_fp32_tt,
            comb_sk_out_tt,
        )

        # 5. ttnn glue: build mix tile from pre's first mhc cols.
        pre_sliced = ttnn.slice(pre_tt, [0, 0], [NUM_TOKENS, MHC])
        pre_3d = ttnn.reshape(pre_sliced, [NUM_TOKENS, MHC, 1])
        mix_padded = ttnn.pad(
            pre_3d,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, _MHC_TILE - 1)],
            value=0.0,
        )
        mix_tt = ttnn.reshape(
            mix_padded, [NUM_TOKENS * _MHC_TILE, _MHC_TILE])

        # 6. ttnn glue: reshape a_tt rows -> [num_tokens*TILE, hidden].
        ttnn.slice(a_tt, [0, 0], [NUM_TOKENS, MHC * DIM],
                   output_tensor=a_sliced_scratch_tt)
        a_3d = ttnn.reshape(a_sliced_scratch_tt, [NUM_TOKENS, MHC, DIM])
        x_padded = ttnn.pad(
            a_3d,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, 0)],
            value=0.0,
        )
        x_tt = ttnn.reshape(x_padded, [NUM_TOKENS * _MHC_TILE, DIM])

        # 7. apply_mix_h -> apply_mix_out_tt (fp32)
        apply_mix_kernel(x_tt, mix_tt, sc_fp32_tt, apply_mix_out_tt)

        # 8. typecast fp32 -> bf16 (apply_mix_out_tt is [TILE, DIM]).
        ttnn.typecast(apply_mix_out_tt, dtype=ttnn.bfloat16,
                      output_tensor=state["rms_in_tt"])

        # 9. fused rmsnorm + wq_a matmul. gamma is pre-baked into wq_a_w_tt.
        fused_rms_matmul_kernel(
            state["rms_in_tt"], wq_a_w_tt, sc_bf16_tt, state["y_padded_tt"])
        y_row = ttnn.slice(state["y_padded_tt"], [0, 0],
                           [NUM_TOKENS, Q_LORA_RANK])
        y_3d = ttnn.reshape(y_row, [1, 1, Q_LORA_RANK])
        ttnn.copy(y_3d, out_tt)

        # 10. Optionally publish the bf16 pre-norm hc_pre output as
        # [TILE, DIM] for downstream consumers. Lk-C needs the post-norm
        # bf16 tensor; the fused rms+matmul kernel above pre-bakes gamma
        # into wq_a so it doesn't materialize a separate norm tile. The
        # caller runs a tiny rms_norm on this pre-norm buffer to get the
        # post-norm bridge tensor for wkv (one extra device op, no host).
        if x_pre_norm_bf16_out is not None:
            ttnn.copy(state["rms_in_tt"], x_pre_norm_bf16_out)

    return lk_a_kernel


def reference(mesh, a_tt, hc_fn_cpu, hc_scale_cpu, hc_base_cpu,
              gamma_cpu, wq_a_w_tt):
    """Run the exact ttnn / ttl op chain from inference.py."""
    mhc = DeviceMHC(
        mesh=mesh,
        hc_fn=hc_fn_cpu, hc_scale=hc_scale_cpu, hc_base=hc_base_cpu,
        hc_mult=MHC, hc_eps=HC_EPS,
        sinkhorn_iters=HC_SINKHORN_ITERS, norm_eps=NORM_EPS,
    )
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=gamma_cpu, eps=NORM_EPS)

    # block.hc_pre — DeviceMHC.hc_pre_device.
    hc_out_fp32 = mhc.hc_pre_device(NUM_TOKENS, NUM_TOKENS_PAD, a_tt=a_tt)
    # block.norm — typecast then rmsnorm via the tt-lang kernel.
    ttnn.typecast(hc_out_fp32, dtype=ttnn.bfloat16,
                  output_tensor=rmsn._x_upload_tt)
    norm_out_tt = rmsn.forward_device(rmsn._x_upload_tt, NUM_TOKENS)
    # bridge into [B, S, hidden].
    ttnn.slice(norm_out_tt, [0, 0], [NUM_TOKENS, DIM],
               output_tensor=mhc._norm_slice_tt)
    bridge_tt = ttnn.reshape(mhc._norm_slice_tt, [1, 1, DIM])
    # wq_a matmul (no all_gather; weight is replicated for the test).
    return ttnn.matmul(bridge_tt, wq_a_w_tt,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        # Random inputs at decode shape.
        a = torch.randn(NUM_TOKENS_PAD, D, dtype=torch.float32) * 0.1
        hc_fn = torch.randn(MHC_MULT3, D, dtype=torch.float32) * 0.05
        hc_scale = torch.randn(3, dtype=torch.float32)
        hc_base = torch.randn(MHC_MULT3, dtype=torch.float32) * 0.01
        gamma = torch.ones(DIM, dtype=torch.bfloat16)
        wq_a_w = torch.randn(DIM, Q_LORA_RANK, dtype=torch.bfloat16) * 0.02

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        a_tt = ttnn.as_tensor(a.contiguous(), dtype=ttnn.float32, **rep)
        wq_a_w_tt = ttnn.as_tensor(wq_a_w.contiguous(), dtype=ttnn.bfloat16, **rep)

        # Pre-bake gamma into wq_a for the fused rms+matmul kernel:
        # Wg[k, n] = gamma[k] * W[k, n]. Reference path keeps original W.
        wq_a_baked = (gamma.float()[:, None] * wq_a_w.float()) \
            .to(torch.bfloat16).contiguous()
        wq_a_baked_tt = ttnn.as_tensor(
            wq_a_baked, dtype=ttnn.bfloat16, **rep)

        # Reference: exact ttnn/ttl chain from inference.py.
        ref_out_tt = reference(mesh, a_tt, hc_fn, hc_scale, hc_base,
                               gamma, wq_a_w_tt)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_a_kernel(mesh, hc_fn, hc_scale, hc_base, gamma)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, Q_LORA_RANK, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(a_tt, wq_a_baked_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-A", ref_host, kernel_host)

        benchmark("Lk-A ref",
                  lambda: reference(mesh, a_tt, hc_fn, hc_scale, hc_base,
                                    gamma, wq_a_w_tt),
                  mesh)
        benchmark("Lk-A ttl",
                  lambda: kernel(a_tt, wq_a_baked_tt, out_tt),
                  mesh)

        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
