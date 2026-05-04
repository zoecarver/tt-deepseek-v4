"""Lk-E PCC test: hc_post_attn + hc_pre_ffn + ffn_norm + shared expert (sans all_reduce).

Reference covers everything from `wo_b all_gather` (the attn output is now
replicated across the mesh) up to the shared-expert all_reduce.

For test self-containment the kernel runs the full hc_pre_attn +
hc_post_attn + hc_pre_ffn pipeline. In production Lk-A's hc_pre_attn
would supply the post/comb_sk stash directly (see TODO below).

All tt-lang kernel definitions are inlined: norm_fn_ksplit, split_mixes,
sinkhorn, apply_mix_h, mhc_post, rmsnorm, and three SUMMA matmul
instantiations for w1, w3, w2.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark

from inference import (
    DeviceMHC, DeviceRMSNorm, _MHC_TILE, _MHC_PAD_SENTINEL,
    _MHC_POST_MULT, _mhc_pack_fn, _mhc_build_split_constant_tiles,
    _mhc_sinkhorn_mask_tile, _mhc_sinkhorn_eps_mask_tile,
    _mhc_post_to_a_tt,
)


DIM = 4096
MHC = 4
MHC_MULT3 = MHC * 2 + MHC * MHC                # 24
D = MHC * DIM                                   # 16384
INTER_DIM = 2048
NORM_EPS = 1e-6
HC_EPS = 1e-6
HC_SINKHORN_ITERS = 20
SWIGLU_LIMIT = 10.0
NUM_TOKENS = 1
NUM_TOKENS_PAD = _MHC_TILE
B, S = 1, 1
TILE = 32
NORM_FN_KP = 8

# TODO: mega in production Lk-A emits hc_attn_post_mix + hc_attn_comb_sk
# (the hc_pre_attn stash) and Lk-E only does hc_post_attn → hc_pre_ffn →
# ffn_norm → shared expert. For self-contained PCC testing the kernel
# below re-runs hc_pre_attn on prev_a; once Lk-A's stash exit interface
# is wired through we can drop the duplicated hc_pre_attn here.


def _make_mhc_norm_fn_ksplit_kernel(K_tiles: int, Kp: int,
                                    rms_eps: float, inv_D: float,
                                    *, grid_cols: int = 8,
                                    bk: int = 2):
    """K-split rmsnorm + matmul: a (1, K_tiles) @ b (K_tiles, 1) -> out (1, 1).

    Each core handles K_BPN=K_tiles/Kp K-tiles, batched in BK=bk-tile
    chunks. a_cb shape=(1, BK), b_cb shape=(BK, 1) -> a @ b is one
    SUMMA accumulation per BK block. xsq is (1, BK), reduce_sum
    dim=[1] collapses both within-tile cols AND across BK tile-cols.
    """
    if Kp < 2:
        raise ValueError(f"ksplit kernel requires Kp >= 2, got {Kp}")
    if K_tiles % Kp:
        raise ValueError(f"K_tiles={K_tiles} not divisible by Kp={Kp}")
    if Kp < grid_cols:
        grid_cols = Kp
    if Kp % grid_cols:
        raise ValueError(f"Kp={Kp} not divisible by grid_cols={grid_cols}")

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
    # Grid sized to num_tiles (one tile per core). Production callers pass
    # num_tiles=1 (one MHC tile of mix scalars), so grid collapses to (1,1).
    grid_cols = min(num_tiles, 8)
    grid_rows = -(-num_tiles // grid_cols)

    @ttl.operation(grid=(grid_cols, grid_rows))
    def split_mixes_kernel(
        input_mixes, scale_tile, base_tile,
        pre_mask, pre_eps_tile, post_mult_mask, comb_mask,
        pre_out, post_out, comb_out,
    ):
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
    # Grid sized to num_slices (one MHC tile per slice, ~20 reductions
    # each). Production passes num_slices=NUM_TOKENS=1, so grid=(1,1).
    grid_cols = min(num_slices, 8)
    grid_rows = -(-num_slices // grid_cols)

    @ttl.operation(grid=(grid_cols, grid_rows),
                   options="--no-ttl-reduce-full-fp32")
    def sinkhorn_kernel(x, mask, eps_mask, scaler, out):
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
                    state_copy_dfb.reserve().store(state_dfb.wait() * m + em)
                    state_dfb.reserve().store(state_copy_dfb.wait())
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
    """Per (token, h_tile): out[t, h] = reduce_sum(x[t, h] * mix[t]_bc, dim=0).

    Batches h_block consecutive h-tiles per work unit. DFBs hold
    shape=(1, h_block); the mix broadcast and elementwise + reduce all
    span the BH-tile block. mix stays (1, 1) (per-token scalar).
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


def _make_mhc_post_kernel(num_tokens: int, h_tiles: int, *, h_block: int = 16):
    """Inlined from inference.py / tt-lang-kernels/post.py.

    Per-token, per-h-tile: out = x * post_mix_bc + comb^T @ residual.

    Parallelizes over (token, h_block) work units. Each work unit
    processes h_block consecutive h-tiles in a single pass. DFBs hold
    shape=(1, h_block) tile blocks so x/res/out and the broadcast
    post_bc / matmul / post_term scratch buffers all amortize their
    reserve/wait overhead across h_block tiles. post_mix and comb_T
    stay (1, 1) (per-token scalars).
    """
    if h_tiles % h_block:
        raise ValueError(
            f"h_tiles={h_tiles} not divisible by h_block={h_block}")
    h_blocks = h_tiles // h_block
    total_work = num_tokens * h_blocks
    BH = h_block

    @ttl.operation(grid="auto")
    def post_kernel(x, residual, comb_T, post_mix, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)
        res_dfb = ttl.make_dataflow_buffer_like(residual, shape=(1, BH), block_count=2)
        comb_dfb = ttl.make_dataflow_buffer_like(comb_T, shape=(1, 1), block_count=2)
        post_dfb = ttl.make_dataflow_buffer_like(post_mix, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BH), block_count=2)
        post_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)
        post_term_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)
        matmul_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    post = post_dfb.wait()
                    comb_t = comb_dfb.wait()
                    pbc = post_bc_dfb.reserve()
                    pbc.store(ttl.math.broadcast(post, pbc, dims=[1]))
                    post_bc = post_bc_dfb.wait()
                    x_blk = x_dfb.wait()
                    res_blk = res_dfb.wait()
                    post_term_dfb.reserve().store(x_blk * post_bc)
                    matmul_dfb.reserve().store(comb_t @ res_blk)
                    out_dfb.reserve().store(
                        post_term_dfb.wait() + matmul_dfb.wait()
                    )

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    global_t = global_w // h_blocks
                    global_hb = global_w % h_blocks
                    h_start = global_hb * BH
                    ttl.copy(post_mix[global_t, 0], post_dfb.reserve()).wait()
                    ttl.copy(comb_T[global_t, 0], comb_dfb.reserve()).wait()
                    ttl.copy(
                        x[global_t:global_t + 1, h_start:h_start + BH],
                        x_dfb.reserve(),
                    ).wait()
                    ttl.copy(
                        residual[global_t:global_t + 1, h_start:h_start + BH],
                        res_dfb.reserve(),
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

    return post_kernel


def _make_rmsnorm_kernel(num_row_tiles: int, h_tiles: int,
                         rms_eps: float, inv_D: float,
                         *, h_block: int = 16):
    """Per row-tile rmsnorm. ssq accumulates across the row in BH-tile
    chunks, then inv_rms scales the row in matching BH-tile chunks.

    DFB shapes are (1, h_block); reduce_sum over dim=[1] collapses both
    within-tile cols and across the BH tiles in one shot.
    """
    if h_tiles % h_block:
        raise ValueError(
            f"h_tiles={h_tiles} not divisible by h_block={h_block}")
    h_blocks = h_tiles // h_block
    BH = h_block

    @ttl.operation(
        grid="auto",
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def rmsnorm_kernel(x, gamma, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_row_tiles // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)
        g_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, BH), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BH), block_count=2)
        sq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        xsq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BH), block_count=2)
        red_step_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BH), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            sc = sc_dfb.wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    x0 = x_dfb.wait()
                    xsq_dfb.reserve().store(x0 * x0)
                    sq_dfb.reserve().store(
                        ttl.math.reduce_sum(xsq_dfb.wait(), sc, dims=[1])
                    )
                    for _ in range(h_blocks - 1):
                        xk = x_dfb.wait()
                        xsq_dfb.reserve().store(xk * xk)
                        red_step_dfb.reserve().store(
                            ttl.math.reduce_sum(xsq_dfb.wait(), sc, dims=[1])
                        )
                        prev = sq_dfb.wait()
                        sq_dfb.reserve().store(prev + red_step_dfb.wait())
                    sq = sq_dfb.wait()
                    inv_bc = inv_bc_dfb.reserve()
                    inv_bc.store(ttl.math.broadcast(
                        ttl.math.rsqrt(
                            sq * ttl.math.fill(sq, inv_D)
                            + ttl.math.fill(sq, rms_eps)
                        ),
                        inv_bc, dims=[1],
                    ))
                    inv = inv_bc_dfb.wait()
                    for _ in range(h_blocks):
                        xk = x_dfb.wait()
                        gk = g_dfb.wait()
                        out_dfb.reserve().store(xk * gk * inv)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for hb in range(h_blocks):
                        h_start = hb * BH
                        ttl.copy(
                            x[global_t:global_t + 1, h_start:h_start + BH],
                            x_dfb.reserve(),
                        ).wait()
                    for hb in range(h_blocks):
                        h_start = hb * BH
                        ttl.copy(
                            x[global_t:global_t + 1, h_start:h_start + BH],
                            x_dfb.reserve(),
                        ).wait()
                        ttl.copy(
                            gamma[0:1, h_start:h_start + BH],
                            g_dfb.reserve(),
                        ).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for hb in range(h_blocks):
                        h_start = hb * BH
                        ttl.copy(
                            out_dfb.wait(),
                            out[global_t:global_t + 1, h_start:h_start + BH],
                        ).wait()

    return rmsnorm_kernel


def _make_summa_matmul_kernel(M: int, K: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Kp != 1:
        raise ValueError(f"K_parts must be 1, got {Kp}")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
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


def _make_hc_pre_state(mesh, hc_fn_cpu, hc_scale_cpu, hc_base_cpu):
    """Allocate the constant tensors for one hc_pre instance (attn or ffn)."""
    rep_fp32 = dict(device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    fn_packed = _mhc_pack_fn(
        hc_fn_cpu.detach().to(torch.float32).cpu(), MHC_MULT3)
    fn_tt = ttnn.as_tensor(fn_packed, **rep_fp32)
    (scale_tile, base_tile, pre_mask_tile, pre_eps_tile,
     post_mult_mask_tile, comb_mask_tile) = _mhc_build_split_constant_tiles(
        hc_scale_cpu.detach().to(torch.float32).cpu(),
        hc_base_cpu.detach().to(torch.float32).cpu(),
        MHC, _MHC_POST_MULT, HC_EPS,
    )
    return {
        "fn_tt": fn_tt,
        "scale_tt": ttnn.as_tensor(scale_tile, **rep_fp32),
        "base_tt": ttnn.as_tensor(base_tile, **rep_fp32),
        "pre_mask_tt": ttnn.as_tensor(pre_mask_tile, **rep_fp32),
        "pre_eps_tt": ttnn.as_tensor(pre_eps_tile, **rep_fp32),
        "post_mult_mask_tt": ttnn.as_tensor(post_mult_mask_tile, **rep_fp32),
        "comb_mask_tt": ttnn.as_tensor(comb_mask_tile, **rep_fp32),
    }


def make_lk_e_kernel(mesh, hc_attn_fn_cpu, hc_attn_scale_cpu, hc_attn_base_cpu,
                     hc_ffn_fn_cpu, hc_ffn_scale_cpu, hc_ffn_base_cpu,
                     ffn_norm_gamma_cpu):
    """Mega kernel for Lk-E: hc_pre_attn + hc_post_attn + hc_pre_ffn +
    ffn_norm + shared expert SwiGLU.

    See TODO at top: hc_pre_attn is here only for self-contained testing;
    in production Lk-A's stash provides post/comb_sk directly.
    """
    K_tiles = D // TILE
    h_tiles = DIM // TILE

    norm_fn_kernel = _make_mhc_norm_fn_ksplit_kernel(
        K_tiles=K_tiles, Kp=NORM_FN_KP,
        rms_eps=NORM_EPS, inv_D=1.0 / D)
    split_mixes_kernel = _make_mhc_split_mixes_kernel(num_tiles=1)
    sinkhorn_kernel = _make_mhc_sinkhorn_kernel(
        num_slices=NUM_TOKENS, repeat=HC_SINKHORN_ITERS, eps=HC_EPS)
    apply_mix_kernel = _make_mhc_apply_mix_h_kernel(
        num_tokens=NUM_TOKENS, h_tiles=h_tiles)
    post_kernel = _make_mhc_post_kernel(
        num_tokens=NUM_TOKENS, h_tiles=h_tiles)
    rms_kernel = _make_rmsnorm_kernel(
        num_row_tiles=1, h_tiles=h_tiles,
        rms_eps=NORM_EPS, inv_D=1.0 / DIM)
    matmul_w1 = _make_summa_matmul_kernel(
        M=TILE, K=DIM, N=INTER_DIM,
        block_cfg=(1, 4, 8), part_cfg=(1, 8, 1))
    matmul_w3 = _make_summa_matmul_kernel(
        M=TILE, K=DIM, N=INTER_DIM,
        block_cfg=(1, 4, 8), part_cfg=(1, 8, 1))
    matmul_w2 = _make_summa_matmul_kernel(
        M=TILE, K=INTER_DIM, N=DIM,
        block_cfg=(1, 4, 8), part_cfg=(1, 8, 1))

    rep_fp32 = dict(device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    rep_bf16 = dict(device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    attn_consts = _make_hc_pre_state(
        mesh, hc_attn_fn_cpu, hc_attn_scale_cpu, hc_attn_base_cpu)
    ffn_consts = _make_hc_pre_state(
        mesh, hc_ffn_fn_cpu, hc_ffn_scale_cpu, hc_ffn_base_cpu)

    sc_fp32_tt = ttnn.as_tensor(
        torch.ones((_MHC_TILE, _MHC_TILE), dtype=torch.float32), **rep_fp32)
    sc_bf16_tt = ttnn.as_tensor(
        torch.ones((TILE, TILE), dtype=torch.bfloat16), **rep_bf16)

    sk_mask_tt = ttnn.as_tensor(_mhc_sinkhorn_mask_tile(MHC), **rep_fp32)
    sk_eps_mask_tt = ttnn.as_tensor(
        _mhc_sinkhorn_eps_mask_tile(MHC, HC_EPS), **rep_fp32)

    gamma_packed = ffn_norm_gamma_cpu.flatten().to(torch.bfloat16) \
        .unsqueeze(0).expand(TILE, -1).contiguous()
    gamma_ffn_tt = ttnn.as_tensor(gamma_packed, **rep_bf16)

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

    def _run_hc_pre(consts, a_in, mixes_tt, pre_tt, post_tt, comb_tt,
                    comb_sk_out_tt, apply_mix_out_tt, a_sliced_scratch_tt):
        """Run hc_pre body on a_in [num_tokens_pad, D] fp32. Returns
        (post_tt_out, comb_sk_out_tt, apply_mix_out_tt). post_tt is the
        split_mixes output (post slot), comb_sk is the sinkhorn output."""
        norm_fn_kernel(a_in, consts["fn_tt"], sc_fp32_tt, mixes_tt)
        split_mixes_kernel(
            mixes_tt, consts["scale_tt"], consts["base_tt"],
            consts["pre_mask_tt"], consts["pre_eps_tt"],
            consts["post_mult_mask_tt"], consts["comb_mask_tt"],
            pre_tt, post_tt, comb_tt,
        )
        comb_sliced = ttnn.slice(
            comb_tt, [0, 2 * MHC], [NUM_TOKENS, 2 * MHC + MHC * MHC])
        comb_3d = ttnn.reshape(comb_sliced, [NUM_TOKENS, MHC, MHC])
        comb_padded = ttnn.pad(
            comb_3d,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, _MHC_TILE - MHC)],
            value=_MHC_PAD_SENTINEL)
        comb_sk_in_tt = ttnn.reshape(
            comb_padded, [NUM_TOKENS * _MHC_TILE, _MHC_TILE])
        sinkhorn_kernel(
            comb_sk_in_tt, sk_mask_tt, sk_eps_mask_tt, sc_fp32_tt,
            comb_sk_out_tt)
        pre_sliced = ttnn.slice(pre_tt, [0, 0], [NUM_TOKENS, MHC])
        pre_3d = ttnn.reshape(pre_sliced, [NUM_TOKENS, MHC, 1])
        mix_padded = ttnn.pad(
            pre_3d,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, _MHC_TILE - 1)],
            value=0.0)
        mix_tt = ttnn.reshape(
            mix_padded, [NUM_TOKENS * _MHC_TILE, _MHC_TILE])

        ttnn.slice(a_in, [0, 0], [NUM_TOKENS, MHC * DIM],
                   output_tensor=a_sliced_scratch_tt)
        a_3d = ttnn.reshape(a_sliced_scratch_tt, [NUM_TOKENS, MHC, DIM])
        x_padded = ttnn.pad(
            a_3d,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, 0)], value=0.0)
        x_tt = ttnn.reshape(x_padded, [NUM_TOKENS * _MHC_TILE, DIM])
        apply_mix_kernel(x_tt, mix_tt, sc_fp32_tt, apply_mix_out_tt)
        return post_tt, comb_sk_out_tt, apply_mix_out_tt

    def lk_e_kernel(attn_out_tt, prev_a_tt, w1_tt, w2_tt, w3_tt,
                    shared_partial_out, next_a_out,
                    norm_slice_out=None):
        if "scratch" not in state:
            # attn-side hc_pre scratch
            state["mixes_a"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["pre_a"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["post_a"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["comb_a"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["comb_sk_out_a"] = _zeros_fp32((NUM_TOKENS * _MHC_TILE, _MHC_TILE))
            state["apply_mix_out_a"] = _zeros_fp32((NUM_TOKENS * _MHC_TILE, DIM))
            state["a_sliced_scratch_a"] = _zeros_fp32((NUM_TOKENS, MHC * DIM))
            # ffn-side hc_pre scratch
            state["mixes_f"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["pre_f"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["post_f"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["comb_f"] = _zeros_fp32((NUM_TOKENS_PAD, _MHC_TILE))
            state["comb_sk_out_f"] = _zeros_fp32((NUM_TOKENS * _MHC_TILE, _MHC_TILE))
            state["apply_mix_out_f"] = _zeros_fp32((NUM_TOKENS * _MHC_TILE, DIM))
            state["a_sliced_scratch_f"] = _zeros_fp32((NUM_TOKENS, MHC * DIM))
            # post body
            state["x_post_tt"] = _zeros_fp32((NUM_TOKENS * _MHC_TILE, DIM))
            state["post_out_tt"] = _zeros_fp32((NUM_TOKENS * _MHC_TILE, DIM))
            # rmsnorm
            state["rms_in"] = _zeros_bf16((TILE, DIM))
            state["rms_out"] = _zeros_bf16((TILE, DIM))
            state["norm_slice"] = _zeros_bf16((NUM_TOKENS, DIM))
            # SwiGLU
            state["y1_tt"] = _zeros_bf16((TILE, INTER_DIM))
            state["y3_tt"] = _zeros_bf16((TILE, INTER_DIM))
            state["partial_pad"] = _zeros_bf16((TILE, DIM))
            state["scratch"] = True

        # TODO: claude: I don't see any reason this whole thing cannot be a single fused kernel, you should have all the primtives you need for reshape, slice, etc in datamovement and you can use element_read/write if needed. If you need to do some ttnn ceremony before or after the kernel that's OK.
        # 1. hc_pre_attn on prev_a -> populates post_a, comb_sk_out_a stash.
        post_attn_tt, comb_sk_attn_tt, _ = _run_hc_pre(
            attn_consts, prev_a_tt,
            state["mixes_a"], state["pre_a"], state["post_a"], state["comb_a"],
            state["comb_sk_out_a"], state["apply_mix_out_a"],
            state["a_sliced_scratch_a"],
        )

        # 2. ttnn glue: prep attn_out -> x_post_tt fp32 [num_tokens*TILE, hidden].
        x_3d = ttnn.reshape(attn_out_tt, [NUM_TOKENS, 1, DIM])
        x_repeated = ttnn.repeat(x_3d, ttnn.Shape([1, MHC, 1]))
        x_padded_3d = ttnn.pad(
            x_repeated,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, 0)], value=0.0)
        x_post_input = ttnn.reshape(
            x_padded_3d, [NUM_TOKENS * _MHC_TILE, DIM])
        ttnn.typecast(x_post_input, dtype=ttnn.float32,
                      output_tensor=state["x_post_tt"])

        # 3. hc_post_attn body: build res_tt, post_tile, comb_T tile, run kernel.
        ttnn.slice(prev_a_tt, [0, 0], [NUM_TOKENS, MHC * DIM],
                   output_tensor=state["a_sliced_scratch_a"])
        a_3d = ttnn.reshape(state["a_sliced_scratch_a"],
                            [NUM_TOKENS, MHC, DIM])
        res_padded = ttnn.pad(
            a_3d,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, 0)], value=0.0)
        res_tt = ttnn.reshape(res_padded, [NUM_TOKENS * _MHC_TILE, DIM])

        post_sliced = ttnn.slice(
            post_attn_tt, [0, MHC], [NUM_TOKENS, 2 * MHC])
        post_3d = ttnn.reshape(post_sliced, [NUM_TOKENS, MHC, 1])
        post_padded = ttnn.pad(
            post_3d,
            padding=[(0, 0), (0, _MHC_TILE - MHC), (0, _MHC_TILE - 1)],
            value=0.0)
        post_tile_tt = ttnn.reshape(
            post_padded, [NUM_TOKENS * _MHC_TILE, _MHC_TILE])

        comb_3d_sk = ttnn.reshape(
            comb_sk_attn_tt, [NUM_TOKENS, _MHC_TILE, _MHC_TILE])
        comb_T_3d = ttnn.transpose(comb_3d_sk, -2, -1)
        comb_T_tile_tt = ttnn.reshape(
            comb_T_3d, [NUM_TOKENS * _MHC_TILE, _MHC_TILE])

        post_kernel(state["x_post_tt"], res_tt, comb_T_tile_tt, post_tile_tt,
                    state["post_out_tt"])

        # 4. _mhc_post_to_a_tt: format hc_post output as next layer's residual.
        post_3d = ttnn.reshape(state["post_out_tt"],
                               [NUM_TOKENS, _MHC_TILE, DIM])
        post_sliced2 = ttnn.slice(post_3d, [0, 0, 0],
                                  [NUM_TOKENS, MHC, DIM])
        post_flat = ttnn.reshape(post_sliced2, [NUM_TOKENS, MHC * DIM])
        a_input_tt = ttnn.pad(
            post_flat,
            padding=[(0, NUM_TOKENS_PAD - NUM_TOKENS), (0, 0)], value=0.0)
        ttnn.copy(a_input_tt, next_a_out)

        # 5. hc_pre_ffn on a_input_tt -> apply_mix_out_f (fp32).
        _, _, ffn_hc_out_fp32 = _run_hc_pre(
            ffn_consts, a_input_tt,
            state["mixes_f"], state["pre_f"], state["post_f"], state["comb_f"],
            state["comb_sk_out_f"], state["apply_mix_out_f"],
            state["a_sliced_scratch_f"],
        )

        # 6. typecast fp32 -> bf16, rmsnorm.
        ttnn.typecast(ffn_hc_out_fp32, dtype=ttnn.bfloat16,
                      output_tensor=state["rms_in"])
        rms_kernel(state["rms_in"], gamma_ffn_tt, sc_bf16_tt,
                   state["rms_out"])
        ttnn.slice(state["rms_out"], [0, 0], [NUM_TOKENS, DIM],
                   output_tensor=state["norm_slice"])
        if norm_slice_out is not None:
            ttnn.copy(state["norm_slice"], norm_slice_out)

        # 7. shared expert SwiGLU.
        x_padded_2d = ttnn.pad(
            state["norm_slice"],
            padding=[(0, TILE - NUM_TOKENS), (0, 0)], value=0.0)

        matmul_w1(x_padded_2d, w1_tt, state["y1_tt"])
        matmul_w3(x_padded_2d, w3_tt, state["y3_tt"])

        y1_clamped = ttnn.clamp(state["y1_tt"], max=SWIGLU_LIMIT)
        y3_clamped = ttnn.clamp(
            state["y3_tt"], min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        silu_y1 = ttnn.silu(y1_clamped)
        mid = ttnn.multiply(silu_y1, y3_clamped)

        matmul_w2(mid, w2_tt, state["partial_pad"])
        partial_row = ttnn.slice(state["partial_pad"], [0, 0],
                                 [NUM_TOKENS, DIM])
        partial_3d = ttnn.reshape(partial_row, [1, NUM_TOKENS, DIM])
        ttnn.copy(partial_3d, shared_partial_out)

    return lk_e_kernel


def reference(mesh, attn_out_tt, prev_a_tt,
              hc_attn_fn_cpu, hc_attn_scale_cpu, hc_attn_base_cpu,
              hc_ffn_fn_cpu, hc_ffn_scale_cpu, hc_ffn_base_cpu,
              attn_norm_gamma_cpu, ffn_norm_gamma_cpu,
              w1_tt, w2_tt, w3_tt):
    mhc_attn = DeviceMHC(
        mesh=mesh, hc_fn=hc_attn_fn_cpu, hc_scale=hc_attn_scale_cpu,
        hc_base=hc_attn_base_cpu,
        hc_mult=MHC, hc_eps=HC_EPS,
        sinkhorn_iters=HC_SINKHORN_ITERS, norm_eps=NORM_EPS,
    )
    mhc_ffn = DeviceMHC(
        mesh=mesh, hc_fn=hc_ffn_fn_cpu, hc_scale=hc_ffn_scale_cpu,
        hc_base=hc_ffn_base_cpu,
        hc_mult=MHC, hc_eps=HC_EPS,
        sinkhorn_iters=HC_SINKHORN_ITERS, norm_eps=NORM_EPS,
    )
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=ffn_norm_gamma_cpu, eps=NORM_EPS)

    mhc_attn.hc_pre_device(NUM_TOKENS, NUM_TOKENS_PAD, a_tt=prev_a_tt)

    x_2d = ttnn.reshape(attn_out_tt, [NUM_TOKENS, 1, DIM])
    x_repeated = ttnn.repeat(x_2d, ttnn.Shape([1, MHC, 1]))
    x_padded = ttnn.pad(
        x_repeated,
        padding=[(0, 0), (0, _MHC_TILE - MHC), (0, 0)], value=0.0)
    x_post_input = ttnn.reshape(x_padded, [NUM_TOKENS * _MHC_TILE, DIM])
    ttnn.typecast(x_post_input, dtype=ttnn.float32,
                  output_tensor=mhc_attn._x_upload_tt)
    post_out_tt = mhc_attn.hc_post_device(NUM_TOKENS)

    a_input_tt = _mhc_post_to_a_tt(
        ttnn, post_out_tt, NUM_TOKENS, NUM_TOKENS_PAD, MHC, DIM)
    ffn_hc_out_fp32 = mhc_ffn.hc_pre_device(
        NUM_TOKENS, NUM_TOKENS_PAD, a_tt=a_input_tt)

    ttnn.typecast(ffn_hc_out_fp32, dtype=ttnn.bfloat16,
                  output_tensor=rmsn._x_upload_tt)
    ffn_norm_out_tt = rmsn.forward_device(rmsn._x_upload_tt, NUM_TOKENS)
    ttnn.slice(ffn_norm_out_tt, [0, 0], [NUM_TOKENS, DIM],
               output_tensor=mhc_ffn._norm_slice_tt)
    x_ffn_tt = mhc_ffn._norm_slice_tt

    x_3d = ttnn.reshape(x_ffn_tt, [1, NUM_TOKENS, DIM])
    y1 = ttnn.matmul(x_3d, w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y3 = ttnn.matmul(x_3d, w3_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if SWIGLU_LIMIT > 0:
        y1 = ttnn.clamp(y1, max=SWIGLU_LIMIT)
        y3 = ttnn.clamp(y3, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    silu = ttnn.silu(y1)
    mid = ttnn.multiply(silu, y3)
    partial = ttnn.matmul(mid, w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return partial, a_input_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        attn_out = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        prev_a = torch.randn(NUM_TOKENS_PAD, D, dtype=torch.float32) * 0.1
        hc_attn_fn = torch.randn(MHC_MULT3, D, dtype=torch.float32) * 0.05
        hc_attn_scale = torch.randn(3, dtype=torch.float32)
        hc_attn_base = torch.randn(MHC_MULT3, dtype=torch.float32) * 0.01
        hc_ffn_fn = torch.randn(MHC_MULT3, D, dtype=torch.float32) * 0.05
        hc_ffn_scale = torch.randn(3, dtype=torch.float32)
        hc_ffn_base = torch.randn(MHC_MULT3, dtype=torch.float32) * 0.01
        attn_norm_gamma = torch.ones(DIM, dtype=torch.bfloat16)
        ffn_norm_gamma = torch.ones(DIM, dtype=torch.bfloat16)
        w1 = torch.randn(INTER_DIM, DIM, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(DIM, INTER_DIM, dtype=torch.bfloat16) * 0.02
        w3 = torch.randn(INTER_DIM, DIM, dtype=torch.bfloat16) * 0.02
        w1_kn = w1.transpose(0, 1).contiguous()
        w3_kn = w3.transpose(0, 1).contiguous()
        w2_kn = w2.transpose(0, 1).contiguous()

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        attn_out_tt = ttnn.as_tensor(attn_out.contiguous(), dtype=ttnn.bfloat16, **rep)
        prev_a_tt = ttnn.as_tensor(prev_a.contiguous(), dtype=ttnn.float32, **rep)
        w1_tt = ttnn.as_tensor(w1_kn, dtype=ttnn.bfloat16, **rep)
        w2_tt = ttnn.as_tensor(w2_kn, dtype=ttnn.bfloat16, **rep)
        w3_tt = ttnn.as_tensor(w3_kn, dtype=ttnn.bfloat16, **rep)

        ref_partial_tt, ref_next_a_tt = reference(
            mesh, attn_out_tt, prev_a_tt,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            attn_norm_gamma, ffn_norm_gamma,
            w1_tt, w2_tt, w3_tt)
        ref_partial_host = download_chip0(mesh, mesh_shape, ref_partial_tt)
        ref_next_a_host = download_chip0(mesh, mesh_shape, ref_next_a_tt)

        kernel = make_lk_e_kernel(
            mesh,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            ffn_norm_gamma)
        partial_out_tt = ttnn.from_torch(
            torch.zeros(1, NUM_TOKENS, DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        next_a_tt = ttnn.from_torch(
            torch.zeros(NUM_TOKENS_PAD, D, dtype=torch.float32),
            dtype=ttnn.float32, **rep)
        kernel(attn_out_tt, prev_a_tt, w1_tt, w2_tt, w3_tt,
               partial_out_tt, next_a_tt)
        kernel_partial_host = download_chip0(mesh, mesh_shape, partial_out_tt)
        kernel_next_a_host = download_chip0(mesh, mesh_shape, next_a_tt)

        ok_p = report_pcc("Lk-E/shared", ref_partial_host, kernel_partial_host)
        ok_a = report_pcc("Lk-E/next_a", ref_next_a_host, kernel_next_a_host)

        benchmark("Lk-E ref",
                  lambda: reference(
                      mesh, attn_out_tt, prev_a_tt,
                      hc_attn_fn, hc_attn_scale, hc_attn_base,
                      hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                      attn_norm_gamma, ffn_norm_gamma,
                      w1_tt, w2_tt, w3_tt),
                  mesh)
        benchmark("Lk-E ttl",
                  lambda: kernel(attn_out_tt, prev_a_tt, w1_tt, w2_tt, w3_tt,
                                 partial_out_tt, next_a_tt),
                  mesh)

        sys.exit(0 if (ok_p and ok_a) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
