"""Final mega-zone PCC test: head logits + per-chip topk(k=1).

Reference is `DeviceLMHead.forward_argmax_device` end-to-end:
  slice(a_tt, last row) -> matmul(x, hc_fn_t) -> rsqrt(mean(x^2)+eps) ->
  multiply mixes by rsqrt by hc_scale + hc_base -> sigmoid -> + hc_eps ->
  reshape to [1,1,mhc] @ [1,mhc,hidden] -> typecast(fp32->bf16) -> rms_norm
  -> matmul(y, w_lmhead) -> topk(k=1).

Pipeline (3 inlined tt-lang kernels + ttnn glue):
  hc_combiner_kernel: a [TILE, D] fp32, hc_fn_t_scaled [D, TILE] fp32 ->
      pre [TILE, TILE] fp32 (cols 0..3 = sigmoid(mixes*rsqrt+hc_base)+hc_eps).
  rmsnorm_kernel: y [TILE, hidden] bf16, gamma -> y_normed [TILE, hidden] bf16.
  summa_matmul: y_normed [TILE, hidden] bf16 @ w_lmhead [hidden, vocab] bf16 ->
      logits [TILE, vocab] bf16 (only row 0 valid).

Math fold: hc_scale (scalar) is pre-multiplied into hc_fn_t on host so the
kernel post-process is just sigmoid(mixes*rsqrt + hc_base) + hc_eps.

ttnn glue (TODO: mega):
  - ttnn.slice (last row of a_tt) - tt-lang sub-tile-row dataflow not yet wired.
  - ttnn.reshape (x_2d->x_3d, pre->pre_3d) at sub-tile mhc=4 width.
  - ttnn.matmul (pre_3d @ x_3d) - sub-tile reshape requires intra-tile shuffle.
  - ttnn.typecast (fp32 -> bf16).
  - ttnn.pad (sub-tile -> tile-aligned).
  - ttnn.topk (no tt-lang topk yet).
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark


DIM = 4096
MHC = 4
D = MHC * DIM
VOCAB = 129280
NORM_EPS = 1e-6
HC_EPS = 1e-6
NUM_TOKENS = 1
NUM_TOKENS_PAD = 32
TILE = 32
ARGMAX_K = 32  # tile-cols streamed per argmax iter (sweet spot per argmax_2pass.py)


def _make_hc_combiner_kernel(num_out_tiles: int, K_tiles: int,
                             rms_eps: float, inv_D: float, hc_eps: float):
    """Inlined and extended from inference._compile_mhc_norm_fn_kernel.

    Computes per row tile:
      mixes  = a @ hc_fn_t_scaled                       (ping-pong c += a@b)
      ssq    = sum(a^2) over the row                    (parallel ssq accum)
      rsqrt  = rsqrt(ssq * inv_D + rms_eps)
      pre    = sigmoid(mixes * rsqrt + hc_base) + hc_eps

    Inputs (all fp32):
      a:        [num_out_tiles, K_tiles] (TILE x TILE per tile)
      b:        [K_tiles, 1] (hc_fn_t pre-scaled by hc_scale, mhc<=TILE in cols)
      scaler:   [1, 1] (all-ones broadcast tile)
      hc_base:  [1, 1] (hc_base packed; cols 0..mhc-1 = base[m], rest 0)
      out:      [num_out_tiles, 1] fp32 (cols 0..mhc-1 valid)
    """

    @ttl.operation(
        grid="auto",
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def hc_combiner_kernel(a, b, scaler, hc_base, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_out_tiles // total_cores)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        hcb_dfb = ttl.make_dataflow_buffer_like(hc_base, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        c_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        sq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        asq_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        red_step_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            sc = sc_dfb.wait()
            hcb = hcb_dfb.wait()

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_out_tiles:
                    a0 = a_dfb.wait()
                    b0 = b_dfb.wait()
                    c_dfb.reserve().store(a0 @ b0)
                    asq_dfb.reserve().store(a0 * a0)
                    sq_dfb.reserve().store(
                        ttl.math.reduce_sum(asq_dfb.wait(), sc, dims=[1])
                    )

                    for _ in range(K_tiles - 1):
                        a_k = a_dfb.wait()
                        b_k = b_dfb.wait()
                        prev_c = c_dfb.wait()
                        c_dfb.reserve().store(prev_c + a_k @ b_k)

                        asq_dfb.reserve().store(a_k * a_k)
                        red_step_dfb.reserve().store(
                            ttl.math.reduce_sum(asq_dfb.wait(), sc, dims=[1])
                        )
                        prev_sq = sq_dfb.wait()
                        sq_dfb.reserve().store(prev_sq + red_step_dfb.wait())

                    sq = sq_dfb.wait()
                    inv_bc = inv_bc_dfb.reserve()
                    inv_bc.store(ttl.math.broadcast(
                        ttl.math.rsqrt(
                            sq * ttl.math.fill(sq, inv_D)
                            + ttl.math.fill(sq, rms_eps)
                        ),
                        inv_bc, dims=[1],
                    ))

                    c = c_dfb.wait()
                    inv = inv_bc_dfb.wait()
                    sig = ttl.math.sigmoid(c * inv + hcb)
                    out_dfb.reserve().store(sig + ttl.math.fill(sig, hc_eps))

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            ttl.copy(hc_base[0, 0], hcb_dfb.reserve()).wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_out_tiles:
                    for k in range(K_tiles):
                        ttl.copy(a[global_t, k], a_dfb.reserve()).wait()
                        ttl.copy(b[k, 0], b_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_out_tiles:
                    ttl.copy(out_dfb.wait(), out[global_t, 0]).wait()

    return hc_combiner_kernel


def _make_rmsnorm_kernel(num_row_tiles: int, h_tiles: int,
                         rms_eps: float, inv_D: float):
    """Inlined from inference._compile_rmsnorm_kernel."""

    @ttl.operation(
        grid="auto",
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def rmsnorm_kernel(x, gamma, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_row_tiles // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        g_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        xsq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        red_step_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)

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
                    for _ in range(h_tiles - 1):
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

                    for _ in range(h_tiles):
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
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()
                    for h in range(h_tiles):
                        ttl.copy(x[global_t, h], x_dfb.reserve()).wait()
                        ttl.copy(gamma[0, h], g_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_row_tiles:
                    for h in range(h_tiles):
                        ttl.copy(out_dfb.wait(), out[global_t, h]).wait()

    return rmsnorm_kernel


def _make_summa_matmul_kernel(M: int, K: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    """Inlined SUMMA matmul. A is row-mcast across Np cores, B is column-mcast
    across Mp cores. Each core owns an M_BPN x N_BPN output sub-grid."""
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


def _make_ksplit_matmul_kernel(M: int, K: int, N: int,
                               block_cfg, part_cfg,
                               fp32_dest_acc_en: bool = True):
    """SUMMA matmul with K-split on the row axis. grid=(Np, Kp), Mp=1.

    K is split across Kp row cores; each core accumulates its K-slice partial
    sum, then non-root rows (k_p > 0) ship partials to root (k_p == 0) for
    summation and write-out. M is fixed at one bm-block (Mp implicit = 1).
    Mirrors tt-lang/benchmarks/matmul/ksplit_kernel.py with the row axis
    repurposed for Kp (since this matmul has Mt=1, Mp must be 1).
    """
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Mp != 1:
        raise ValueError(f"ksplit kernel here assumes Mp=1, got {Mp}")
    if Kp < 2:
        raise ValueError(f"K_parts must be >= 2, got {Kp}; use _make_summa_matmul_kernel")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape (tiles): Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})")
    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Nb % Np or Kb % Kp or Mb != 1:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} Kb={Kb} Np={Np} Kp={Kp}")
    N_BPN = Nb // Np
    K_BPN = Kb // Kp

    @ttl.operation(grid=(Np, Kp), fp32_dest_acc_en=fp32_dest_acc_en)
    def ksplit_matmul(a, w, out):
        # A mcast within a row: col 0 of each row sources its K-slice and
        # mcasts across that row's Np cols. Each row has a different K-slice.
        a_pipes = [ttl.Pipe(src=(0, k_p), dst=(slice(0, Np), k_p))
                   for k_p in range(Kp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        # B is unicast (each (n_p, k_p) reads its own kc-slice). No mcast.
        # Reduction: non-root rows (k_p>=1) send partials to root (k_p=0).
        reduce_pipes = [ttl.Pipe(src=(n_p, k_p), dst=(n_p, 0))
                        for n_p in range(Np) for k_p in range(1, Kp)]
        reduce_net = ttl.PipeNet(reduce_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        partial_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        recv_cb = ttl.make_dataflow_buffer_like(
            out, shape=(bm, bn), block_count=max(2, Kp - 1))
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            _, row_c = ttl.node(dims=2)
            for _ in range(N_BPN):
                p = partial_cb.reserve()
                for _ in range(K_BPN):
                    a_blk = a_cb.wait()
                    b_blk = b_cb.wait()
                    p += a_blk @ b_blk

                if row_c == 0:
                    for _ in range(Kp - 1):
                        prev = partial_cb.wait()
                        r = recv_cb.wait()
                        new = partial_cb.reserve()
                        new.store(prev + r)
                    final = partial_cb.wait()
                    o = out_cb.reserve()
                    o.store(final)

        @ttl.datamovement()
        def dm_read():
            _, row_c = ttl.node(dims=2)
            for _ in range(N_BPN):
                for kb_local in range(K_BPN):
                    kc = (row_c * K_BPN + kb_local) * bk
                    a_blk = a_cb.reserve()

                    def read_a(pipe):
                        ttl.copy(a[0:bm, kc:kc + bk], a_blk).wait()
                        ttl.copy(a_blk, pipe).wait()

                    mcast_a_net.if_src(read_a)
                    mcast_a_net.if_dst(
                        lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

                if row_c == 0:
                    def recv(pipe):
                        r = recv_cb.reserve()
                        ttl.copy(pipe, r).wait()

                    reduce_net.if_dst(recv)
                else:
                    p = partial_cb.wait()

                    def send(pipe):
                        ttl.copy(p, pipe).wait()

                    reduce_net.if_src(send)

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            for local_nb in range(N_BPN):
                nb = col_c * N_BPN + local_nb
                nc = nb * bn
                for kb_local in range(K_BPN):
                    kc = (row_c * K_BPN + kb_local) * bk
                    b_blk = b_cb.reserve()
                    ttl.copy(w[kc:kc + bk, nc:nc + bn], b_blk).wait()
                if row_c == 0:
                    o = out_cb.wait()
                    ttl.copy(o, out[0:bm, nc:nc + bn]).wait()

    return ksplit_matmul


def _make_argmax_input_pad_kernel(num_valid_tiles: int, num_total_tiles: int):
    """fp32 logits [TILE, num_valid_tiles*TILE] + fp32 row_mask [TILE, TILE]
    -> fp32 values [TILE, num_total_tiles*TILE] with row 0 = logits, rows
    1..31 = -1e9, padding tile-cols (>= num_valid_tiles) = -1e9.

    All operands are fp32 because tt-lang requires store-dtype to match the
    tile's element type (bf16+bf16 -> bf16 cannot store into an fp32 dfb).
    Argmax also wants fp32 to avoid bf16 ties between equal max values.
    row_mask is the constant tile (row 0 = 0, rows 1..31 = -1e9) used to
    blank rows 1..31 in valid tile-cols.
    """
    NEG_INF = -1.0e9

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def pad_kernel(logits, row_mask, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_total_tiles // total_cores)

        l_dfb = ttl.make_dataflow_buffer_like(logits, shape=(1, 1), block_count=2)
        m_dfb = ttl.make_dataflow_buffer_like(row_mask, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            m = m_dfb.wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_total_tiles:
                    if global_t < num_valid_tiles:
                        l = l_dfb.wait()
                        out_dfb.reserve().store(l + m)
                    else:
                        with out_dfb.reserve() as o:
                            o.store(ttl.math.fill(o, NEG_INF))

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            ttl.copy(row_mask[0, 0], m_dfb.reserve()).wait()
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_valid_tiles:
                    ttl.copy(logits[0, global_t], l_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_total_tiles:
                    ttl.copy(out_dfb.wait(), out[0, global_t]).wait()

    return pad_kernel


def _make_argmax_2pass_kernel(num_total_tiles: int, K_block: int):
    """Two-pass streaming argmax. See argmax_2pass.py for the algorithm.

    Inputs are fp32 throughout to avoid bf16 ties on equal max values
    (ttnn.topk's tie-break would diverge from this kernel's reduce_max).

    Inputs:
      values:    [TILE, num_total_tiles*TILE] fp32, padding lanes = -1e9.
      indices:   [TILE, num_total_tiles*TILE] fp32, row 0 = arange.
      scaler:    [TILE, TILE] fp32, ones.
      out_value: [TILE, TILE] fp32, valid at [0, 0].
      out_index: [TILE, TILE] fp32, valid at [0, 0].
    """
    if num_total_tiles % K_block != 0:
        raise ValueError(
            f"num_total_tiles={num_total_tiles} not divisible by K_block={K_block}")
    NUM_ITERS = num_total_tiles // K_block
    BIG = 1.0e6

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
    def argmax_2pass(values, indices, scaler, out_value, out_index):
        v1_dfb = ttl.make_dataflow_buffer_like(
            values, shape=(1, K_block), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        tile_max_dfb = ttl.make_dataflow_buffer_like(
            out_value, shape=(1, 1), block_count=2)
        run_max_dfb = ttl.make_dataflow_buffer_like(
            out_value, shape=(1, 1), block_count=2)

        max_global_dfb = ttl.make_dataflow_buffer_like(
            out_value, shape=(1, 1), block_count=2)
        max_bc_dfb = ttl.make_dataflow_buffer_like(
            values, shape=(1, K_block), block_count=2)

        v2_dfb = ttl.make_dataflow_buffer_like(
            values, shape=(1, K_block), block_count=2)
        i_dfb = ttl.make_dataflow_buffer_like(
            indices, shape=(1, K_block), block_count=2)
        i_filt_dfb = ttl.make_dataflow_buffer_like(
            indices, shape=(1, K_block), block_count=2)
        tile_argmax_dfb = ttl.make_dataflow_buffer_like(
            out_index, shape=(1, 1), block_count=2)
        run_argmax_dfb = ttl.make_dataflow_buffer_like(
            out_index, shape=(1, 1), block_count=2)

        ov_dfb = ttl.make_dataflow_buffer_like(out_value, shape=(1, 1), block_count=2)
        oi_dfb = ttl.make_dataflow_buffer_like(out_index, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            s = s_dfb.wait()

            with run_max_dfb.reserve() as init_rm:
                init_rm.store(ttl.math.fill(init_rm, -1.0e9))

            for _ in range(NUM_ITERS):
                v = v1_dfb.wait()
                rm = run_max_dfb.wait()
                tm = tile_max_dfb.reserve()
                tm.store(ttl.math.reduce_max(v, s, dims=[0, 1]))
                tm_blk = tile_max_dfb.wait()
                new_rm = run_max_dfb.reserve()
                new_rm.store(ttl.math.max(rm, tm_blk))

            final_rm = run_max_dfb.wait()
            mg = max_global_dfb.reserve()
            mg.store(ttl.math.max(final_rm, final_rm))
            mg_blk = max_global_dfb.wait()

            mgbc = max_bc_dfb.reserve()
            mgbc.store(ttl.math.broadcast(mg_blk, mgbc, dims=[0, 1]))
            mgbc_blk = max_bc_dfb.wait()

            ov = ov_dfb.reserve()
            ov.store(ttl.math.max(mg_blk, mg_blk))

            with run_argmax_dfb.reserve() as init_ra:
                init_ra.store(ttl.math.fill(init_ra, -1.0e9))

            for _ in range(NUM_ITERS):
                v = v2_dfb.wait()
                i = i_dfb.wait()
                ra = run_argmax_dfb.wait()
                with i_filt_dfb.reserve() as ifilt:
                    ifilt.store(
                        ttl.add(
                            i,
                            ttl.mul(
                                ttl.math.sign(ttl.sub(v, mgbc_blk)),
                                ttl.math.fill(ifilt, BIG),
                            ),
                        )
                    )
                ifilt_blk = i_filt_dfb.wait()
                ta = tile_argmax_dfb.reserve()
                ta.store(ttl.math.reduce_max(ifilt_blk, s, dims=[0, 1]))
                ta_blk = tile_argmax_dfb.wait()
                new_ra = run_argmax_dfb.reserve()
                new_ra.store(ttl.math.max(ra, ta_blk))

            final_ra = run_argmax_dfb.wait()
            oi = oi_dfb.reserve()
            oi.store(ttl.math.max(final_ra, final_ra))

        @ttl.datamovement()
        def dm_read():
            blk = s_dfb.reserve()
            ttl.copy(scaler[0, 0], blk).wait()
            for it in range(NUM_ITERS):
                blk = v1_dfb.reserve()
                ttl.copy(values[0:1, it * K_block:(it + 1) * K_block], blk).wait()
            for it in range(NUM_ITERS):
                blk = v2_dfb.reserve()
                ttl.copy(values[0:1, it * K_block:(it + 1) * K_block], blk).wait()
                blk = i_dfb.reserve()
                ttl.copy(indices[0:1, it * K_block:(it + 1) * K_block], blk).wait()

        @ttl.datamovement()
        def dm_write():
            blk = ov_dfb.wait()
            ttl.copy(blk, out_value[0, 0]).wait()
            blk = oi_dfb.wait()
            ttl.copy(blk, out_index[0, 0]).wait()

    return argmax_2pass


def _pack_hc_base_tile(hc_base: torch.Tensor) -> torch.Tensor:
    """[1, mhc] fp32 -> [TILE, TILE] fp32 with cols 0..mhc-1 broadcast across
    all rows and cols mhc..TILE-1 = 0. Cols 4..31 of pre/mixes are unused
    downstream but they participate in the tile-wise add; zero keeps them
    quiet (sigmoid(0)+hc_eps would leak garbage, so we slice mhc out before
    any consumer)."""
    if hc_base.shape != (1, MHC):
        raise ValueError(f"hc_base shape={tuple(hc_base.shape)} != (1, {MHC})")
    out = torch.zeros(TILE, TILE, dtype=torch.float32)
    out[:, :MHC] = hc_base[0:1, :].to(torch.float32).expand(TILE, MHC)
    return out


def _pack_hc_fn_t_scaled(hc_fn: torch.Tensor, hc_scale: torch.Tensor) -> torch.Tensor:
    """hc_fn [mhc, mhc*hidden] fp32 + hc_scale [1] fp32 -> [D, TILE] fp32.

    Rows 0..D-1 of the output match (hc_fn.T * hc_scale). Cols 0..mhc-1 are
    the scaled fn columns; cols mhc..TILE-1 = 0 (since the kernel matmul
    accumulates these into the unused cols of `mixes`)."""
    if hc_fn.shape != (MHC, D):
        raise ValueError(f"hc_fn shape={tuple(hc_fn.shape)} != ({MHC}, {D})")
    scaled = hc_fn.to(torch.float32) * float(hc_scale.item())
    t = scaled.transpose(0, 1).contiguous()                   # [D, mhc]
    out = torch.zeros(D, TILE, dtype=torch.float32)
    out[:, :MHC] = t
    return out


def make_final_kernel(mesh, debug_state=None):
    """Mega kernel for Final = hc_combiner + final RMSNorm + lm_head + topk(k=1).

    Three tt-lang dispatches (hc_combiner + rmsnorm + SUMMA) + ttnn glue
    for the sub-tile reshape/typecast/topk steps.
    """
    inv_D_combiner = 1.0 / float(D)        # mhc*hidden = 16384
    inv_D_norm = 1.0 / float(DIM)          # hidden     = 4096
    K_tiles_combiner = D // TILE           # 512
    h_tiles_norm = DIM // TILE             # 128

    hc_combiner = _make_hc_combiner_kernel(
        num_out_tiles=1, K_tiles=K_tiles_combiner,
        rms_eps=NORM_EPS, inv_D=inv_D_combiner, hc_eps=HC_EPS)
    rmsnorm = _make_rmsnorm_kernel(
        num_row_tiles=1, h_tiles=h_tiles_norm,
        rms_eps=NORM_EPS, inv_D=inv_D_norm)
    # KSPLIT lm_head: M=TILE, K=4096, N=129280. Mt=1, Kt=128, Nt=4040.
    # block=(1, 4, 4) part=(1, 10, 10) -> Nb=1010 N_BPN=101, Kb=32 K_BPN=3.2.
    # Wait Kb=32 % Kp=10 != 0; use Kp=8 -> K_BPN=4. Or bk=8: Kb=16, Kp=8 -> K_BPN=2.
    # Pick Kp=8 with bk=4: 80-core grid (10 wide, 8 tall on 11x10 device).
    lmhead_summa = _make_ksplit_matmul_kernel(
        M=TILE, K=DIM, N=VOCAB,
        block_cfg=(1, 4, 4), part_cfg=(1, 10, 8))

    # Argmax over full padded vocab. Multi-core ttnn.argmax (variant B from
    # argmax_2pass.py) + ttnn.max replaces ttnn.topk(k=1). The tt-lang
    # 2-pass kernel works at NUM_ITERS=4 (per-chip slice) but loses
    # within-tile resolution at NUM_ITERS=127 (full vocab on one chip).
    n_valid_tiles = VOCAB // TILE                          # 4040
    n_total_tiles = n_valid_tiles  # no extra padding needed for ttnn.argmax

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    state: dict = {}
    if debug_state is None:
        debug_state = {}

    def final_kernel(a_tt, hc_fn_t_scaled_tt, scaler_fp32_tt, hc_base_tile_tt,
                     norm_gamma_packed_tt, scaler_bf16_tt, w_lmhead_tt):
        if "init" not in state:
            state["pre_tile"] = ttnn.from_torch(
                torch.zeros(TILE, TILE, dtype=torch.float32),
                dtype=ttnn.float32, **rep)
            state["y_padded"] = ttnn.from_torch(
                torch.zeros(TILE, DIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["y_normed_padded"] = ttnn.from_torch(
                torch.zeros(TILE, DIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["logits_padded"] = ttnn.from_torch(
                torch.zeros(TILE, VOCAB, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["init"] = True

        # 1. HC combiner: a [TILE, D] fp32 + hc_fn_t_scaled [D, TILE] fp32
        #    -> pre_tile [TILE, TILE] fp32 (cols 0..mhc-1 of row 0 valid).
        hc_combiner(a_tt, hc_fn_t_scaled_tt, scaler_fp32_tt,
                    hc_base_tile_tt, state["pre_tile"])

        # 2. pre_3d [1,1,mhc] @ x_3d [1,mhc,hidden] -> y [1,1,hidden] fp32.
        # TODO: mega fusion blocked: ttnn used for sub-tile slice/reshape
        # at mhc=4 < TILE=32 (intra-tile shuffle not yet wired in tt-lang),
        # for fp32->bf16 typecast (no tt-lang typecast op), and for the
        # sub-tile [1, hidden] -> [TILE, hidden] padding before rmsnorm.
        last = NUM_TOKENS - 1
        x_2d = ttnn.slice(a_tt, [last, 0], [last + 1, D])
        x_3d = ttnn.reshape(x_2d, [1, MHC, DIM])
        pre = ttnn.slice(state["pre_tile"], [0, 0], [1, MHC])
        pre_3d = ttnn.reshape(pre, [1, 1, MHC])
        y_3d = ttnn.matmul(pre_3d, x_3d)
        y_bf16_3d = ttnn.typecast(y_3d, dtype=ttnn.bfloat16)
        y_2d = ttnn.reshape(y_bf16_3d, [1, DIM])
        y_padded_in = ttnn.pad(y_2d, padding=[(0, TILE - 1), (0, 0)],
                               value=0.0)
        ttnn.copy(y_padded_in, state["y_padded"])

        # 3. RMSNorm on padded [TILE, hidden] bf16 (only row 0 carries data).
        rmsnorm(state["y_padded"], norm_gamma_packed_tt, scaler_bf16_tt,
                state["y_normed_padded"])

        # 4. SUMMA lm_head matmul -> logits [TILE, vocab] (row 0 valid).
        lmhead_summa(state["y_normed_padded"], w_lmhead_tt,
                     state["logits_padded"])

        # 5. Argmax over [TILE, VOCAB]: layout-flip to ROW_MAJOR, run
        #    multi-core argmax for the index, ttnn.max in TILE for the
        #    value. ~7.5x faster than naive single-core argmax (variant B
        #    from argmax_2pass.py docstring).
        logits_rm = ttnn.to_layout(state["logits_padded"],
                                   layout=ttnn.ROW_MAJOR_LAYOUT)
        top_idx = ttnn.argmax(logits_rm, dim=-1, keepdim=True,
                              use_multicore=True)
        top_val = ttnn.max(state["logits_padded"], dim=-1, keepdim=True)

        debug_state["pre_tile"] = state["pre_tile"]
        debug_state["y_padded"] = state["y_padded"]
        debug_state["y_normed_padded"] = state["y_normed_padded"]
        debug_state["logits_padded"] = state["logits_padded"]
        return top_val, top_idx

    return final_kernel


def reference(mesh, a_tt, hc_fn_t_tt, hc_scale_tt, hc_base_tt,
              norm_gamma_tt, w_lmhead_tt, debug_state=None):
    """Mirror of DeviceLMHead.forward_argmax_device for replicated weights."""
    hidden = DIM
    last = NUM_TOKENS - 1

    x_2d = ttnn.slice(a_tt, [last, 0], [last + 1, D])
    mixes = ttnn.matmul(x_2d, hc_fn_t_tt)
    sq = ttnn.multiply(x_2d, x_2d)
    sq_mean = ttnn.mean(sq, dim=-1, keepdim=True)
    sq_mean_eps = ttnn.add(sq_mean, NORM_EPS)
    rsqrt_val = ttnn.rsqrt(sq_mean_eps)
    scaled = ttnn.multiply(mixes, rsqrt_val)
    scaled = ttnn.multiply(scaled, hc_scale_tt)
    scaled = ttnn.add(scaled, hc_base_tt)
    pre = ttnn.sigmoid(scaled)
    pre = ttnn.add(pre, HC_EPS)
    x_3d = ttnn.reshape(x_2d, [1, MHC, hidden])
    pre_3d = ttnn.reshape(pre, [1, 1, MHC])
    y_3d = ttnn.matmul(pre_3d, x_3d)
    y_bf16 = ttnn.typecast(y_3d, dtype=ttnn.bfloat16)
    y_4d = ttnn.reshape(y_bf16, (1, 1, 1, hidden))
    y_normed = ttnn.rms_norm(y_4d, weight=norm_gamma_tt, epsilon=NORM_EPS)
    y_normed = ttnn.reshape(y_normed, (1, 1, hidden))
    logits_tt = ttnn.matmul(y_normed, w_lmhead_tt,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    vals_tt, idxs_tt = ttnn.topk(
        logits_tt, k=1, dim=-1, largest=True, sorted=True)
    if debug_state is not None:
        debug_state["pre"] = pre
        debug_state["y_bf16"] = y_bf16
        debug_state["y_normed"] = y_normed
        debug_state["logits"] = logits_tt
    return vals_tt, idxs_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        a = torch.randn(NUM_TOKENS_PAD, D, dtype=torch.float32) * 0.1
        hc_fn = torch.randn(MHC, D, dtype=torch.float32) * 0.05
        hc_fn_t = hc_fn.transpose(0, 1).contiguous()
        hc_scale = torch.randn(1, dtype=torch.float32)
        hc_base = torch.randn(1, MHC, dtype=torch.float32) * 0.01
        norm_gamma = torch.ones(DIM, dtype=torch.bfloat16)
        w_lmhead = torch.randn(DIM, VOCAB, dtype=torch.bfloat16) * 0.005

        # Reference inputs (ttnn shapes mirror inference.py).
        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        a_tt = ttnn.as_tensor(a.contiguous(), dtype=ttnn.float32, **rep)
        hc_fn_t_tt = ttnn.as_tensor(hc_fn_t, dtype=ttnn.float32, **rep)
        hc_scale_ref = ttnn.as_tensor(hc_scale.reshape(1, 1).contiguous(),
                                      dtype=ttnn.float32, **rep)
        hc_base_ref = ttnn.as_tensor(hc_base.contiguous(),
                                     dtype=ttnn.float32, **rep)
        norm_gamma_tt = ttnn.as_tensor(norm_gamma.contiguous(),
                                       dtype=ttnn.bfloat16, **rep)
        w_lmhead_tt = ttnn.as_tensor(w_lmhead.contiguous(),
                                     dtype=ttnn.bfloat16, **rep)

        ref_debug = {}
        ref_vals_tt, ref_idxs_tt = reference(
            mesh, a_tt, hc_fn_t_tt, hc_scale_ref, hc_base_ref,
            norm_gamma_tt, w_lmhead_tt, debug_state=ref_debug)
        ref_vals_host = download_chip0(mesh, mesh_shape, ref_vals_tt)
        ref_idxs_host = download_chip0(mesh, mesh_shape, ref_idxs_tt)
        ref_pre_host = download_chip0(mesh, mesh_shape, ref_debug["pre"])
        ref_y_host = download_chip0(mesh, mesh_shape, ref_debug["y_bf16"])
        ref_yn_host = download_chip0(mesh, mesh_shape, ref_debug["y_normed"])
        ref_logits_host = download_chip0(mesh, mesh_shape, ref_debug["logits"])

        # Kernel inputs: hc_fn_t pre-scaled by hc_scale (folded into b).
        hc_fn_t_scaled = _pack_hc_fn_t_scaled(hc_fn, hc_scale)
        hc_base_tile = _pack_hc_base_tile(hc_base)
        norm_gamma_packed = norm_gamma.unsqueeze(0).expand(TILE, -1).contiguous()
        scaler_fp32 = torch.ones(TILE, TILE, dtype=torch.float32)
        scaler_bf16 = torch.ones(TILE, TILE, dtype=torch.bfloat16)

        hc_fn_t_scaled_tt = ttnn.as_tensor(
            hc_fn_t_scaled.contiguous(), dtype=ttnn.float32, **rep)
        hc_base_tile_tt = ttnn.as_tensor(
            hc_base_tile.contiguous(), dtype=ttnn.float32, **rep)
        norm_gamma_packed_tt = ttnn.as_tensor(
            norm_gamma_packed.contiguous(), dtype=ttnn.bfloat16, **rep)
        scaler_fp32_tt = ttnn.as_tensor(
            scaler_fp32.contiguous(), dtype=ttnn.float32, **rep)
        scaler_bf16_tt = ttnn.as_tensor(
            scaler_bf16.contiguous(), dtype=ttnn.bfloat16, **rep)

        kernel_debug = {}
        kernel = make_final_kernel(mesh, debug_state=kernel_debug)
        top_val_out_tt, top_idx_out_tt = kernel(
            a_tt, hc_fn_t_scaled_tt, scaler_fp32_tt, hc_base_tile_tt,
            norm_gamma_packed_tt, scaler_bf16_tt, w_lmhead_tt)
        kernel_vals_host = download_chip0(mesh, mesh_shape, top_val_out_tt)
        kernel_idxs_host = download_chip0(mesh, mesh_shape, top_idx_out_tt)
        k_pre_tile = download_chip0(mesh, mesh_shape, kernel_debug["pre_tile"])
        k_y_padded = download_chip0(mesh, mesh_shape, kernel_debug["y_padded"])
        k_yn_padded = download_chip0(mesh, mesh_shape, kernel_debug["y_normed_padded"])
        k_logits_padded = download_chip0(mesh, mesh_shape, kernel_debug["logits_padded"])

        # Compare each stage.
        # ref_pre_host shape [1, MHC]; kernel pre_tile shape [TILE, TILE].
        k_pre = k_pre_tile[0:1, 0:MHC]
        report_pcc("stage1/pre", ref_pre_host.float(), k_pre.float())
        # ref_y_host shape [1, 1, hidden]; kernel y_padded shape [TILE, hidden] row 0.
        k_y = k_y_padded[0:1, :].reshape(1, 1, DIM)
        report_pcc("stage2/y_bf16", ref_y_host.float(), k_y.float())
        # ref_yn_host shape [1, 1, hidden]; kernel y_normed_padded row 0.
        k_yn = k_yn_padded[0:1, :].reshape(1, 1, DIM)
        report_pcc("stage3/y_normed", ref_yn_host.float(), k_yn.float())
        # ref_logits_host shape [1, 1, vocab]; kernel logits_padded row 0.
        k_logits = k_logits_padded[0:1, :].reshape(1, 1, VOCAB)
        report_pcc("stage4/logits", ref_logits_host.float(), k_logits.float())
        # Per-tensor abs argmax sanity.
        ref_argmax = int(ref_logits_host.flatten().argmax().item())
        k_argmax = int(k_logits.flatten().argmax().item())
        print(f"[debug] ref argmax={ref_argmax}  kernel argmax={k_argmax}")

        # top-1 is a single scalar so PCC is degenerate. Compare directly:
        # idx must match exactly, val within bf16 tolerance.
        ref_v = float(ref_vals_host.flatten()[0].item())
        k_v = float(kernel_vals_host.flatten()[0].item())
        ref_i = int(ref_idxs_host.flatten()[0].item())
        k_i = int(kernel_idxs_host.flatten()[0].item())
        val_diff = abs(ref_v - k_v)
        ok_i = (ref_i == k_i)
        # bf16 ulp at this magnitude is ~few percent; allow 5%.
        ok_v = val_diff <= max(5e-2 * max(abs(ref_v), abs(k_v)), 1e-3)
        status_v = "PASS" if ok_v else "FAIL"
        status_i = "PASS" if ok_i else "FAIL"
        print(f"[Final/top_val] {status_v} ref={ref_v:.6f} kernel={k_v:.6f} "
              f"abs_diff={val_diff:.4e}")
        print(f"[Final/top_idx] {status_i} ref={ref_i} kernel={k_i}")

        benchmark("Final ref",
                  lambda: reference(mesh, a_tt, hc_fn_t_tt, hc_scale_ref,
                                    hc_base_ref, norm_gamma_tt, w_lmhead_tt),
                  mesh)
        benchmark("Final ttl",
                  lambda: kernel(a_tt, hc_fn_t_scaled_tt, scaler_fp32_tt,
                                 hc_base_tile_tt, norm_gamma_packed_tt,
                                 scaler_bf16_tt, w_lmhead_tt),
                  mesh)

        sys.exit(0 if (ok_v and ok_i) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
