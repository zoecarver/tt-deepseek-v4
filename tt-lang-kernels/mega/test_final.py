"""Final mega-zone PCC test: head logits + per-chip topk(k=1).

Reference is `DeviceLMHead.forward_argmax_device` end-to-end:
  slice(a_tt, last row) -> matmul(x, hc_fn_t) -> rsqrt(mean(x^2)+eps) ->
  multiply mixes by rsqrt by hc_scale + hc_base -> sigmoid -> + hc_eps ->
  reshape to [1,1,mhc] @ [1,mhc,hidden] -> typecast(fp32->bf16) -> rms_norm
  -> matmul(y, w_lmhead) -> topk(k=1).

Pipeline (3 inlined tt-lang kernels + ttnn glue):
  hc_combiner_kernel: a [TILE, D] bf16, hc_fn_t_scaled [D, TILE] bf16 ->
      pre [TILE, TILE] bf16 (cols 0..3 = sigmoid(mixes*rsqrt+hc_base)+hc_eps).
  rmsnorm_kernel: y [TILE, hidden] bf16, gamma -> y_normed [TILE, hidden] bf16.
  ksplit_matmul: y_normed [TILE, hidden] bf16 @ w_lmhead [hidden, vocab] bf16 ->
      logits [TILE, vocab] bf16 (only row 0 valid).

Math fold: hc_scale (scalar) is pre-multiplied into hc_fn_t on host so the
kernel post-process is just sigmoid(mixes*rsqrt + hc_base) + hc_eps.

a_tt is demoted to bf16 at offload (caller's responsibility) — the kernel
runs end-to-end in bf16 with no typecast boundary between combiner and
rmsnorm.

ttnn glue (TODO: mega):
  - ttnn.slice (last row of a_tt) - tt-lang sub-tile-row dataflow not yet wired.
  - ttnn.reshape (x_2d->x_3d, pre->pre_3d) at sub-tile mhc=4 width.
  - ttnn.matmul (pre_3d @ x_3d) - sub-tile reshape requires intra-tile shuffle.
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


def _make_hc_combiner_kernel(num_out_tiles: int, K_tiles: int,
                             rms_eps: float, inv_D: float, hc_eps: float,
                             *, bk: int = 64):
    """Inlined and extended from inference._compile_mhc_norm_fn_kernel.

    a_dfb shape=(1, bk), b_dfb shape=(bk, 1): per-core K-loop runs
    K_NB=K_tiles/bk SUMMA-style accumulations. xsq matches at (1, bk)
    and reduce_sum dim=[1] collapses both within-tile cols and across
    bk tile-cols. bk=2 is the max stable; bk=4 hits PCC due to DST
    register pressure with fp32_dest_acc_en.
    """
    if K_tiles % bk:
        raise ValueError(f"K_tiles={K_tiles} not divisible by bk={bk}")
    K_NB = K_tiles // bk

    @ttl.operation(
        grid="auto",
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=True,
    )
    def hc_combiner_kernel(a, b, scaler, hc_base, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        tiles_per_core = -(-num_out_tiles // total_cores)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, bk), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(bk, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        hcb_dfb = ttl.make_dataflow_buffer_like(hc_base, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        c_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        sq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        asq_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, bk), block_count=2)
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

                    for _ in range(K_NB - 1):
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
                    for kb_local in range(K_NB):
                        kc = kb_local * bk
                        ttl.copy(
                            a[global_t:global_t + 1, kc:kc + bk],
                            a_dfb.reserve(),
                        ).wait()
                        ttl.copy(
                            b[kc:kc + bk, 0:1],
                            b_dfb.reserve(),
                        ).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_t in range(tiles_per_core):
                global_t = core_idx * tiles_per_core + local_t
                if global_t < num_out_tiles:
                    ttl.copy(out_dfb.wait(), out[global_t, 0]).wait()

    return hc_combiner_kernel


def _make_fused_rms_ksplit_kernel(M: int, K: int, N: int,
                                  block_cfg, part_cfg,
                                  rms_eps: float,
                                  fp32_dest_acc_en: bool = True):
    """Fused rmsnorm(x, gamma) @ Wg as a single ttl.operation.

    Same lowering as test_lk_b's fused kernel. Wg = gamma[:, None] * W is
    pre-baked on host. Each (n_p, k_p) core accumulates its K_BPN matmul
    partial AND its K_BPN ssq partial. The ssq is reduced across k_p via a
    parallel PipeNet; only k_p=0 finalizes inv = rsqrt(ssq/D + eps), applies
    it to the reduced matmul output, and writes. Mp = 1.

    Constrained to K_BPN = 1 here because the only known-good test_lk_b
    config is K_BPN=1; K_BPN > 1 cases hung at dispatch when tried.
    """
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Mp != 1:
        raise ValueError(f"fused ksplit assumes Mp=1, got {Mp}")
    if Kp < 2:
        raise ValueError(f"K_parts must be >= 2, got {Kp}")
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
    if K_BPN != 1:
        raise ValueError(
            f"fused rms+ksplit only supports K_BPN=1 (got {K_BPN}); pick "
            f"bk so Kt/bk == Kp")
    inv_D = 1.0 / float(K)

    @ttl.operation(
        grid=(Np, Kp),
        fp32_dest_acc_en=fp32_dest_acc_en,
        options="--no-ttl-reduce-full-fp32",
    )
    def fused_kernel(a, w_g, scaler, out):
        a_pipes = [ttl.Pipe(src=(0, k_p), dst=(slice(0, Np), k_p))
                   for k_p in range(Kp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        reduce_pipes = [ttl.Pipe(src=(n_p, k_p), dst=(n_p, 0))
                        for n_p in range(Np) for k_p in range(1, Kp)]
        reduce_net = ttl.PipeNet(reduce_pipes)
        ssq_reduce_pipes = [ttl.Pipe(src=(n_p, k_p), dst=(n_p, 0))
                            for n_p in range(Np) for k_p in range(1, Kp)]
        ssq_reduce_net = ttl.PipeNet(ssq_reduce_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w_g, shape=(bk, bn), block_count=2)
        partial_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        recv_cb = ttl.make_dataflow_buffer_like(
            out, shape=(bm, bn), block_count=max(2, Kp - 1))
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        xsq_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        ssq_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, 1), block_count=2)
        ssq_recv_cb = ttl.make_dataflow_buffer_like(
            a, shape=(bm, 1), block_count=max(2, Kp - 1))
        inv_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, 1), block_count=2)
        inv_bc_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            _, row_c = ttl.node(dims=2)
            sc = sc_cb.wait()

            # First N output block: matmul partial + ssq partial fused.
            a0 = a_cb.wait()
            b0 = b_cb.wait()
            partial_cb.reserve().store(a0 @ b0)
            xsq_cb.reserve().store(a0 * a0)
            ssq_cb.reserve().store(
                ttl.math.reduce_sum(xsq_cb.wait(), sc, dims=[1])
            )

            if row_c == 0:
                # Cross-row reduce of acc partials.
                for _ in range(Kp - 1):
                    prev = partial_cb.wait()
                    r = recv_cb.wait()
                    partial_cb.reserve().store(prev + r)
                # Cross-row reduce of ssq partials.
                for _ in range(Kp - 1):
                    prev_ssq = ssq_cb.wait()
                    r_ssq = ssq_recv_cb.wait()
                    ssq_cb.reserve().store(prev_ssq + r_ssq)
                # Finalize inv_rms.
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

                acc_done = partial_cb.wait()
                out_cb.reserve().store(acc_done * inv)

                # Subsequent N blocks: matmul + reduce + apply inv.
                for _ in range(N_BPN - 1):
                    a0n = a_cb.wait()
                    b0n = b_cb.wait()
                    partial_cb.reserve().store(a0n @ b0n)
                    for _ in range(Kp - 1):
                        prev = partial_cb.wait()
                        r = recv_cb.wait()
                        partial_cb.reserve().store(prev + r)
                    acc_n = partial_cb.wait()
                    out_cb.reserve().store(acc_n * inv)
            else:
                # Non-root: produce remaining N_BPN-1 partial matmuls; the
                # first was already produced above and dm_read sends each.
                for _ in range(N_BPN - 1):
                    a0n = a_cb.wait()
                    b0n = b_cb.wait()
                    partial_cb.reserve().store(a0n @ b0n)

        @ttl.datamovement()
        def dm_read():
            ttl.copy(scaler[0, 0], sc_cb.reserve()).wait()
            _, row_c = ttl.node(dims=2)

            for n_idx in range(N_BPN):
                kc = row_c * bk
                a_blk = a_cb.reserve()

                def read_a(pipe):
                    ttl.copy(a[0:bm, kc:kc + bk], a_blk).wait()
                    ttl.copy(a_blk, pipe).wait()

                mcast_a_net.if_src(read_a)
                mcast_a_net.if_dst(
                    lambda pipe: (ttl.copy(pipe, a_blk).wait(),))

                if row_c == 0:
                    def recv_acc(pipe):
                        r = recv_cb.reserve()
                        ttl.copy(pipe, r).wait()
                    reduce_net.if_dst(recv_acc)
                else:
                    p = partial_cb.wait()

                    def send_acc(pipe):
                        ttl.copy(p, pipe).wait()
                    reduce_net.if_src(send_acc)

                if n_idx == 0:
                    if row_c == 0:
                        def recv_ssq(pipe):
                            r = ssq_recv_cb.reserve()
                            ttl.copy(pipe, r).wait()
                        ssq_reduce_net.if_dst(recv_ssq)
                    else:
                        ssq = ssq_cb.wait()

                        def send_ssq(pipe):
                            ttl.copy(ssq, pipe).wait()
                        ssq_reduce_net.if_src(send_ssq)

        @ttl.datamovement()
        def dm_write():
            col_c, row_c = ttl.node(dims=2)
            for local_nb in range(N_BPN):
                nb = col_c * N_BPN + local_nb
                nc = nb * bn
                kc = row_c * bk
                b_blk = b_cb.reserve()
                ttl.copy(w_g[kc:kc + bk, nc:nc + bn], b_blk).wait()
                if row_c == 0:
                    o = out_cb.wait()
                    ttl.copy(o, out[0:bm, nc:nc + bn]).wait()

    return fused_kernel


def _pack_hc_base_tile(hc_base: torch.Tensor) -> torch.Tensor:
    """[1, mhc] fp32 -> [TILE, TILE] bf16 with cols 0..mhc-1 broadcast across
    all rows and cols mhc..TILE-1 = 0. Cols 4..31 of pre/mixes are unused
    downstream but they participate in the tile-wise add; zero keeps them
    quiet (sigmoid(0)+hc_eps would leak garbage, so we slice mhc out before
    any consumer)."""
    if hc_base.shape != (1, MHC):
        raise ValueError(f"hc_base shape={tuple(hc_base.shape)} != (1, {MHC})")
    out = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    out[:, :MHC] = hc_base[0:1, :].to(torch.bfloat16).expand(TILE, MHC)
    return out


def _pack_hc_fn_t_scaled(hc_fn: torch.Tensor, hc_scale: torch.Tensor) -> torch.Tensor:
    """hc_fn [mhc, mhc*hidden] fp32 + hc_scale [1] fp32 -> [D, TILE] bf16.

    Rows 0..D-1 of the output match (hc_fn.T * hc_scale). Cols 0..mhc-1 are
    the scaled fn columns; cols mhc..TILE-1 = 0 (since the kernel matmul
    accumulates these into the unused cols of `mixes`)."""
    if hc_fn.shape != (MHC, D):
        raise ValueError(f"hc_fn shape={tuple(hc_fn.shape)} != ({MHC}, {D})")
    scaled = hc_fn.to(torch.float32) * float(hc_scale.item())
    t = scaled.transpose(0, 1).contiguous()                   # [D, mhc]
    out = torch.zeros(D, TILE, dtype=torch.bfloat16)
    out[:, :MHC] = t.to(torch.bfloat16)
    return out


def make_final_kernel(mesh, debug_state=None):
    """Mega kernel for Final = hc_combiner + fused(RMSNorm + lm_head) + topk(k=1).

    Two tt-lang dispatches (hc_combiner + fused-rms-ksplit) + ttnn glue
    for the sub-tile reshape/typecast/topk steps.
    """
    inv_D_combiner = 1.0 / float(D)        # mhc*hidden = 16384
    K_tiles_combiner = D // TILE           # 512

    hc_combiner = _make_hc_combiner_kernel(
        num_out_tiles=1, K_tiles=K_tiles_combiner,
        rms_eps=NORM_EPS, inv_D=inv_D_combiner, hc_eps=HC_EPS)
    # Fused rmsnorm + KSPLIT lm_head. M=TILE, K=4096, N=129280.
    # Mt=1, Kt=128, Nt=4040. block=(1, 4, 16) part=(1, 10, 8) -> Nb=1010,
    # Kb=8, K_BPN=1 (80 cores). Wg = gamma[:, None] * W_lmhead pre-baked
    # on host. K_BPN=1 is required (see factory docstring).
    fused_rms_ksplit = _make_fused_rms_ksplit_kernel(
        M=TILE, K=DIM, N=VOCAB,
        block_cfg=(1, 4, 16), part_cfg=(1, 10, 8),
        rms_eps=NORM_EPS)

    # Argmax over full padded vocab. Multi-core ttnn.argmax (variant B from
    # argmax_2pass.py) + ttnn.max replaces ttnn.topk(k=1).
    # TODO: mega fusion blocked (bucket #5 — primitive): single-stage
    # argmax_2pass cannot encode indices in [0, VOCAB=129280) into the
    # `i + sign(v - max) * BIG` trick because bf16's 8-bit mantissa only
    # holds integers exactly up to 256, and fp32 reduce_max is broken
    # upstream. Fix is hierarchical argmax (3 stages of ≤256). See README
    # "What's actually unwired vs primitive-blocked", bucket #5. Punted.
    n_valid_tiles = VOCAB // TILE                          # 4040
    n_total_tiles = n_valid_tiles  # no extra padding needed for ttnn.argmax

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    state: dict = {}
    if debug_state is None:
        debug_state = {}

    def final_kernel(a_bf16_tt, hc_fn_t_scaled_tt, hc_base_tile_tt,
                     scaler_bf16_tt, w_baked_tt):
        if "init" not in state:
            state["pre_tile"] = ttnn.from_torch(
                torch.zeros(TILE, TILE, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["y_padded"] = ttnn.from_torch(
                torch.zeros(TILE, DIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["logits_padded"] = ttnn.from_torch(
                torch.zeros(TILE, VOCAB, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["init"] = True

        # 1. HC combiner: a [TILE, D] bf16 + hc_fn_t_scaled [D, TILE] bf16
        #    -> pre_tile [TILE, TILE] bf16 (cols 0..mhc-1 of row 0 valid).
        # TODO: claude: fuse hc_combiner + rmsnorm + lmhead_summa via cross-row
        # PipeNet (broadcast pre to all 80 cores in the lmhead grid).
        hc_combiner(a_bf16_tt, hc_fn_t_scaled_tt, scaler_bf16_tt,
                    hc_base_tile_tt, state["pre_tile"])

        # 2. pre_3d [1,1,mhc] @ x_3d [1,mhc,hidden] -> y [1,1,hidden] bf16.
        # TODO: mega fusion blocked: ttnn used for sub-tile slice/reshape
        # at mhc=4 < TILE=32 (intra-tile shuffle not yet wired in tt-lang)
        # and for the sub-tile [1, hidden] -> [TILE, hidden] padding before
        # rmsnorm. Now that everything is bf16 the explicit typecast is gone.
        last = NUM_TOKENS - 1
        x_2d = ttnn.slice(a_bf16_tt, [last, 0], [last + 1, D])
        x_3d = ttnn.reshape(x_2d, [1, MHC, DIM])
        pre = ttnn.slice(state["pre_tile"], [0, 0], [1, MHC])
        pre_3d = ttnn.reshape(pre, [1, 1, MHC])
        y_3d = ttnn.matmul(pre_3d, x_3d)
        y_2d = ttnn.reshape(y_3d, [1, DIM])
        y_padded_in = ttnn.pad(y_2d, padding=[(0, TILE - 1), (0, 0)],
                               value=0.0)
        ttnn.copy(y_padded_in, state["y_padded"])

        # 3. Fused RMSNorm + KSPLIT lm_head: gamma is pre-baked into
        #    w_baked, and the kernel emits logits [TILE, vocab] directly
        #    (row 0 valid).
        fused_rms_ksplit(state["y_padded"], w_baked_tt, scaler_bf16_tt,
                         state["logits_padded"])

        # 4. Argmax over [TILE, VOCAB]: layout-flip to ROW_MAJOR, run
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
    # Larger kernel_config_extra_bytes: the fused rms+ksplit program
    # exceeds the default ~70KB kernel-config buffer (see test_lk_b).
    mesh = open_mesh(kernel_config_extra_bytes=128 * 1024)
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
        # bf16 across the board: a_tt demoted at offload so the combiner
        # reads bf16 and the prologue matmul output stays bf16 (no typecast).
        # Pre-bake norm_gamma into w_lmhead so the fused rms+ksplit kernel
        # treats gamma as part of the weight (Wg[k,n] = gamma[k]*W[k,n]).
        a_bf16 = a.to(torch.bfloat16)
        hc_fn_t_scaled = _pack_hc_fn_t_scaled(hc_fn, hc_scale)
        hc_base_tile = _pack_hc_base_tile(hc_base)
        scaler_bf16 = torch.ones(TILE, TILE, dtype=torch.bfloat16)
        w_baked = (norm_gamma.float()[:, None] * w_lmhead.float()) \
            .to(torch.bfloat16).contiguous()

        a_bf16_tt = ttnn.as_tensor(
            a_bf16.contiguous(), dtype=ttnn.bfloat16, **rep)
        hc_fn_t_scaled_tt = ttnn.as_tensor(
            hc_fn_t_scaled.contiguous(), dtype=ttnn.bfloat16, **rep)
        hc_base_tile_tt = ttnn.as_tensor(
            hc_base_tile.contiguous(), dtype=ttnn.bfloat16, **rep)
        scaler_bf16_tt = ttnn.as_tensor(
            scaler_bf16.contiguous(), dtype=ttnn.bfloat16, **rep)
        w_baked_tt = ttnn.as_tensor(
            w_baked, dtype=ttnn.bfloat16, **rep)

        kernel_debug = {}
        kernel = make_final_kernel(mesh, debug_state=kernel_debug)
        top_val_out_tt, top_idx_out_tt = kernel(
            a_bf16_tt, hc_fn_t_scaled_tt, hc_base_tile_tt,
            scaler_bf16_tt, w_baked_tt)
        kernel_vals_host = download_chip0(mesh, mesh_shape, top_val_out_tt)
        kernel_idxs_host = download_chip0(mesh, mesh_shape, top_idx_out_tt)
        k_pre_tile = download_chip0(mesh, mesh_shape, kernel_debug["pre_tile"])
        k_y_padded = download_chip0(mesh, mesh_shape, kernel_debug["y_padded"])
        k_logits_padded = download_chip0(mesh, mesh_shape, kernel_debug["logits_padded"])

        # Compare each stage. y_normed is no longer materialized (fused into
        # the lmhead matmul) so stage3 is dropped.
        # ref_pre_host shape [1, MHC]; kernel pre_tile shape [TILE, TILE].
        k_pre = k_pre_tile[0:1, 0:MHC]
        report_pcc("stage1/pre", ref_pre_host.float(), k_pre.float())
        # ref_y_host shape [1, 1, hidden]; kernel y_padded shape [TILE, hidden] row 0.
        k_y = k_y_padded[0:1, :].reshape(1, 1, DIM)
        report_pcc("stage2/y_bf16", ref_y_host.float(), k_y.float())
        # ref_logits_host shape [1, 1, vocab]; kernel logits_padded row 0.
        k_logits = k_logits_padded[0:1, :].reshape(1, 1, VOCAB)
        report_pcc("stage3/logits", ref_logits_host.float(), k_logits.float())
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
                  lambda: kernel(a_bf16_tt, hc_fn_t_scaled_tt,
                                 hc_base_tile_tt,
                                 scaler_bf16_tt, w_baked_tt),
                  mesh)

        sys.exit(0 if (ok_v and ok_i) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
