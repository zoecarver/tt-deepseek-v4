"""Lk-C PCC test: q_rsqrt_norm + q rotary + wkv (sans wkv all_gather).

Reference covers the half of `DeviceAttention.forward_device` between
the wq_b all_gather and the wkv all_gather:
- reshape q_full to [B,S,H,D]
- per-head rsqrt-norm
- pick cos/sin via embedding(start_pos, table); reshape to [1,S,1,rd/2]
- slice q nope/rope, rotary on rope half, concat
- ttnn.matmul(x, wkv) - partial pre-all_gather

The wq_a/wq_b path that produces q_full / qr is upstream (Lk-A/B).
For this test we feed in random q_full and x directly.

Fused as a single ttl.operation. Grid is (Np_wkv + N_q_cores, 1).
The first Np_wkv cores run a SUMMA matmul on the wkv path. The
remaining N_q_cores = N_HEADS // TILE cores each handle one row tile
of q: rmsnorm across HEAD_DIM, write the normed nope half to q_out,
then matmul against P + rotary combine for the rope half.

The wkv and q-stack paths are independent (no shared input or output)
so they run in parallel on different cores in the same kernel.

Rotary lowering trick: bake the rotate_half permutation into a swap
matrix P and the rotate_half sign into the sin table (sin_signed).
Then rotary becomes x * cos_extended + (x @ P) * sin_signed, where
cos_extended = pair-repeat(cos) and P = block_diag([[0,1],[1,0]] x rd/2).
cos/sin tables are pre-replicated across TILE rows on the host so the
tt-lang kernel can read tile-aligned cos/sin tiles without an extra
broadcast op. ttnn.embedding(start_pos, ...) is still ttnn glue
because it depends on a device uint32 index (TODO: mega).
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark

from inference import (
    _device_apply_rotary_interleaved, _device_q_rsqrt_norm,
)


DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_HEAD_DIM   # 448
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B, S = 1, 1
TILE = 32


def _make_fused_lkc_kernel(K_wkv: int, N_wkv: int,
                           NHEADS: int, HD: int, RD: int,
                           wkv_block_cfg, wkv_part_cfg,
                           h_block_tiles: int,
                           rms_eps: float,
                           fp32_dest_acc_en: bool = True):
    """Fused Lk-C: wkv SUMMA matmul + q rmsnorm + q rotary as one ttl.operation.

    Grid: (Np_wkv + N_q_cores, 1). Cores [0, Np_wkv) run a pure-SUMMA
    wkv matmul (M=TILE, K=K_wkv, N=N_wkv). Cores [Np_wkv, total) each
    own one row tile of q: per-row rmsnorm over HD, write normed nope
    blocks to q_out, then matmul their normed rope block against P
    (block-diag swap) and combine with cos/sin.

    Each q core walks its row in 2-tile-wide H blocks (h_block_tiles=2):
    NOPE_BLOCKS = NOPE_DIM/TILE/h_block_tiles nope blocks then exactly
    one rope block. The rope block carries the rotary tail; everything
    else is pure rmsnorm.

    The wkv and q paths share grid + scaler, nothing else.
    """
    bm_wkv, bn_wkv, bk_wkv = wkv_block_cfg
    Mp_wkv, Np_wkv, Kp_wkv = wkv_part_cfg
    if Mp_wkv != 1 or Kp_wkv != 1:
        raise ValueError(
            f"WKV path assumes Mp=Kp=1, got Mp={Mp_wkv} Kp={Kp_wkv}")
    Mt_wkv = TILE // TILE
    Nt_wkv = N_wkv // TILE
    Kt_wkv = K_wkv // TILE
    if Mt_wkv % bm_wkv or Nt_wkv % bn_wkv or Kt_wkv % bk_wkv:
        raise ValueError(
            f"wkv block must divide shape: Mt={Mt_wkv} Nt={Nt_wkv} Kt={Kt_wkv} "
            f"block=(bm={bm_wkv}, bn={bn_wkv}, bk={bk_wkv})")
    Mb_wkv = Mt_wkv // bm_wkv
    Nb_wkv = Nt_wkv // bn_wkv
    Kb_wkv = Kt_wkv // bk_wkv
    if Nb_wkv % Np_wkv or Mb_wkv != Mp_wkv:
        raise ValueError(
            f"wkv block/part: Mb={Mb_wkv} Nb={Nb_wkv} Mp={Mp_wkv} Np={Np_wkv}")
    M_BPN_wkv = Mb_wkv // Mp_wkv
    N_BPN_wkv = Nb_wkv // Np_wkv

    N_q_cores = NHEADS // TILE
    HD_t = HD // TILE
    H_BLOCK = h_block_tiles
    if HD_t % H_BLOCK:
        raise ValueError(f"HD_t={HD_t} not multiple of H_BLOCK={H_BLOCK}")
    H_BLOCKS_Q = HD_t // H_BLOCK
    NOPE_t = (HD - RD) // TILE
    if NOPE_t % H_BLOCK:
        raise ValueError(f"NOPE_t={NOPE_t} not multiple of H_BLOCK={H_BLOCK}")
    NOPE_BLOCKS_Q = NOPE_t // H_BLOCK
    ROPE_t = RD // TILE
    if ROPE_t != H_BLOCK:
        raise ValueError(
            f"rope must fit one H_BLOCK: ROPE_t={ROPE_t} H_BLOCK={H_BLOCK}")
    inv_D_q = 1.0 / float(HD)

    Np_total = Np_wkv + N_q_cores

    @ttl.operation(grid=(Np_total, 1),
                   fp32_dest_acc_en=fp32_dest_acc_en,
                   options="--no-ttl-reduce-full-fp32")
    def fused_kernel(q_full, gamma_q, x, wkv_w, P, cos_b, sin_b, scaler,
                     q_out, wkv_out):
        # WKV A row-mcast (only across cores 0..Np_wkv-1).
        wkv_a_pipes = [ttl.Pipe(src=(0, 0), dst=(slice(0, Np_wkv), 0))]
        wkv_a_net = ttl.PipeNet(wkv_a_pipes)

        # WKV CBs.
        wkv_a_cb = ttl.make_dataflow_buffer_like(
            x, shape=(bm_wkv, bk_wkv), block_count=2)
        wkv_b_cb = ttl.make_dataflow_buffer_like(
            wkv_w, shape=(bk_wkv, bn_wkv), block_count=2)
        wkv_out_cb = ttl.make_dataflow_buffer_like(
            wkv_out, shape=(bm_wkv, bn_wkv), block_count=2)

        # Q-side CBs.
        qx_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(1, H_BLOCK), block_count=2)
        qg_cb = ttl.make_dataflow_buffer_like(
            gamma_q, shape=(1, H_BLOCK), block_count=2)
        qsc_cb = ttl.make_dataflow_buffer_like(
            scaler, shape=(1, 1), block_count=1)
        qxsq_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(1, H_BLOCK), block_count=2)
        qsq_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(1, 1), block_count=2)
        qred_step_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(1, 1), block_count=2)
        qinv_bc_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(1, H_BLOCK), block_count=2)
        qout_cb = ttl.make_dataflow_buffer_like(
            q_out, shape=(1, H_BLOCK), block_count=2)
        qnormed_rope_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(1, H_BLOCK), block_count=2)
        qP_cb = ttl.make_dataflow_buffer_like(
            P, shape=(H_BLOCK, H_BLOCK), block_count=2)
        qcos_cb = ttl.make_dataflow_buffer_like(
            cos_b, shape=(1, H_BLOCK), block_count=2)
        qsin_cb = ttl.make_dataflow_buffer_like(
            sin_b, shape=(1, H_BLOCK), block_count=2)
        qswap_cb = ttl.make_dataflow_buffer_like(
            q_full, shape=(1, H_BLOCK), block_count=2)

        @ttl.compute()
        def compute():
            n_p, _ = ttl.node(dims=2)

            if n_p < Np_wkv:
                # WKV SUMMA matmul.
                for _ in range(M_BPN_wkv):
                    for _ in range(N_BPN_wkv):
                        p = wkv_out_cb.reserve()
                        for _ in range(Kb_wkv):
                            a_blk = wkv_a_cb.wait()
                            b_blk = wkv_b_cb.wait()
                            p += a_blk @ b_blk
            else:
                # Q-stack: rmsnorm across H_BLOCKS_Q blocks; rope tail rotary.
                sc = qsc_cb.wait()

                x0 = qx_cb.wait()
                qxsq_cb.reserve().store(x0 * x0)
                qsq_cb.reserve().store(
                    ttl.math.reduce_sum(qxsq_cb.wait(), sc, dims=[1])
                )
                for _ in range(H_BLOCKS_Q - 1):
                    xk = qx_cb.wait()
                    qxsq_cb.reserve().store(xk * xk)
                    qred_step_cb.reserve().store(
                        ttl.math.reduce_sum(qxsq_cb.wait(), sc, dims=[1])
                    )
                    prev = qsq_cb.wait()
                    qsq_cb.reserve().store(prev + qred_step_cb.wait())

                sq = qsq_cb.wait()
                inv_bc_t = qinv_bc_cb.reserve()
                inv_bc_t.store(ttl.math.broadcast(
                    ttl.math.rsqrt(
                        sq * ttl.math.fill(sq, inv_D_q)
                        + ttl.math.fill(sq, rms_eps)
                    ),
                    inv_bc_t, dims=[1],
                ))
                inv_bc = qinv_bc_cb.wait()

                # Nope blocks: write normalized values directly.
                for _ in range(NOPE_BLOCKS_Q):
                    xk = qx_cb.wait()
                    gk = qg_cb.wait()
                    qout_cb.reserve().store(xk * gk * inv_bc)

                # Rope block: normalize into scratch, matmul against P, combine.
                xk = qx_cb.wait()
                gk = qg_cb.wait()
                qnormed_rope_cb.reserve().store(xk * gk * inv_bc)
                normed = qnormed_rope_cb.wait()
                P_blk = qP_cb.wait()
                c = qcos_cb.wait()
                si = qsin_cb.wait()
                qswap_cb.reserve().store(normed @ P_blk)
                swap_blk = qswap_cb.wait()
                qout_cb.reserve().store(normed * c + swap_blk * si)

        @ttl.datamovement()
        def dm_read():
            n_p, _ = ttl.node(dims=2)

            if n_p < Np_wkv:
                # WKV: read x once, mcast across the Np_wkv row.
                for local_mb in range(M_BPN_wkv):
                    mr = local_mb * bm_wkv
                    for _ in range(N_BPN_wkv):
                        for kb in range(Kb_wkv):
                            kc = kb * bk_wkv
                            a_blk = wkv_a_cb.reserve()

                            def read_a(pipe):
                                ttl.copy(x[mr:mr + bm_wkv, kc:kc + bk_wkv],
                                         a_blk).wait()
                                ttl.copy(a_blk, pipe).wait()

                            wkv_a_net.if_src(read_a)
                            wkv_a_net.if_dst(
                                lambda pipe: (ttl.copy(pipe, a_blk).wait(),))
            else:
                # Q-stack: scaler, q row (twice), gamma, P, cos, sin.
                m_row = n_p - Np_wkv
                ttl.copy(scaler[0, 0], qsc_cb.reserve()).wait()
                # x for ssq (8 H blocks).
                for h_block in range(H_BLOCKS_Q):
                    h_start = h_block * H_BLOCK
                    ttl.copy(
                        q_full[m_row:m_row + 1, h_start:h_start + H_BLOCK],
                        qx_cb.reserve()).wait()
                # x and gamma for normalize (8 H blocks).
                for h_block in range(H_BLOCKS_Q):
                    h_start = h_block * H_BLOCK
                    ttl.copy(
                        q_full[m_row:m_row + 1, h_start:h_start + H_BLOCK],
                        qx_cb.reserve()).wait()
                    ttl.copy(
                        gamma_q[0:1, h_start:h_start + H_BLOCK],
                        qg_cb.reserve()).wait()
                # Rotary side inputs (rope block only).
                ttl.copy(P[0:H_BLOCK, 0:H_BLOCK], qP_cb.reserve()).wait()
                ttl.copy(cos_b[0:1, 0:H_BLOCK], qcos_cb.reserve()).wait()
                ttl.copy(sin_b[0:1, 0:H_BLOCK], qsin_cb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            n_p, _ = ttl.node(dims=2)

            if n_p < Np_wkv:
                # WKV: read B per N-block, write output tile.
                for local_mb in range(M_BPN_wkv):
                    mr = local_mb * bm_wkv
                    for local_nb in range(N_BPN_wkv):
                        nb = n_p * N_BPN_wkv + local_nb
                        nc = nb * bn_wkv
                        for kb in range(Kb_wkv):
                            kc = kb * bk_wkv
                            b_blk = wkv_b_cb.reserve()
                            ttl.copy(wkv_w[kc:kc + bk_wkv, nc:nc + bn_wkv],
                                     b_blk).wait()
                        o = wkv_out_cb.wait()
                        ttl.copy(o,
                                 wkv_out[mr:mr + bm_wkv, nc:nc + bn_wkv]).wait()
            else:
                # Q-stack: write H_BLOCKS_Q blocks (nope + rope_rot).
                m_row = n_p - Np_wkv
                for h_block in range(H_BLOCKS_Q):
                    h_start = h_block * H_BLOCK
                    o = qout_cb.wait()
                    ttl.copy(
                        o, q_out[m_row:m_row + 1, h_start:h_start + H_BLOCK]
                    ).wait()

    return fused_kernel


def _build_rotary_tables(cos_full_cpu: torch.Tensor, sin_full_cpu: torch.Tensor,
                         inverse: bool):
    """Build host-side pre-processed cos/sin tables for the kernel rotary.

    Output:
      cos_extended_packed: [max_seq_len, TILE * rd] bf16
        each row holds TILE replicas of [c0, c0, c1, c1, ..., c_{rd/2-1},
        c_{rd/2-1}] so that ttnn.embedding(start_pos, ...) followed by
        reshape([TILE, rd]) yields a tile-aligned [TILE, rd] tensor with
        the same cos values on every row.
      sin_signed_packed: [max_seq_len, TILE * rd] bf16
        same layout, but the sin values are signed so that
        out = x * cos_extended + (x @ P) * sin_signed reproduces the
        forward (or inverse) rotary exactly.

    sin sign pattern:
      forward: [-s, +s, -s, +s, ...]   (so that x[2k+1] picks up -sin(k))
      inverse: [+s, -s, +s, -s, ...]
    """
    max_seq_len, rd_half = cos_full_cpu.shape
    rd = 2 * rd_half
    if rd % TILE != 0:
        raise ValueError(f"rd={rd} not multiple of TILE={TILE}")

    cos_extended = cos_full_cpu.repeat_interleave(2, dim=-1)  # (T, rd)
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
    """Block-diagonal swap matrix P [rd, rd] with 2x2 [[0,1],[1,0]] blocks."""
    if rd % 2:
        raise ValueError(f"rd={rd} must be even")
    P = torch.zeros(rd, rd, dtype=torch.bfloat16)
    for k in range(rd // 2):
        P[2 * k, 2 * k + 1] = 1.0
        P[2 * k + 1, 2 * k] = 1.0
    return P


def make_lk_c_kernel(mesh, cos_full_cpu, sin_full_cpu):
    """Mega kernel for Lk-C = q_rsqrt_norm + q rotary + wkv matmul.

    Single fused ttl.operation. cos/sin row gather still uses
    ttnn.embedding (no tt-lang scalar-indexed gather yet).
    """
    # WKV: M=TILE, K=DIM=4096, N=HEAD_DIM=512.
    # block=(1, 4, 8) part=(1, 4, 1) -> 4 cores, N_BPN=1, Kb=16.
    # Q-side: 2 cores, one per N_HEADS row tile. Total 6 cores.
    fused_kernel = _make_fused_lkc_kernel(
        K_wkv=DIM, N_wkv=HEAD_DIM,
        NHEADS=N_HEADS, HD=HEAD_DIM, RD=ROPE_HEAD_DIM,
        wkv_block_cfg=(1, 4, 8), wkv_part_cfg=(1, 4, 1),
        h_block_tiles=2,
        rms_eps=NORM_EPS,
    )

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    # gamma=ones for q_rsqrt_norm. Packed [TILE, HEAD_DIM] so the kernel's
    # gamma[0, h] reads cover the full row.
    gamma_q_packed = torch.ones(HEAD_DIM, dtype=torch.bfloat16) \
        .unsqueeze(0).expand(TILE, -1).contiguous()
    gamma_q_tt = ttnn.as_tensor(gamma_q_packed, dtype=ttnn.bfloat16, **rep)

    cos_ext_packed, sin_signed_packed = _build_rotary_tables(
        cos_full_cpu, sin_full_cpu, inverse=False)
    cos_ext_tt = ttnn.as_tensor(cos_ext_packed, dtype=ttnn.bfloat16, **rep)
    sin_signed_tt = ttnn.as_tensor(sin_signed_packed, dtype=ttnn.bfloat16, **rep)

    P_cpu = _build_swap_matrix(ROPE_HEAD_DIM)
    P_tt = ttnn.as_tensor(P_cpu.contiguous(), dtype=ttnn.bfloat16, **rep)

    state: dict = {}

    def _alloc_replicated_zeros(shape):
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def lk_c_kernel(q_full, x, cos_full, sin_full, start_pos, wkv_w,
                    q_out, wkv_out):
        if "scratch" not in state:
            state["scaler_tt"] = ttnn.from_torch(
                torch.ones((TILE, TILE), dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state["q_padded_tt"] = _alloc_replicated_zeros(
                (N_HEADS, HEAD_DIM))
            state["wkv_padded_tt"] = _alloc_replicated_zeros(
                (TILE, HEAD_DIM))
            state["scratch"] = True

        # Reshape q_full [1,1,N_HEADS*HD] -> [N_HEADS, HD] and pad x.
        q_2d = ttnn.reshape(q_full, [N_HEADS, HEAD_DIM])
        x_2d = ttnn.reshape(x, [B * S, DIM])
        x_padded = ttnn.pad(
            x_2d, padding=[(0, TILE - B * S), (0, 0)], value=0.0)

        # cos/sin gather: ttnn.embedding (no tt-lang scalar gather yet).
        cos_b_2d = ttnn.embedding(start_pos, cos_ext_tt, layout=ttnn.TILE_LAYOUT)
        sin_b_2d = ttnn.embedding(start_pos, sin_signed_tt, layout=ttnn.TILE_LAYOUT)
        cos_b = ttnn.reshape(cos_b_2d, [TILE, ROPE_HEAD_DIM])
        sin_b = ttnn.reshape(sin_b_2d, [TILE, ROPE_HEAD_DIM])

        # Single fused ttl.operation: rmsnorm(q) + rotary(q_rope) + wkv matmul.
        fused_kernel(
            q_2d, gamma_q_tt, x_padded, wkv_w, P_tt, cos_b, sin_b,
            state["scaler_tt"], state["q_padded_tt"], state["wkv_padded_tt"],
        )

        # Reshape q_out and wkv_out into the test-shaped buffers.
        q_4d = ttnn.reshape(state["q_padded_tt"],
                            [B, S, N_HEADS, HEAD_DIM])
        ttnn.copy(q_4d, q_out)
        wkv_row = ttnn.slice(state["wkv_padded_tt"],
                             [0, 0], [B * S, HEAD_DIM])
        wkv_3d = ttnn.reshape(wkv_row, [B, S, HEAD_DIM])
        ttnn.copy(wkv_3d, wkv_out)

    return lk_c_kernel


def reference(mesh, q_full_tt, x_tt, cos_full_tt, sin_full_tt,
              start_pos_tt, wkv_w_tt):
    # q-stack tail: reshape, q_rsqrt_norm, pick cos/sin, slice nope/rope,
    # rotary on rope, concat. (Mirror of DeviceAttention.forward_device,
    # the attn.q phase after wq_b matmul.)
    q_tt = ttnn.reshape(q_full_tt, [B, S, N_HEADS, HEAD_DIM])
    q_tt = _device_q_rsqrt_norm(ttnn, q_tt, NORM_EPS)
    rd_half = ROPE_HEAD_DIM // 2
    cos = ttnn.embedding(start_pos_tt, cos_full_tt, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(start_pos_tt, sin_full_tt, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, S, 1, rd_half])
    sin = ttnn.reshape(sin, [1, S, 1, rd_half])
    q_nope = ttnn.slice(q_tt, [0, 0, 0, 0], [B, S, N_HEADS, HEAD_DIM - ROPE_HEAD_DIM])
    q_rope = ttnn.slice(q_tt, [0, 0, 0, HEAD_DIM - ROPE_HEAD_DIM], [B, S, N_HEADS, HEAD_DIM])
    q_rope = _device_apply_rotary_interleaved(ttnn, q_rope, cos, sin, inverse=False)
    q_tt = ttnn.concat([q_nope, q_rope], dim=-1)

    # wkv matmul (no all_gather; weight replicated).
    wkv_out_tt = ttnn.matmul(x_tt, wkv_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return q_tt, wkv_out_tt


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        q_full = torch.randn(1, 1, N_HEADS * HEAD_DIM, dtype=torch.bfloat16) * 0.1
        x = torch.randn(1, 1, DIM, dtype=torch.bfloat16) * 0.1
        cos_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        sin_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        wkv_w = torch.randn(DIM, HEAD_DIM, dtype=torch.bfloat16) * 0.02
        start_pos = torch.tensor([[1]], dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        q_full_tt = ttnn.as_tensor(q_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep)
        cos_full_tt = ttnn.as_tensor(cos_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        sin_full_tt = ttnn.as_tensor(sin_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        wkv_w_tt = ttnn.as_tensor(wkv_w.contiguous(), dtype=ttnn.bfloat16, **rep)
        start_pos_tt = ttnn.from_torch(
            start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        ref_q_tt, ref_wkv_tt = reference(
            mesh, q_full_tt, x_tt, cos_full_tt, sin_full_tt,
            start_pos_tt, wkv_w_tt)
        ref_q_host = download_chip0(mesh, mesh_shape, ref_q_tt)
        ref_wkv_host = download_chip0(mesh, mesh_shape, ref_wkv_tt)

        kernel = make_lk_c_kernel(mesh, cos_full, sin_full)
        q_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, N_HEADS, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        wkv_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(q_full_tt, x_tt, cos_full_tt, sin_full_tt,
               start_pos_tt, wkv_w_tt, q_out_tt, wkv_out_tt)
        kernel_q_host = download_chip0(mesh, mesh_shape, q_out_tt)
        kernel_wkv_host = download_chip0(mesh, mesh_shape, wkv_out_tt)

        ok_q = report_pcc("Lk-C/q", ref_q_host, kernel_q_host)
        ok_kv = report_pcc("Lk-C/wkv", ref_wkv_host, kernel_wkv_host)

        benchmark("Lk-C ref",
                  lambda: reference(mesh, q_full_tt, x_tt, cos_full_tt,
                                    sin_full_tt, start_pos_tt, wkv_w_tt),
                  mesh)
        benchmark("Lk-C ttl",
                  lambda: kernel(q_full_tt, x_tt, cos_full_tt, sin_full_tt,
                                 start_pos_tt, wkv_w_tt, q_out_tt, wkv_out_tt),
                  mesh)

        sys.exit(0 if (ok_q and ok_kv) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
