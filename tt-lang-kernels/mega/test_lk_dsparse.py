"""Lk-Dsparse PCC test: sparse_attn body + inverse rotary + wo_a + wo_b.

Reference covers everything from the topk indices ready -> wo_b matmul,
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

Pipeline (one inlined SUMMA factory, 11 dispatches total):
  1 score + 1 output + 8 per-group wo_a + 1 wo_b = 11 SUMMAs

ttnn glue (TODO: mega):
  - paged_update_cache (sharded L1 staging + experimental kernel)
  - _idxs_int_tile_to_idxs_and_mask: lt + reshape + where + clamp +
    typecast int32->uint32 + to_layout (no tt-lang int compares /
    layout converts)
  - embedding gather (no tt-lang gather)
  - softmax_scale multiply (could fold into pre-scaled kv on host but
    kv changes per step)
  - concat sink, softmax, slice (no tt-lang softmax)
  - embedding(start_pos, ...) for the inverse rotary cos/sin lookup
    (rotary math itself is now in tt-lang via swap-SUMMA + combine)
  - group reshape + permute (no tt-lang permute)
  - per-group input/weight slicing for the wo_a dispatch loop
  - final permute + reshape to assemble o_flat
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark

from inference import (
    DeviceSparseAttn, _device_apply_rotary_interleaved,
)


DIM = 4096
N_HEADS = 64
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
N_GROUPS = 8
PER_GROUP = (N_HEADS * HEAD_DIM) // N_GROUPS  # 4096
O_LORA_RANK = 1024
KV_CACHE_SIZE_PAD = 128
WIN = 128
K = WIN                            # window-only topk for the test (no compressor)
MAX_SEQ_LEN = 512
B, S = 1, 1
TILE = 32


def _make_summa_matmul_kernel(M: int, K_dim: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    """Pure SUMMA matmul. A row-mcast across Np cores; B col-mcast across Mp."""
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
    """SUMMA matmul with K-split on the row axis. grid=(Np, Kp), Mp=1."""
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Mp != 1:
        raise ValueError(f"ksplit kernel here assumes Mp=1, got {Mp}")
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

    @ttl.operation(grid=(Np, Kp), fp32_dest_acc_en=fp32_dest_acc_en)
    def ksplit_matmul(a, w, out):
        a_pipes = [ttl.Pipe(src=(0, k_p), dst=(slice(0, Np), k_p))
                   for k_p in range(Kp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
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


def _make_swap_combine_kernel(M: int, K_dim: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    """SUMMA matmul fused with rotary combine.
      out[m, n] = x[m, n] * cos[0, n] + (x @ P)[m, n] * sin[0, n]
    cos, sin shape: [TILE, N] (only row 0 is consumed; downstream broadcasts).
    P shape: [K_dim, N] swap-pairs constant.

    Each (m_p, n_p) core matmul-accumulates the swap tile, reads its
    diagonal x tile + cos/sin tiles, combines, and writes out.
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


def _make_rotary_combine_kernel(num_row_tiles: int, num_h_tiles: int):
    """out = x * cos + x_swap * sin; cos/sin tile-replicated.

    Distributes (row_tile, h_tile) pairs across an auto grid for better
    parallelism: H=64, rd=64 -> 2*2=4 tiles distributed over 4 cores.
    """
    total_work = num_row_tiles * num_h_tiles

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def rotary_combine(x, x_swap, cos, sin, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        xs_dfb = ttl.make_dataflow_buffer_like(x_swap, shape=(1, 1), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(cos, shape=(1, 1), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(sin, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    xt = x_dfb.wait()
                    xst = xs_dfb.wait()
                    ct = c_dfb.wait()
                    st = s_dfb.wait()
                    out_dfb.reserve().store(xt * ct + xst * st)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    t = global_w // num_h_tiles
                    h = global_w % num_h_tiles
                    ttl.copy(x[t, h], x_dfb.reserve()).wait()
                    ttl.copy(x_swap[t, h], xs_dfb.reserve()).wait()
                    ttl.copy(cos[0, h], c_dfb.reserve()).wait()
                    ttl.copy(sin[0, h], s_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    t = global_w // num_h_tiles
                    h = global_w % num_h_tiles
                    ttl.copy(out_dfb.wait(), out[t, h]).wait()

    return rotary_combine


def _make_summa_matmul_scale_mask_kernel(M: int, K_dim: int, N: int,
                                          block_cfg, part_cfg,
                                          scale: float, mask_amp: float,
                                          fp32_dest_acc_en: bool = True):
    """SUMMA matmul out = a @ b.T fused with scale + sign-trick mask.

      masked[m, n] = (a @ w.T)[m, n] * scale +
                     (sign(idx[0, n] + 0.5) - 1) * (mask_amp / 2)

    B operand is stored as [N, K_dim] (not [K_dim, N]); each block of b
    is read transposed and ttl.transpose'd before the inner matmul. This
    folds an external ttnn.transpose(b, -2, -1) on the score path.
    """
    half_amp = mask_amp / 2.0
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
    def matmul_scale_mask(a, w, idx, masked_out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p))
                   for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp)))
                   for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)
        idx_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp)))
                     for n_p in range(Np)]
        mcast_idx_net = ttl.PipeNet(idx_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bn, bk), block_count=2)
        bt_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        idx_cb = ttl.make_dataflow_buffer_like(
            idx, shape=(1, bn), block_count=2)
        raw_cb = ttl.make_dataflow_buffer_like(
            masked_out, shape=(bm, bn), block_count=2)
        mask_scratch = ttl.make_dataflow_buffer_like(
            masked_out, shape=(1, bn), block_count=2)
        masked_cb = ttl.make_dataflow_buffer_like(
            masked_out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = raw_cb.reserve()
                    for _ in range(Kb):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        with bt_cb.reserve() as bt:
                            bt.store(ttl.transpose(b_blk))
                        p += a_blk @ bt_cb.wait()
                    r = raw_cb.wait()
                    i = idx_cb.wait()
                    mask_scratch.reserve().store(
                        (ttl.math.sign(i + ttl.math.fill(i, 0.5))
                         - ttl.math.fill(i, 1.0))
                        * ttl.math.fill(i, half_amp))
                    m = mask_scratch.wait()
                    masked_cb.reserve().store(
                        r * ttl.math.fill(r, scale) + m)

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
                            ttl.copy(w[nc:nc + bn, kc:kc + bk], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, b_blk).wait(),))
                    idx_blk = idx_cb.reserve()

                    def read_idx(pipe):
                        ttl.copy(idx[0:1, nc:nc + bn], idx_blk).wait()
                        ttl.copy(idx_blk, pipe).wait()

                    mcast_idx_net.if_src(read_idx)
                    mcast_idx_net.if_dst(
                        lambda pipe: (ttl.copy(pipe, idx_blk).wait(),))
                    ttl.copy(masked_cb.wait(),
                             masked_out[mr:mr + bm, nc:nc + bn]).wait()

    return matmul_scale_mask


def _make_scale_sign_mask_kernel(num_row_tiles: int, num_k_tiles: int,
                                  scale: float, mask_amp: float):
    """Fused scale + sign-trick mask:

        masked[mt, kt] = scores[mt, kt] * softmax_scale +
                         (sign(idx[0, kt] + 0.5) - 1) * (mask_amp / 2)

    Replaces (ttnn.lt int<0 + where(-inf, 0)) + multiply(scale) + add(mask):
      idx >= 0 -> mask = 0
      idx < 0  -> mask = -mask_amp  (functionally -inf for softmax)
    idx is broadcast across the H (row) tiles. Tiles are distributed across
    an auto grid, one (mt, kt) pair per core when possible.
    """
    half_amp = mask_amp / 2.0
    total_work = num_row_tiles * num_k_tiles

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def scale_sign_mask(scores, idxs_bf16, masked_out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        scores_dfb = ttl.make_dataflow_buffer_like(
            scores, shape=(1, 1), block_count=2)
        idxs_dfb = ttl.make_dataflow_buffer_like(
            idxs_bf16, shape=(1, 1), block_count=2)
        mask_dfb = ttl.make_dataflow_buffer_like(
            masked_out, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(
            masked_out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    s = scores_dfb.wait()
                    i = idxs_dfb.wait()
                    mask_dfb.reserve().store(
                        (ttl.math.sign(i + ttl.math.fill(i, 0.5))
                         - ttl.math.fill(i, 1.0))
                        * ttl.math.fill(i, half_amp))
                    m = mask_dfb.wait()
                    out_dfb.reserve().store(s * ttl.math.fill(s, scale) + m)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    kt = global_w // num_row_tiles
                    mt = global_w % num_row_tiles
                    ttl.copy(scores[mt, kt], scores_dfb.reserve()).wait()
                    ttl.copy(idxs_bf16[0, kt], idxs_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    kt = global_w // num_row_tiles
                    mt = global_w % num_row_tiles
                    ttl.copy(out_dfb.wait(), masked_out[mt, kt]).wait()

    return scale_sign_mask


def _make_softmax_with_sink_kernel(H: int, K_dim: int):
    """Softmax over [H, K_dim + sink] with the sink column dropped from output.

    Inputs:
      x: [H, K_dim] bf16. Per-row scores.
      sink_padded: [H, TILE] bf16. Column 0 = per-head sink, columns 1..TILE-1
        = -1e9 sentinel (so they have no effect on max/sum).
      scaler: [TILE, TILE] bf16 ones (reduce_max/reduce_sum scratch).
      out: [H, K_dim] bf16. softmax probs over [scores | sink], sink dropped.

    Per H-slice we run three streaming passes over the K-tiles + 1 sink-tile:
      1) running max via reduce_max(dims=[1]) + ttl.math.max combine
      2) running sum of exp(x - max_bc) via reduce_sum(dims=[1]) + add
      3) write exp(x - max_bc) * recip(sum)_bc for K-tiles only

    Single-core. K is small (4 tiles for Lk-Dsparse) so re-streaming x 3x is
    cheaper than scratching the exp results.
    """
    Ht = H // TILE
    Kt = K_dim // TILE

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
    def softmax_with_sink(x, sink_padded, scaler, out):
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        sink_dfb = ttl.make_dataflow_buffer_like(sink_padded, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        rmax_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        run_max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        max_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        rsum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        run_sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rinv_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            sc = sc_dfb.wait()
            for _ in range(Ht):
                # Pass 1: running max across K-tiles + sink-tile.
                init_rm = run_max_dfb.reserve()
                init_rm.store(ttl.math.fill(init_rm, -1.0e9))
                for _ in range(Kt):
                    v = x_dfb.wait()
                    rm = run_max_dfb.wait()
                    rmax_dfb.reserve().store(
                        ttl.math.reduce_max(v, sc, dims=[1]))
                    tm_blk = rmax_dfb.wait()
                    run_max_dfb.reserve().store(ttl.math.max(rm, tm_blk))
                vs = sink_dfb.wait()
                rm = run_max_dfb.wait()
                rmax_dfb.reserve().store(
                    ttl.math.reduce_max(vs, sc, dims=[1]))
                tm_blk = rmax_dfb.wait()
                run_max_dfb.reserve().store(ttl.math.max(rm, tm_blk))
                final_rm = run_max_dfb.wait()
                mb = max_bc_dfb.reserve()
                mb.store(ttl.math.broadcast(final_rm, mb, dims=[1]))
                mb_blk = max_bc_dfb.wait()

                # Pass 2: running sum of exp(x - max_bc) across K-tiles + sink.
                init_rs = run_sum_dfb.reserve()
                init_rs.store(ttl.math.fill(init_rs, 0.0))
                for _ in range(Kt):
                    v2 = x_dfb.wait()
                    rs = run_sum_dfb.wait()
                    rsum_dfb.reserve().store(
                        ttl.math.reduce_sum(
                            ttl.math.exp(v2 - mb_blk), sc, dims=[1]))
                    ts_blk = rsum_dfb.wait()
                    run_sum_dfb.reserve().store(rs + ts_blk)
                vs2 = sink_dfb.wait()
                rs = run_sum_dfb.wait()
                rsum_dfb.reserve().store(
                    ttl.math.reduce_sum(
                        ttl.math.exp(vs2 - mb_blk), sc, dims=[1]))
                ts_blk = rsum_dfb.wait()
                run_sum_dfb.reserve().store(rs + ts_blk)

                final_rs = run_sum_dfb.wait()
                rinv = rinv_bc_dfb.reserve()
                rinv.store(ttl.math.broadcast(
                    ttl.math.recip(final_rs), rinv, dims=[1]))
                rinv_blk = rinv_bc_dfb.wait()

                # Pass 3: write exp(x - max_bc) * rinv_bc, skip sink.
                for _ in range(Kt):
                    v3 = x_dfb.wait()
                    out_dfb.reserve().store(
                        ttl.math.exp(v3 - mb_blk) * rinv_blk)

        @ttl.datamovement()
        def dm_read():
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for h in range(Ht):
                # Pass 1: stream x and sink for max.
                for kt in range(Kt):
                    ttl.copy(x[h, kt], x_dfb.reserve()).wait()
                ttl.copy(sink_padded[h, 0], sink_dfb.reserve()).wait()
                # Pass 2: stream x and sink for exp/sum.
                for kt in range(Kt):
                    ttl.copy(x[h, kt], x_dfb.reserve()).wait()
                ttl.copy(sink_padded[h, 0], sink_dfb.reserve()).wait()
                # Pass 3: stream x for output.
                for kt in range(Kt):
                    ttl.copy(x[h, kt], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            for h in range(Ht):
                for kt in range(Kt):
                    ttl.copy(out_dfb.wait(), out[h, kt]).wait()

    return softmax_with_sink


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


def make_lk_dsparse_kernel(mesh, cos_full_cpu, sin_full_cpu,
                           sharded_input_memcfg, softmax_scale: float):
    """Mega kernel for Lk-Dsparse.

    Pipeline:
      1. paged_update_cache (kv -> kv_cache at kv_slot)  [TODO: mega]
      2. Build idxs_uint32_rm + valid_bf16_tile mask     [TODO: mega]
      3. ttnn.embedding gather -> kv_gather              [TODO: mega]
      4. ttnn.transpose -> kv_gather_t                   [TODO: mega]
      5. SUMMA score: q [H, D] @ kv_gather_t [D, K] -> scores [H, K]
      6. multiply by softmax_scale + add mask            [TODO: mega]
      7. concat sink, softmax, drop sink                 [TODO: mega]
      8. SUMMA output: probs [H, K] @ kv_gather [K, D] -> o [H, D]
      9. inverse rotary: swap-SUMMA + rotary-combine on o[..., -rd:]
     10. group reshape/permute                           [TODO: mega]
     11. SUMMA wo_a x 8 groups: o_g [TILE, per_group] @ wo_a_g [per_group, O_LORA_RANK]
     12. permute back + reshape -> o_flat [TILE, N_GROUPS*O_LORA_RANK]
     13. SUMMA wo_b: o_flat [TILE, 8192] @ wo_b_w [8192, DIM] -> out [TILE, DIM]
     14. slice + reshape -> [B, S, DIM]
    """
    H = N_HEADS
    D = HEAD_DIM
    rd = ROPE_HEAD_DIM

    # SUMMA score fused with scale + sign-trick mask. M=H=64, K=D=512, N=K=128.
    # Mt=2, Kt=16, Nt=4. block (1, 1, 8) part (2, 4, 1): 8 cores, Kb=2.
    matmul_score_scale_mask = _make_summa_matmul_scale_mask_kernel(
        M=H, K_dim=D, N=K,
        block_cfg=(1, 1, 8), part_cfg=(2, 4, 1),
        scale=softmax_scale, mask_amp=1.0e4)

    # SUMMA output: M=H=64, K=K=128, N=D=512. Mt=2, Kt=4, Nt=16.
    # block (1, 4, 4) part (2, 4, 1): Mb=2, Nb=4, Kb=1 -> 8 cores.
    matmul_output = _make_summa_matmul_kernel(
        M=H, K_dim=K, N=D,
        block_cfg=(1, 4, 4), part_cfg=(2, 4, 1))

    # SUMMA wo_a per-group: M=TILE, K=PER_GROUP=4096, N=O_LORA_RANK=1024.
    # Mt=1, Kt=128, Nt=32. block (1, 4, 8) part (1, 8, 1): Nb=8, Kb=16 -> 8 cores.
    matmul_wo_a = _make_summa_matmul_kernel(
        M=TILE, K_dim=PER_GROUP, N=O_LORA_RANK,
        block_cfg=(1, 4, 8), part_cfg=(1, 8, 1))

    # KSPLIT wo_b: M=TILE, K=N_GROUPS*O_LORA_RANK=8192, N=DIM=4096.
    # Mt=1, Kt=256, Nt=128. block=(1, 16, 8) part=(1, 8, 8) -> 64 cores
    # (Nb=8, N_BPN=1, Kb=32, K_BPN=4).
    matmul_wo_b = _make_ksplit_matmul_kernel(
        M=TILE, K=N_GROUPS * O_LORA_RANK, N=DIM,
        block_cfg=(1, 16, 8), part_cfg=(1, 8, 8))

    # Inverse rotary fused swap-SUMMA + cos/sin combine: M=H=64, K=N=rd=64.
    # block=(1,1,2), part=(2,2,1) -> 4 cores.
    swap_combine_kernel = _make_swap_combine_kernel(
        M=H, K_dim=rd, N=rd,
        block_cfg=(1, 1, 2), part_cfg=(2, 2, 1))

    # Fused softmax over [scores | sink], drop-sink. Replaces ttnn concat +
    # softmax + slice. Single-core, three-pass over Ht=2 H-slices.
    softmax_with_sink_kernel = _make_softmax_with_sink_kernel(H=H, K_dim=K)

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    cos_ext_packed, sin_signed_packed = _build_rotary_tables(
        cos_full_cpu, sin_full_cpu, inverse=True)
    cos_ext_tt = ttnn.as_tensor(cos_ext_packed, dtype=ttnn.bfloat16, **rep)
    sin_signed_tt = ttnn.as_tensor(sin_signed_packed, dtype=ttnn.bfloat16, **rep)
    P_cpu = _build_swap_matrix(rd)
    P_tt = ttnn.as_tensor(P_cpu.contiguous(), dtype=ttnn.bfloat16, **rep)

    state: dict = {}

    def lk_dsparse_kernel(q_tt, kv_tt, kv_cache_tt, kv_slot_tt, topk_idxs_tt,
                          sink_padded_tt, sm_scaler_tt, cos_full_tt,
                          sin_full_tt, start_pos_tt,
                          wo_a_w_tt, wo_b_w_tt, out):
        if "init" not in state:
            state["o_padded"] = ttnn.from_torch(
                torch.zeros(H, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["wo_b_out_padded"] = ttnn.from_torch(
                torch.zeros(TILE, DIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["o_rope_rot"] = ttnn.from_torch(
                torch.zeros(H, ROPE_HEAD_DIM, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["scores_masked_padded"] = ttnn.from_torch(
                torch.zeros(H, K, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["probs"] = ttnn.from_torch(
                torch.zeros(H, K, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["init"] = True

        # 1. paged_update_cache.
        # TODO: mega fusion blocked (bucket #1 — unwired, not primitive):
        # tt-lang now has element_write (see /Users/zcarver/Developer/
        # tt-lang/element_read_write). Slot-write at kv_slot_tt is a
        # straight element_write of kv_4d into kv_cache_tt[kv_slot, :].
        # See README bucket #1. Tracked as task C10.
        kv_4d = ttnn.reshape(kv_tt, [1, B, 1, D])
        kv_4d_sharded = ttnn.to_memory_config(kv_4d, memory_config=sharded_input_memcfg)
        ttnn.experimental.paged_update_cache(
            kv_cache_tt, kv_4d_sharded, update_idxs_tensor=kv_slot_tt)

        # 2. Build idxs_uint32_rm + bf16 topk idxs for the fused mask kernel.
        # Mask construction (lt + where(-inf, 0)) is folded into the tt-lang
        # scale_sign_mask kernel below via the bf16 sign trick.
        # TODO: mega fusion blocked: ttnn used for clamp/typecast/to_layout.
        # Lowering needs tt-lang int->uint32 cast + tile<->row-major converter.
        topk_idxs_bf16 = ttnn.typecast(topk_idxs_tt, dtype=ttnn.bfloat16)
        topk_2d = ttnn.reshape(topk_idxs_bf16, [B, K])
        topk_padded_in = ttnn.pad(
            topk_2d, padding=[(0, TILE - B), (0, 0)], value=0.0)
        safe = ttnn.clamp(topk_idxs_tt, min=0)
        safe = ttnn.reshape(safe, [B, S * K])
        safe = ttnn.typecast(safe, dtype=ttnn.uint32)
        idxs_tt = ttnn.to_layout(safe, ttnn.ROW_MAJOR_LAYOUT)

        # 3. Gather kv_cache rows at the topk indices.
        # TODO: mega fusion blocked: ttnn used for embedding (gather). No
        # tt-lang gather primitive yet.
        kv_full_tt = ttnn.reshape(kv_cache_tt, [KV_CACHE_SIZE_PAD, D])
        kv_gather = ttnn.embedding(idxs_tt, kv_full_tt, layout=ttnn.TILE_LAYOUT)
        kv_gather_4d = ttnn.reshape(kv_gather, [B, S, K, D])

        # 4. (Removed ttnn.transpose - folded into score matmul as B-transpose.)

        # 5. SUMMA score fused with scale + sign-trick mask.
        # masked = (q @ kv_gather.T) * softmax_scale + sign_mask(idx).
        # The score matmul is built with B-transpose: kv_gather is passed
        # in its natural [K, D] layout, transposed per-block in compute.
        kv_gather_2d = ttnn.reshape(kv_gather_4d, [K, D])
        q_2d = ttnn.reshape(q_tt, [H, D])
        matmul_score_scale_mask(q_2d, kv_gather_2d, topk_padded_in,
                                state["scores_masked_padded"])

        # 7. Fused softmax over [scores | sink] with sink dropped. Replaces
        # ttnn.concat + ttnn.softmax + ttnn.slice with one tt-lang kernel.
        softmax_with_sink_kernel(
            state["scores_masked_padded"], sink_padded_tt, sm_scaler_tt,
            state["probs"])

        # 8. SUMMA output: probs [H, K] @ kv_gather [K, D] -> o [H, D].
        # Reuses kv_gather_2d from step 5.
        matmul_output(state["probs"], kv_gather_2d, state["o_padded"])

        # 9. Inverse rotary on o[..., -rd:] via swap-SUMMA + rotary-combine.
        # Operates on the 2D [H, D] layout and slices the rope tail directly.
        # TODO: mega fusion blocked: ttnn used for embedding(start_pos, ...).
        cos_b_2d = ttnn.embedding(start_pos_tt, cos_ext_tt, layout=ttnn.TILE_LAYOUT)
        sin_b_2d = ttnn.embedding(start_pos_tt, sin_signed_tt, layout=ttnn.TILE_LAYOUT)
        cos_b = ttnn.reshape(cos_b_2d, [TILE, rd])
        sin_b = ttnn.reshape(sin_b_2d, [TILE, rd])
        o_nope_2d = ttnn.slice(state["o_padded"], [0, 0], [H, D - rd])
        o_rope_2d = ttnn.slice(state["o_padded"], [0, D - rd], [H, D])
        swap_combine_kernel(o_rope_2d, P_tt, cos_b, sin_b,
                            state["o_rope_rot"])
        o_full_2d = ttnn.concat([o_nope_2d, state["o_rope_rot"]], dim=-1)
        o_concat = ttnn.reshape(o_full_2d, [B, S, H, D])

        # 10. Group reshape + permute.
        # TODO: mega fusion you should be able to build a a tt-lang batched-matmul-with-replicated-A operation.
        o_perm = ttnn.reshape(o_concat, [B, S, N_GROUPS, PER_GROUP])
        o_perm = ttnn.permute(o_perm, [2, 0, 1, 3])  # [G, B, S, per_group]
        o_g = ttnn.reshape(o_perm, [N_GROUPS, B * S, PER_GROUP])

        # 11. wo_a as a single batched matmul [G, M, K] @ [G, K, N] -> [G, M, N].
        # TODO: mega: replace with batched-SUMMA tt-lang kernel.
        o_g_padded_3d = ttnn.pad(
            o_g, padding=[(0, 0), (0, TILE - B * S), (0, 0)], value=0.0)
        o_wo_a_g = ttnn.matmul(o_g_padded_3d, wo_a_w_tt,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # 12. Permute [G, TILE, O_LORA_RANK] -> [TILE, G*O_LORA_RANK].
        o_wo_a_perm = ttnn.permute(o_wo_a_g, [1, 0, 2])
        o_flat_padded = ttnn.reshape(
            o_wo_a_perm, [TILE, N_GROUPS * O_LORA_RANK])

        # 13. SUMMA wo_b: o_flat [TILE, 8192] @ wo_b_w [8192, DIM] -> [TILE, DIM].
        matmul_wo_b(o_flat_padded, wo_b_w_tt, state["wo_b_out_padded"])

        # 13. Slice + reshape -> [B, S, DIM].
        out_row = ttnn.slice(state["wo_b_out_padded"], [0, 0], [B * S, DIM])
        out_3d = ttnn.reshape(out_row, [B, S, DIM])
        ttnn.copy(out_3d, out)

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

        # Two parallel copies of kv_cache so the kernel and reference each
        # mutate their own (paged_update_cache writes in place).
        kv_cache_ref_tt = ttnn.as_tensor(kv_cache.contiguous(),
                                         dtype=ttnn.bfloat16, **rep)
        kv_cache_kernel_tt = ttnn.as_tensor(kv_cache.contiguous(),
                                            dtype=ttnn.bfloat16, **rep)

        q_tt = ttnn.as_tensor(q.contiguous(), dtype=ttnn.bfloat16, **rep)
        kv_tt = ttnn.as_tensor(kv.contiguous(), dtype=ttnn.bfloat16, **rep)
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

        # Sink replicated [1, 1, n_heads, 1] (same shape as DeviceSparseAttn
        # uses internally) for the reference path.
        sink_4d = attn_sink.to(torch.bfloat16).view(1, 1, N_HEADS, 1).contiguous()
        sink_4d_tt = ttnn.as_tensor(sink_4d, dtype=ttnn.bfloat16, **rep)

        # Padded sink for the fused softmax-with-sink kernel: [H, TILE] where
        # col 0 = sink_per_head, cols 1..TILE-1 = -1e9 (sentinel so they have
        # no effect on max/sum). Plus an all-ones scaler tile for reduce.
        sink_padded = torch.full(
            (N_HEADS, TILE), -1.0e9, dtype=torch.bfloat16)
        sink_padded[:, 0] = attn_sink.to(torch.bfloat16)
        sink_padded_tt = ttnn.as_tensor(
            sink_padded.contiguous(), dtype=ttnn.bfloat16, **rep)
        sm_scaler = torch.ones((TILE, TILE), dtype=torch.bfloat16)
        sm_scaler_tt = ttnn.as_tensor(
            sm_scaler.contiguous(), dtype=ttnn.bfloat16, **rep)

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
            mesh, q_tt, kv_tt, kv_cache_ref_tt, kv_slot_tt, topk_idxs_tt,
            attn_sink, cos_full_tt, sin_full_tt, start_pos_tt,
            wo_a_w_tt, wo_b_w_tt, softmax_scale, sharded_memcfg)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_dsparse_kernel(
            mesh, cos_full, sin_full, sharded_memcfg, softmax_scale)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(q_tt, kv_tt, kv_cache_kernel_tt, kv_slot_tt, topk_idxs_tt,
               sink_padded_tt, sm_scaler_tt, cos_full_tt, sin_full_tt,
               start_pos_tt, wo_a_w_tt, wo_b_w_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-Dsparse", ref_host, kernel_host)

        benchmark("Lk-Dsparse ref",
                  lambda: reference(
                      mesh, q_tt, kv_tt, kv_cache_ref_tt, kv_slot_tt,
                      topk_idxs_tt, attn_sink, cos_full_tt, sin_full_tt,
                      start_pos_tt, wo_a_w_tt, wo_b_w_tt, softmax_scale,
                      sharded_memcfg),
                  mesh)
        benchmark("Lk-Dsparse ttl",
                  lambda: kernel(
                      q_tt, kv_tt, kv_cache_kernel_tt, kv_slot_tt,
                      topk_idxs_tt, sink_padded_tt, sm_scaler_tt,
                      cos_full_tt, sin_full_tt, start_pos_tt,
                      wo_a_w_tt, wo_b_w_tt, out_tt),
                  mesh)

        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
