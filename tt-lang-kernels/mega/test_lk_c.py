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

Three inlined tt-lang dispatches:
  - rmsnorm (q_rsqrt_norm)
  - SUMMA matmul x P (rotary swap_pairs)
  - rotary combine (x * cos_extended + x_swap * sin_signed)
  - SUMMA matmul (wkv)

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
MAX_SEQ_LEN = 512
NORM_EPS = 1e-6
B, S = 1, 1
TILE = 32


def _make_rmsnorm_kernel(num_row_tiles: int, h_tiles: int,
                         rms_eps: float, inv_D: float):
    """RMSNorm kernel inlined from inference.py / tt-lang-kernels/rmsnorm.py."""

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
    """Pure-SUMMA matmul kernel modelled on tt-lang-kernels/attention_matmul.py."""
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    if Kp != 1:
        raise ValueError(f"K_parts must be 1, got {Kp}")
    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape: Mt={Mt} Nt={Nt} Kt={Kt} "
            f"block=(bm={bm}, bn={bn}, bk={bk})")
    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Mb % Mp or Nb % Np:
        raise ValueError(f"block/part mismatch: Mb={Mb} Nb={Nb} Mp={Mp} Np={Np}")
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
    """out = x * cos + x_swap * sin (elementwise).

    cos/sin tiles already carry the same data on every TILE row (the
    host pre-replicates the table along the row axis), so the kernel
    just multiplies tile-by-tile - no broadcast op needed.
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
                    # cos/sin: row 0 holds the full TILE-replicated table.
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

    Three inlined tt-lang kernels (rmsnorm, swap-SUMMA + rotary combine,
    wkv-SUMMA). The rotary swap is implemented as a small 64x64 matmul
    against a block-diagonal swap matrix; the combine is one elementwise
    pass. Cos/sin tables are pre-processed on host (interleave-repeat
    cos, sign-bake sin) so the kernel sees tile-aligned tables.
    """
    rms_kernel = _make_rmsnorm_kernel(
        num_row_tiles=N_HEADS // TILE,         # 64 / 32 = 2
        h_tiles=HEAD_DIM // TILE,              # 512 / 32 = 16
        rms_eps=NORM_EPS, inv_D=1.0 / HEAD_DIM)
    matmul_kernel = _make_summa_matmul_kernel(
        M=TILE, K=DIM, N=HEAD_DIM,
        block_cfg=(1, 4, 8), part_cfg=(1, 4, 1))
    # Swap SUMMA: M=N_HEADS=64, K=N=rd=64. Mt=Kt=Nt=2. block=(1,1,2)
    # part=(2,2,1) -> 4 cores, M_BPN=N_BPN=1, Kb=1.
    swap_kernel = _make_summa_matmul_kernel(
        M=N_HEADS, K=ROPE_HEAD_DIM, N=ROPE_HEAD_DIM,
        block_cfg=(1, 1, 2), part_cfg=(2, 2, 1))
    rotary_combine_kernel = _make_rotary_combine_kernel(
        num_row_tiles=N_HEADS // TILE,        # 64 / 32 = 2
        num_h_tiles=ROPE_HEAD_DIM // TILE)    # 64 / 32 = 2

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    # gamma=ones for q_rsqrt_norm. Packed [TILE, HEAD_DIM] so the rmsnorm
    # kernel's `gamma[0, h]` reads cover the full row.
    gamma_q_packed = torch.ones(HEAD_DIM, dtype=torch.bfloat16) \
        .unsqueeze(0).expand(TILE, -1).contiguous()
    gamma_q_tt = ttnn.as_tensor(gamma_q_packed, dtype=ttnn.bfloat16, **rep)
    sc_tt = ttnn.from_torch(
        torch.ones((TILE, TILE), dtype=torch.bfloat16),
        device=mesh, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    # Pre-processed rotary tables. Forward rotary in Lk-C.
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
            state["q_normed_2d_tt"] = _alloc_replicated_zeros(
                (N_HEADS, HEAD_DIM))
            state["q_rope_swap_tt"] = _alloc_replicated_zeros(
                (N_HEADS, ROPE_HEAD_DIM))
            state["q_rope_rot_tt"] = _alloc_replicated_zeros(
                (N_HEADS, ROPE_HEAD_DIM))
            state["wkv_padded_tt"] = _alloc_replicated_zeros((TILE, HEAD_DIM))
            state["scratch"] = True

        # q_rsqrt_norm via rmsnorm tt-lang kernel.
        q_2d = ttnn.reshape(q_full, [N_HEADS, HEAD_DIM])
        rms_kernel(q_2d, gamma_q_tt, sc_tt, state["q_normed_2d_tt"])

        # Rotary on the rope half.
        # TODO: mega fusion blocked: ttnn used for embedding(start_pos, ...)
        # to look up cos/sin (depends on a device uint32 index, no tt-lang
        # gather primitive). Slice/pad/reshape stay in ttnn for the same
        # reason - they're light data movement around the lookup. The
        # actual rotary math (swap_pairs + cos/sin combine) is in tt-lang.
        cos_b_2d = ttnn.embedding(start_pos, cos_ext_tt, layout=ttnn.TILE_LAYOUT)
        sin_b_2d = ttnn.embedding(start_pos, sin_signed_tt, layout=ttnn.TILE_LAYOUT)
        cos_b = ttnn.reshape(cos_b_2d, [TILE, ROPE_HEAD_DIM])
        sin_b = ttnn.reshape(sin_b_2d, [TILE, ROPE_HEAD_DIM])

        q_normed_4d = ttnn.reshape(
            state["q_normed_2d_tt"], [B, S, N_HEADS, HEAD_DIM])
        q_nope = ttnn.slice(
            q_normed_4d, [0, 0, 0, 0],
            [B, S, N_HEADS, HEAD_DIM - ROPE_HEAD_DIM])
        q_rope = ttnn.slice(
            q_normed_4d, [0, 0, 0, HEAD_DIM - ROPE_HEAD_DIM],
            [B, S, N_HEADS, HEAD_DIM])
        q_rope_2d = ttnn.reshape(q_rope, [N_HEADS, ROPE_HEAD_DIM])

        # SUMMA: q_rope_swap = q_rope @ P.
        swap_kernel(q_rope_2d, P_tt, state["q_rope_swap_tt"])

        # tt-lang combine: q_rope_rot = q_rope * cos_b + q_rope_swap * sin_b.
        rotary_combine_kernel(
            q_rope_2d, state["q_rope_swap_tt"], cos_b, sin_b,
            state["q_rope_rot_tt"])

        q_rope_rot_4d = ttnn.reshape(
            state["q_rope_rot_tt"], [B, S, N_HEADS, ROPE_HEAD_DIM])
        q_full_out = ttnn.concat([q_nope, q_rope_rot_4d], dim=-1)
        ttnn.copy(q_full_out, q_out)

        # wkv matmul (tt-lang SUMMA).
        x_2d = ttnn.reshape(x, [B * S, DIM])
        x_padded = ttnn.pad(
            x_2d, padding=[(0, TILE - B * S), (0, 0)], value=0.0)
        matmul_kernel(x_padded, wkv_w, state["wkv_padded_tt"])
        wkv_row = ttnn.slice(state["wkv_padded_tt"], [0, 0], [B * S, HEAD_DIM])
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
