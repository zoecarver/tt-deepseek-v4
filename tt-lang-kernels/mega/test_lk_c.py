"""Lk-C PCC test: q_rsqrt_norm + q rotary + wkv (sans wkv all_gather).

Reference covers the half of `DeviceAttention.forward_device` between
the wq_b all_gather and the wkv all_gather:
- reshape q_full to [B,S,H,D]
- per-head rsqrt-norm
- pick cos/sin via embedding(start_pos, table); reshape to [1,S,1,rd/2]
- slice q nope/rope, rotary on rope half, concat
- ttnn.matmul(x, wkv) — partial pre-all_gather

The wq_a/wq_b path that produces q_full / qr is upstream (Lk-A/B).
For this test we feed in random q_full and x directly.

q_rsqrt_norm is now lowered to the inlined rmsnorm tt-lang kernel
(per-head rsqrt-norm = rmsnorm with gamma=ones). The wkv matmul is
the inlined SUMMA kernel.

TODO: lower the rotary to tt-lang. `_device_apply_rotary_interleaved`
is still pure ttnn (slice / multiply / concat). Lowering needs a real
design pass: at decode the rope half is rd=64 wide which is 2 tiles,
the (real, imag) pairs are interleaved along the last axis (positions
0,2,4,... vs 1,3,5,...) so a tile-tile @ttl.operation must either
transpose pairs to separate tiles first, or read sub-tile granularity
which tt-lang doesn't support natively. Pick a layout for cos/sin that
matches a chosen split layout and keep the tt-lang body to four
elementwise multiplies + one add and one sub.
ttnn.embedding(start_pos, cos/sin) for the table lookup is also still
ttnn — it depends on a device int32 tensor and is not a candidate for
fusion into the rotary kernel.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

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


def make_lk_c_kernel(mesh):
    """Mega kernel for Lk-C = q_rsqrt_norm + q rotary + wkv matmul.

    q_rsqrt_norm is the inlined rmsnorm kernel with gamma=ones (per-head
    rsqrt-norm IS rmsnorm; both compute x * rsqrt(mean(x^2) + eps) and
    differ only by the multiplicative gamma weight). The wkv matmul is
    the inlined SUMMA kernel. The rotary helper is still pure-ttnn —
    see the file docstring TODO for the design notes.
    """
    rms_kernel = _make_rmsnorm_kernel(
        num_row_tiles=N_HEADS // TILE,         # 64 / 32 = 2
        h_tiles=HEAD_DIM // TILE,              # 512 / 32 = 16
        rms_eps=NORM_EPS, inv_D=1.0 / HEAD_DIM)
    matmul_kernel = _make_summa_matmul_kernel(
        M=TILE, K=DIM, N=HEAD_DIM,
        block_cfg=(1, 4, 8), part_cfg=(1, 4, 1))

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
            state["wkv_padded_tt"] = _alloc_replicated_zeros((TILE, HEAD_DIM))
            state["scratch"] = True

        rd_half = ROPE_HEAD_DIM // 2

        # q_rsqrt_norm via rmsnorm tt-lang kernel.
        q_2d = ttnn.reshape(q_full, [N_HEADS, HEAD_DIM])
        rms_kernel(q_2d, gamma_q_tt, sc_tt, state["q_normed_2d_tt"])
        q_tt = ttnn.reshape(state["q_normed_2d_tt"], [B, S, N_HEADS, HEAD_DIM])

        # TODO: lower this rotary block to a @ttl.operation kernel; see
        # the file docstring for the design notes.
        cos = ttnn.embedding(start_pos, cos_full, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(start_pos, sin_full, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, S, 1, rd_half])
        sin = ttnn.reshape(sin, [1, S, 1, rd_half])
        q_nope = ttnn.slice(
            q_tt, [0, 0, 0, 0], [B, S, N_HEADS, HEAD_DIM - ROPE_HEAD_DIM])
        q_rope = ttnn.slice(
            q_tt, [0, 0, 0, HEAD_DIM - ROPE_HEAD_DIM], [B, S, N_HEADS, HEAD_DIM])
        q_rope = _device_apply_rotary_interleaved(
            ttnn, q_rope, cos, sin, inverse=False)
        q_full_out = ttnn.concat([q_nope, q_rope], dim=-1)
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

        kernel = make_lk_c_kernel(mesh)
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
        sys.exit(0 if (ok_q and ok_kv) else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
