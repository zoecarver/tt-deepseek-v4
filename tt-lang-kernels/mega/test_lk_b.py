"""Lk-B PCC test: q_norm + wq_b (sans wq_b all_gather).

Reference is the exact ttnn / ttl chain: reshape into [B*S, q_lora_rank]
→ DeviceRMSNorm.forward_device(q_norm) → reshape to [B, S, q_lora_rank]
→ ttnn.matmul(wq_b). The all_gather after the matmul is excluded.

All tt-lang kernel definitions live in this file (rmsnorm + SUMMA
matmul). The reference path still calls into inference.py's wrappers so
the comparison is apples-to-apples; the candidate path uses only the
local @ttl.operation definitions so they can be fused / re-tuned here
without round-tripping through inference.py.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import DeviceRMSNorm, _RMS_TILE


Q_LORA_RANK = 1024
N_HEADS = 64
HEAD_DIM = 512
N = N_HEADS * HEAD_DIM    # 32768
NORM_EPS = 1e-6
B, S = 1, 1
TILE = 32


def _make_rmsnorm_kernel(num_row_tiles: int, h_tiles: int,
                         rms_eps: float, inv_D: float):
    """RMSNorm kernel inlined from inference.py / tt-lang-kernels/rmsnorm.py.

    Streams one row-tile (32 tokens) per core. Two passes over x per
    row-tile: sum(x^2), then gamma * rsqrt(mean(x^2) + eps) apply.
    """

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
    """Pure-SUMMA matmul kernel modelled on tt-lang-kernels/attention_matmul.py.

    Output = a @ w. A is row-mcast across Np cores, B is column-mcast across
    Mp cores. Each core owns an M_BPN x N_BPN output sub-grid.
    """
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


def make_lk_b_kernel(mesh, gamma_cpu):
    """Mega kernel for Lk-B = rmsnorm(q_lora, gamma) @ wq_b.

    Two tt-lang dispatches (rmsnorm + SUMMA matmul) defined locally in
    this file. Wrapper handles [1,1,K] -> [TILE,K] padding and
    [TILE,N] -> [1,1,N] slice/reshape glue.
    """
    state: dict = {}
    M_PAD = TILE
    rms_kernel = _make_rmsnorm_kernel(
        num_row_tiles=1, h_tiles=Q_LORA_RANK // TILE,
        rms_eps=NORM_EPS, inv_D=1.0 / Q_LORA_RANK)
    matmul_kernel = _make_summa_matmul_kernel(
        M=M_PAD, K=Q_LORA_RANK, N=N,
        block_cfg=(1, 8, 8), part_cfg=(1, 8, 1))

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    # Pack gamma to [TILE, K] (rmsnorm kernel reads gamma[0, h] but
    # tile granularity needs the tile's full row dim populated).
    gamma_packed = gamma_cpu.flatten().to(torch.bfloat16) \
        .unsqueeze(0).expand(TILE, -1).contiguous()
    gamma_tt = ttnn.as_tensor(gamma_packed, dtype=ttnn.bfloat16, **rep)

    def lk_b_kernel(q_lora, wq_b_w, out):
        if "scratch" not in state:
            state["scaler_tt"] = ttnn.from_torch(
                torch.ones((TILE, TILE), dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state["normed_tt"] = ttnn.from_torch(
                torch.zeros((M_PAD, Q_LORA_RANK), dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state["y_padded_tt"] = ttnn.from_torch(
                torch.zeros((M_PAD, N), dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state["scratch"] = True

        # 1. Pad input [1, 1, K] -> [TILE, K].
        q_2d = ttnn.reshape(q_lora, [B * S, Q_LORA_RANK])
        x_padded = ttnn.pad(
            q_2d, padding=[(0, M_PAD - B * S), (0, 0)], value=0.0)

        # 2. RMSNorm -> normed_tt [TILE, K].
        rms_kernel(x_padded, gamma_tt, state["scaler_tt"], state["normed_tt"])

        # 3. Matmul -> y_padded_tt [TILE, N].
        matmul_kernel(state["normed_tt"], wq_b_w, state["y_padded_tt"])

        # 4. Slice + reshape + copy into the test-provided [1, 1, N] out.
        y_row = ttnn.slice(state["y_padded_tt"], [0, 0], [B * S, N])
        y_3d = ttnn.reshape(y_row, [B, S, N])
        ttnn.copy(y_3d, out)

    return lk_b_kernel


def reference(mesh, q_lora_tt, gamma_cpu, wq_b_w_tt):
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=gamma_cpu, eps=NORM_EPS)
    # Mirror DeviceAttention.forward_device's q_norm step:
    #   q_lora_2d = ttnn.reshape(q_lora, [B*S, q_lora_rank])
    #   qr_2d = self._rmsnorm_device(self.q_norm_dev, q_lora_2d, B*S)
    #   qr_tt = ttnn.reshape(qr_2d, [B, S, q_lora_rank])
    q_lora_2d = ttnn.reshape(q_lora_tt, [B * S, Q_LORA_RANK])
    Mpad = -(-(B * S) // _RMS_TILE) * _RMS_TILE
    if (B * S) < Mpad:
        q_lora_2d = ttnn.pad(
            q_lora_2d,
            padding=[(0, Mpad - (B * S)), (0, 0)], value=0.0)
    qr_padded = rmsn.forward_device(q_lora_2d, B * S)
    qr_2d = ttnn.slice(qr_padded, [0, 0], [B * S, Q_LORA_RANK])
    qr_tt = ttnn.reshape(qr_2d, [B, S, Q_LORA_RANK])
    return ttnn.matmul(qr_tt, wq_b_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        q_lora = torch.randn(1, 1, Q_LORA_RANK, dtype=torch.bfloat16) * 0.1
        gamma = torch.ones(Q_LORA_RANK, dtype=torch.bfloat16)
        wq_b_w = torch.randn(Q_LORA_RANK, N, dtype=torch.bfloat16) * 0.02

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        q_lora_tt = ttnn.as_tensor(q_lora.contiguous(), dtype=ttnn.bfloat16, **rep)
        wq_b_w_tt = ttnn.as_tensor(wq_b_w.contiguous(), dtype=ttnn.bfloat16, **rep)

        ref_out_tt = reference(mesh, q_lora_tt, gamma, wq_b_w_tt)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_b_kernel(mesh, gamma)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(q_lora_tt, wq_b_w_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-B", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
