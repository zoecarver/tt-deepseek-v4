"""TT-Lang fused matmul + rmsnorm — SUMMA mcast layout, Kp=1.

Computes `out = rmsnorm(x, gamma, eps) @ W` in a single kernel by exploiting
the fact that rms-normalization scales each row by a scalar `1/rms[m]` that
commutes with the K-linear matmul:

    y[m,k]   = x[m,k] * gamma[k] / rms[m]
    out[m,n] = sum_k y[m,k] * W[k,n]
             = (1/rms[m]) * (x @ Wg)[m,n]
    where Wg[k,n] = gamma[k] * W[k,n]   (pre-baked on host, one-time per layer).

The kernel sees Wg as its B operand and emits no per-step gamma traffic.
Per M-block, the K loop accumulates the matmul partial AND the per-row
`ssq[m] = sum_k(x[m,k]^2)` simultaneously. After K, the kernel finalizes
`inv_rms[m] = rsqrt(ssq[m]/D + eps)`, broadcasts it across the bn output
tiles, and multiplies the matmul output before writing.

Layout: SUMMA-style 2D grid (Np cols, Mp rows). A row-mcast along Np;
B col-mcast along Mp. Each core owns its M_BPN x N_BPN output sub-grid.

Optimal config for V4-Flash decode: 1x4 mesh with W col-sharded along N.
Each chip runs the kernel on its [K, N/4] slice; outputs are N-sharded.
"""
from __future__ import annotations

from typing import Tuple

import ttl


TILE = 32


def make_kernel(
    M: int,
    K: int,
    N: int,
    block_cfg: Tuple[int, int, int],
    part_cfg: Tuple[int, int, int],
    rms_eps: float,
    *,
    fp32_dest_acc_en: bool = True,
):
    """Build a fused matmul+rmsnorm SUMMA kernel.

    `Wg` (the B operand) must be `gamma[:, None] * W` precomputed on host.
    """
    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg

    if Kp != 1:
        raise ValueError(f"K_parts must be 1, got {Kp}")
    if M % TILE or N % TILE or K % TILE:
        raise ValueError(f"M/K/N must be tile-aligned: M={M} K={K} N={N}")

    Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
    if Mt % bm or Nt % bn or Kt % bk:
        raise ValueError(
            f"block must divide shape: bm={bm} bn={bn} bk={bk} "
            f"Mt={Mt} Nt={Nt} Kt={Kt}"
        )

    Mb, Nb, Kb = Mt // bm, Nt // bn, Kt // bk
    if Mb % Mp or Nb % Np:
        raise ValueError(
            f"block/part mismatch: Mb={Mb} Nb={Nb} Mp={Mp} Np={Np}"
        )

    M_BPN = Mb // Mp
    N_BPN = Nb // Np
    inv_D = 1.0 / float(K)

    COL = Np
    ROW = Mp

    @ttl.operation(
        grid=(COL, ROW),
        fp32_dest_acc_en=fp32_dest_acc_en,
        options="--no-ttl-reduce-full-fp32",
    )
    def fused_kernel(a, w_g, scaler, out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p)) for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp))) for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w_g, shape=(bk, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)

        acc_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)
        xsq_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        ssq_step_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, 1), block_count=2)
        ssq_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, 1), block_count=2)
        inv_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, 1), block_count=2)
        inv_bc_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            sc = sc_cb.wait()
            for local_mb in range(M_BPN):
                # First N output block: matmul + ssq fused over K.
                a0 = a_cb.wait()
                b0 = b_cb.wait()
                acc_cb.reserve().store(a0 @ b0)
                xsq_cb.reserve().store(a0 * a0)
                ssq_cb.reserve().store(
                    ttl.math.reduce_sum(xsq_cb.wait(), sc, dims=[1])
                )
                for _ in range(Kb - 1):
                    ak = a_cb.wait()
                    bk_blk = b_cb.wait()
                    prev_acc = acc_cb.wait()
                    acc_cb.reserve().store(prev_acc + ak @ bk_blk)
                    xsq_cb.reserve().store(ak * ak)
                    ssq_step_cb.reserve().store(
                        ttl.math.reduce_sum(xsq_cb.wait(), sc, dims=[1])
                    )
                    prev_ssq = ssq_cb.wait()
                    ssq_cb.reserve().store(prev_ssq + ssq_step_cb.wait())

                sq = ssq_cb.wait()
                inv_t = inv_cb.reserve()
                inv_t.store(
                    ttl.math.broadcast(
                        ttl.math.rsqrt(
                            sq * ttl.math.fill(sq, inv_D)
                            + ttl.math.fill(sq, rms_eps)
                        ),
                        inv_t,
                        dims=[1],
                    )
                )
                inv_bc_t = inv_bc_cb.reserve()
                inv_bc_t.store(
                    ttl.math.broadcast(inv_cb.wait(), inv_bc_t, dims=[1])
                )
                inv = inv_bc_cb.wait()  # held across the N loop below.

                acc_done = acc_cb.wait()
                out_cb.reserve().store(acc_done * inv)

                # Subsequent N output blocks: matmul only, apply inv.
                for _ in range(N_BPN - 1):
                    a0n = a_cb.wait()
                    b0n = b_cb.wait()
                    acc_cb.reserve().store(a0n @ b0n)
                    for _ in range(Kb - 1):
                        akn = a_cb.wait()
                        bkn = b_cb.wait()
                        prev_acc_n = acc_cb.wait()
                        acc_cb.reserve().store(prev_acc_n + akn @ bkn)
                    acc_n = acc_cb.wait()
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
                            ttl.copy(a[mr : mr + bm, kc : kc + bk], a_blk).wait()
                            ttl.copy(a_blk, pipe).wait()

                        mcast_a_net.if_src(read_a)
                        mcast_a_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, a_blk).wait(),)
                        )

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
                            ttl.copy(w_g[kc : kc + bk, nc : nc + bn], b_blk).wait()
                            ttl.copy(b_blk, pipe).wait()

                        mcast_b_net.if_src(read_b)
                        mcast_b_net.if_dst(
                            lambda pipe: (ttl.copy(pipe, b_blk).wait(),)
                        )
                    o = out_cb.wait()
                    ttl.copy(o, out[mr : mr + bm, nc : nc + bn]).wait()

    return fused_kernel


# Hand-picked block/part plans for V4-Flash decode shapes (post col-shard).
# (M, K, N_per_chip) -> ((bm, bn, bk), (Mp, Np, Kp)). N_per_chip is N/nchips
# at deployment; e.g., N_full=1024 col-sharded across 4 chips => N_per_chip=256.
SHAPE_PLANS = {
    # full N -> N/4 per chip on a 1x4 mesh.
    (32, 4096, 256):  ((1, 1, 8), (1, 8, 1)),  # full N=1024 (attn_norm -> wq_a)
    (32, 4096, 512):  ((1, 2, 8), (1, 8, 1)),  # full N=2048 (ffn_norm -> w1/w3)
    (32, 4096, 1024): ((1, 4, 8), (1, 8, 1)),  # full N=4096 (generic)
    (32, 4096, 2048): ((1, 4, 8), (1, 8, 1)),
    (32, 4096, 4096): ((1, 4, 8), (1, 8, 1)),
}


# -----------------------------------------------------------------------------
# Test harness — mesh4 col-shard, bf16, V4-Flash decode shapes.
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    from harness import assert_pcc, scaler_tile

    RMS_EPS = 1e-6
    NCHIPS = 4

    # (M, K, N_full, label, threshold). N_full is col-sharded across NCHIPS.
    PROD_SHAPES = [
        (32, 4096, 1024, "attn_norm -> wq_a",            0.999),
        (32, 4096, 2048, "ffn_norm -> shared_expert.w1", 0.999),
        (32, 4096, 4096, "generic hidden -> hidden",     0.999),
    ]

    def rmsnorm_then_matmul_ref(x, gamma, w, eps):
        xf = x.float()
        inv_rms = xf.square().mean(-1, keepdim=True).add(eps).rsqrt()
        y = xf * gamma.float() * inv_rms
        return (y @ w.float()).bfloat16()

    def pad_2d(t, rows, cols):
        r, c = t.shape
        if r == rows and c == cols:
            return t
        return torch.nn.functional.pad(t, (0, cols - c, 0, rows - r), value=0.0)

    def to_dev(t, mesh, mapper):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def run_shape(mesh, M, K, N_full, label, threshold):
        N_local = N_full // NCHIPS
        block_cfg, part_cfg = SHAPE_PLANS[(M, K, N_local)]
        kernel = make_kernel(M, K, N_local, block_cfg, part_cfg, RMS_EPS)

        print(f"\n[shape] {label}  M={M} K={K} N={N_full} (per-chip N={N_local})")

        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=torch.bfloat16) * 0.5
        gamma = torch.randn(K, dtype=torch.bfloat16) * 0.1 + 1.0
        w = torch.randn(K, N_full, dtype=torch.bfloat16) * 0.02
        ref = rmsnorm_then_matmul_ref(x, gamma, w, RMS_EPS)

        # Pre-bake gamma into Wg on host.
        wg = (gamma.float()[:, None] * w.float()).bfloat16()

        replicate = ttnn.ReplicateTensorToMesh(mesh)
        shard_n = ttnn.ShardTensor2dMesh(
            mesh, mesh_shape=(1, NCHIPS), dims=(None, -1))
        concat_n = ttnn.ConcatMesh2dToTensor(
            mesh, mesh_shape=(1, NCHIPS), dims=(0, -1))

        x_tt = to_dev(x, mesh, replicate)
        wg_tt = to_dev(wg, mesh, shard_n)
        sc_tt = to_dev(scaler_tile(dtype=torch.bfloat16), mesh, replicate)
        out_tt = to_dev(torch.zeros(M, N_full, dtype=torch.bfloat16), mesh, shard_n)

        kernel(x_tt, wg_tt, sc_tt, out_tt)

        actual_full = ttnn.to_torch(out_tt, mesh_composer=concat_n)
        actual = actual_full[:M, :N_full]
        assert_pcc(ref.float(), actual.float(), threshold=threshold)

        for t in (x_tt, wg_tt, sc_tt, out_tt):
            ttnn.deallocate(t)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    try:
        for M, K, N_full, label, threshold in PROD_SHAPES:
            run_shape(mesh, M, K, N_full, label, threshold)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_mesh_device(mesh)
