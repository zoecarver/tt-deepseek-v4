"""K-split version of pre_norm_fn — fused matmul + rmsnorm with K-axis parallelism.

Same math as pre_norm_fn.py:
  mixes[t, m] = (residual[t] @ fn[m, :]) * rsqrt(sum_k residual[t, k]^2 / D + eps)

The original kernel has num_out_tiles=1 at decode (num_tokens_pad=32 -> 1
output tile), so only one core fires per chip. The 512-tile K-loop runs
serially on that one core. This kernel splits the K-loop across `Kp`
cores via a pipe-gather reduce.

Layout: 2D grid (grid_cols, Kp/grid_cols). Each core's k_p index =
row * grid_cols + col. Root at (0, 0). Two reduce pipe nets gather the
matmul partial and the ssq partial separately. Root finalizes
inv_rms = rsqrt(ssq/D + eps) and multiplies into the matmul output.

Optimal config for V4-Flash decode (D=16384, K_tiles=512): Kp=8,
grid=(8, 1). Speedup vs the single-core baseline is 2.80x (0.69ms ->
0.25ms per call) at PCC 0.989. Kp=16 plateaus on speedup; Kp=32 hits a
PCC floor from `--no-ttl-reduce-full-fp32` accumulating bf16-precision
partials. Kernel is replicated across the 4-chip mesh — each chip runs
its own ksplit independently on the same data.
"""
from __future__ import annotations

import ttl


TILE = 32


def make_kernel(
    num_out_tiles: int,
    K_tiles: int,
    Kp: int,
    rms_eps: float,
    inv_D: float,
    *,
    grid_cols: int = 8,
    fp32_dest_acc_en: bool = True,
):
    """Build the K-split norm_fn kernel.

    Constraints:
    - num_out_tiles must equal 1 (decode pads num_tokens=1 to TILE row).
    - Kp >= 2 and K_tiles % Kp == 0.
    - Kp - 1 <= 32 (block_count limit). Higher Kp would need tree-reduce.
    - For Kp < grid_cols, grid_cols is auto-reduced to Kp.
    """
    if Kp < 2:
        raise ValueError(f"ksplit kernel requires Kp >= 2, got {Kp}")
    if K_tiles % Kp:
        raise ValueError(f"K_tiles={K_tiles} not divisible by Kp={Kp}")
    if num_out_tiles != 1:
        raise ValueError(f"num_out_tiles must be 1, got {num_out_tiles}")
    if Kp < grid_cols:
        grid_cols = Kp
    if Kp % grid_cols:
        raise ValueError(f"Kp={Kp} not divisible by grid_cols={grid_cols}")
    if Kp - 1 > 32:
        raise ValueError(
            f"Kp={Kp} requires block_count={Kp-1} > 32; need tree-reduce"
        )

    K_BPN = K_tiles // Kp
    COL = grid_cols
    ROW = Kp // grid_cols

    @ttl.operation(
        grid=(COL, ROW),
        options="--no-ttl-reduce-full-fp32",
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    def norm_fn_ksplit_kernel(a, b, scaler, out):
        # Each non-root core (col, row) sends its partials to root (0, 0).
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

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        c_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        sq_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_bc_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        asq_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        red_step_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

        recv_c_cb = ttl.make_dataflow_buffer_like(
            out, shape=(1, 1), block_count=max(2, Kp - 1)
        )
        recv_sq_cb = ttl.make_dataflow_buffer_like(
            scaler, shape=(1, 1), block_count=max(2, Kp - 1)
        )

        @ttl.compute()
        def compute():
            col_c, row_c = ttl.node(dims=2)
            sc = sc_cb.wait()

            # K-loop on this k-rank's K-slice.
            a0 = a_cb.wait()
            b0 = b_cb.wait()
            c_cb.reserve().store(a0 @ b0)
            asq_cb.reserve().store(a0 * a0)
            sq_cb.reserve().store(
                ttl.math.reduce_sum(asq_cb.wait(), sc, dims=[1])
            )

            for _ in range(K_BPN - 1):
                ak = a_cb.wait()
                bk = b_cb.wait()
                prev_c = c_cb.wait()
                c_cb.reserve().store(prev_c + ak @ bk)

                asq_cb.reserve().store(ak * ak)
                red_step_cb.reserve().store(
                    ttl.math.reduce_sum(asq_cb.wait(), sc, dims=[1])
                )
                prev_sq = sq_cb.wait()
                sq_cb.reserve().store(prev_sq + red_step_cb.wait())

            if col_c == 0 and row_c == 0:
                # Root: gather Kp-1 matmul + ssq partials, sum, finalize inv_rms.
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

            for kb_local in range(K_BPN):
                kc = k_p * K_BPN + kb_local
                ttl.copy(a[0, kc], a_cb.reserve()).wait()
                ttl.copy(b[kc, 0], b_cb.reserve()).wait()

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


def pack_fn(fn, mhc_mult3, *, dtype):
    """[mhc_mult3, D] -> [D, TILE] padded to TILE cols (B operand layout)."""
    import torch
    m3, D = fn.shape
    if m3 != mhc_mult3:
        raise ValueError(f"fn rows {m3} != mhc_mult3 {mhc_mult3}")
    out = torch.zeros(D, TILE, dtype=dtype)
    out[:, :m3] = fn.T.to(dtype)
    return out.contiguous()


# -----------------------------------------------------------------------------
# Test harness — mesh4 replicated, fp32, V4-Flash decode shape.
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn

    from harness import assert_pcc, scaler_tile

    RMS_EPS = 1e-6
    NCHIPS = 4

    # V4-Flash MHC config.
    HC_MULT = 4
    DIM = 4096
    D = HC_MULT * DIM            # 16384
    MHC_MULT3 = (2 + HC_MULT) * HC_MULT  # 24

    # Optimal Kp for K_tiles=512: 8 cores per chip, PCC ~0.989.
    KP = 8

    def norm_fn_ref(residual, fn, eps):
        inv_rms = residual.square().mean(-1, keepdim=True).add(eps).rsqrt()
        return (residual @ fn.T) * inv_rms

    def to_dev(t, mesh, mapper):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def run_shape(mesh, num_tokens, threshold=0.985):
        num_tokens_pad = -(-num_tokens // TILE) * TILE
        K_tiles = D // TILE
        num_out_tiles = num_tokens_pad // TILE
        if num_out_tiles != 1:
            raise NotImplementedError(
                f"this kernel handles num_out_tiles=1 only "
                f"(got num_tokens={num_tokens})"
            )

        print(f"\n[shape] num_tokens={num_tokens} D={D} K_tiles={K_tiles} "
              f"Kp={KP} per-chip-grid=({KP if KP <= 8 else 8}, "
              f"{max(1, KP // 8)})")

        kernel = make_kernel(
            num_out_tiles=num_out_tiles, K_tiles=K_tiles, Kp=KP,
            rms_eps=RMS_EPS, inv_D=1.0 / D,
        )

        torch.manual_seed(0)
        residual = torch.randn(num_tokens, D, dtype=torch.float32) * 0.1
        fn = torch.randn(MHC_MULT3, D, dtype=torch.float32) * 0.05

        a_packed = torch.zeros(num_tokens_pad, D, dtype=torch.float32)
        a_packed[:num_tokens] = residual
        b_packed = pack_fn(fn, MHC_MULT3, dtype=torch.float32)
        out_zero = torch.zeros(num_tokens_pad, TILE, dtype=torch.float32)
        sc = scaler_tile(dtype=torch.float32)

        ref = norm_fn_ref(residual, fn, RMS_EPS)

        replicate = ttnn.ReplicateTensorToMesh(mesh)
        a_tt = to_dev(a_packed, mesh, replicate)
        b_tt = to_dev(b_packed, mesh, replicate)
        sc_tt = to_dev(sc, mesh, replicate)
        out_tt = to_dev(out_zero, mesh, replicate)

        kernel(a_tt, b_tt, sc_tt, out_tt)

        # Replicated output — every chip's result is identical; read chip 0.
        composer = ttnn.ConcatMesh2dToTensor(
            mesh, mesh_shape=(1, NCHIPS), dims=(0, -1))
        actual_full = ttnn.to_torch(out_tt, mesh_composer=composer)
        actual = actual_full[:num_tokens, :MHC_MULT3]

        assert_pcc(ref, actual, threshold=threshold)

        for t in (a_tt, b_tt, sc_tt, out_tt):
            ttnn.deallocate(t)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    try:
        # Decode pads num_tokens=1 to a single TILE row.
        run_shape(mesh, num_tokens=1, threshold=0.985)
        print("\nALL SHAPES PASSED")
    finally:
        ttnn.close_mesh_device(mesh)
