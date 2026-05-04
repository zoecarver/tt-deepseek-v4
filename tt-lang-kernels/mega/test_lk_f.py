"""Lk-F PCC test: gate post-process + routed experts (sans routed all_reduce).

Reference covers everything from the shared-expert all_reduce up to the
routed-expert all_reduce. Mirrors `MoE.forward_device` and
`MoE._forward_device_routed_cached`, sans the closing all_reduce.

For tractable per-expert dispatch count, this test uses a reduced
per_chip=8 (matches a 32-chip Galaxy shard) instead of N_ROUTED=256, and
rigs gate_bias[0..7] to dominate so topk always selects indices 0..7.
chip_ids=[0..7] then matches all 8 selected indices, exercising the full
mask + sum-across-experts path.

Pipeline (2 inlined tt-lang factories, 25 dispatches total + 1 gate post):
  1 gate matmul + 1 gate post (sqrt(softplus(raw)) + bias) +
  8 × (w1 + w3 + w2) per-expert matmuls = 25 SUMMAs + 1 elementwise.

ttnn glue (TODO: mega):
  - topk + gather + sum + div + multiply (no tt-lang topk/gather)
  - mask construction: eq + typecast bool->bf16 + multiply + sum (no tt-lang
    int32 eq, no bool->bf16 typecast)
  - SwiGLU body: clamp + silu + multiply (Lk-E pattern; tt-lang
    clamp-via-relu has open compile issue per swiglu.py note)
  - per-expert weight slicing (could lower if we tile-major w1/w2/w3)
  - per-expert mask multiply + sum-into-accumulator
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark


DIM = 4096
INTER_DIM = 2048
N_ROUTED = 256
TOPK = 8
PER_CHIP = 8
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
B = 1
TILE = 32


def _make_summa_matmul_kernel(M: int, K: int, N: int,
                              block_cfg, part_cfg,
                              fp32_dest_acc_en: bool = True):
    """Pure SUMMA matmul. A row-mcast across Np cores; B col-mcast across Mp."""
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


def _make_matmul_gate_post_kernel(M: int, K: int, N: int,
                                   block_cfg, part_cfg,
                                   fp32_dest_acc_en: bool = True):
    """SUMMA gate matmul fused with sqrt(softplus(raw)) + bias post-process.

    Each (m_p, n_p) core accumulates its output tile, then applies the post
    in compute and writes both scores and biased outputs from dm_write.
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
    def matmul_gate_post(a, w, bias, scores_out, biased_out):
        a_pipes = [ttl.Pipe(src=(0, m_p), dst=(slice(0, Np), m_p))
                   for m_p in range(Mp)]
        mcast_a_net = ttl.PipeNet(a_pipes)
        b_pipes = [ttl.Pipe(src=(n_p, 0), dst=(n_p, slice(0, Mp)))
                   for n_p in range(Np)]
        mcast_b_net = ttl.PipeNet(b_pipes)

        a_cb = ttl.make_dataflow_buffer_like(a, shape=(bm, bk), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(w, shape=(bk, bn), block_count=2)
        bias_cb = ttl.make_dataflow_buffer_like(
            bias, shape=(bm, bn), block_count=2)
        raw_cb = ttl.make_dataflow_buffer_like(
            scores_out, shape=(bm, bn), block_count=2)
        s_scratch = ttl.make_dataflow_buffer_like(
            scores_out, shape=(bm, bn), block_count=2)
        scores_cb = ttl.make_dataflow_buffer_like(
            scores_out, shape=(bm, bn), block_count=2)
        biased_cb = ttl.make_dataflow_buffer_like(
            biased_out, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(M_BPN):
                for _ in range(N_BPN):
                    p = raw_cb.reserve()
                    for _ in range(Kb):
                        a_blk = a_cb.wait()
                        b_blk = b_cb.wait()
                        p += a_blk @ b_blk
                    r = raw_cb.wait()
                    b = bias_cb.wait()
                    s_scratch.reserve().store(
                        ttl.math.sqrt(
                            ttl.math.log(
                                ttl.math.exp(r) + ttl.math.fill(r, 1.0))))
                    s = s_scratch.wait()
                    scores_cb.reserve().store(s)
                    biased_cb.reserve().store(s + b)

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
                    ttl.copy(bias[mr:mr + bm, nc:nc + bn],
                             bias_cb.reserve()).wait()
                    ttl.copy(scores_cb.wait(),
                             scores_out[mr:mr + bm, nc:nc + bn]).wait()
                    ttl.copy(biased_cb.wait(),
                             biased_out[mr:mr + bm, nc:nc + bn]).wait()

    return matmul_gate_post


def make_lk_f_kernel(mesh, gate_bias_cpu):
    """Mega kernel for Lk-F.

    1 gate matmul + 1 fused gate postprocess (sqrt(softplus) + bias) +
    per-expert (matmul w1 + matmul w3 + ttnn SwiGLU + matmul w2).
    Final mask multiply + sum across experts in ttnn (TODO mega).
    """
    # Fused gate matmul + sqrt(softplus) + bias.  M=TILE, K=DIM=4096, N=N_ROUTED=256.
    # Mt=1, Kt=128, Nt=8. block=(1,1,8) part=(1,8,1) -> 8 cores, N_BPN=1, Kb=16.
    matmul_gate_post = _make_matmul_gate_post_kernel(
        M=TILE, K=DIM, N=N_ROUTED,
        block_cfg=(1, 1, 8), part_cfg=(1, 8, 1))

    # SUMMA: w1 / w3. M=TILE, K=DIM=4096, N=INTER_DIM=2048.
    # Mt=1, Kt=128, Nt=64. block=(1,8,4) part=(1,8,1) -> 8 cores, N_BPN=1, Kb=32.
    matmul_w1 = _make_summa_matmul_kernel(
        M=TILE, K=DIM, N=INTER_DIM,
        block_cfg=(1, 8, 4), part_cfg=(1, 8, 1))
    matmul_w3 = matmul_w1

    # SUMMA: w2. M=TILE, K=INTER_DIM=2048, N=DIM=4096.
    # Mt=1, Kt=64, Nt=128. block=(1,16,4) part=(1,8,1) -> 8 cores, N_BPN=1, Kb=16.
    # (16-col grid exceeds BH 11x10 compute grid; widen block to stay <=11 cols.)
    matmul_w2 = _make_summa_matmul_kernel(
        M=TILE, K=INTER_DIM, N=DIM,
        block_cfg=(1, 16, 4), part_cfg=(1, 8, 1))

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    # Pre-pad gate_bias from [1, N_ROUTED] to [TILE, N_ROUTED]. Padding rows
    # are zeroed; only row 0 carries the real bias and downstream slices to it.
    bias_padded_cpu = torch.zeros(TILE, N_ROUTED, dtype=torch.bfloat16)
    bias_padded_cpu[0, :] = gate_bias_cpu.view(N_ROUTED)
    bias_padded_tt = ttnn.from_torch(
        bias_padded_cpu, dtype=ttnn.bfloat16, **rep)

    state: dict = {"bias_padded": bias_padded_tt}

    def lk_f_kernel(x_tt, gate_w_tt, gate_bias_tt,
                    w1_tt, w2_tt, w3_tt, chip_ids_4d_tt):
        if "init" not in state:
            state["scores_padded"] = ttnn.from_torch(
                torch.zeros(TILE, N_ROUTED, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["biased_padded"] = ttnn.from_torch(
                torch.zeros(TILE, N_ROUTED, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, **rep)
            state["init"] = True

        # 1. Pad x [1, DIM] -> [TILE, DIM] for SUMMA.
        # TODO: mega fusion blocked: ttnn.pad/reshape sub-tile -> tile-aligned.
        x_2d = ttnn.reshape(x_tt, [B, DIM])
        x_padded = ttnn.pad(x_2d, padding=[(0, TILE - B), (0, 0)],
                            value=0.0)

        # 2. Fused gate matmul + sqrt(softplus) + bias post-process.
        matmul_gate_post(x_padded, gate_w_tt, state["bias_padded"],
                         state["scores_padded"], state["biased_padded"])
        scores_2d = ttnn.slice(state["scores_padded"], [0, 0], [B, N_ROUTED])
        biased_2d = ttnn.slice(state["biased_padded"], [0, 0], [B, N_ROUTED])
        scores = ttnn.reshape(scores_2d, [1, B, N_ROUTED])
        biased = ttnn.reshape(biased_2d, [1, B, N_ROUTED])

        # 4. topk + gather + normalize + scale.
        # TODO: mega fusion blocked: no tt-lang topk; no tt-lang gather.
        _, indices_tt = ttnn.topk(
            biased, k=TOPK, dim=-1, largest=True, sorted=True)
        gathered = ttnn.gather(scores, dim=-1, index=indices_tt)
        wsum = ttnn.sum(gathered, dim=-1, keepdim=True)
        normed = ttnn.div(gathered, wsum)
        weights_tt = ttnn.multiply(normed, ROUTE_SCALE)

        # 5. Mask construction.
        # TODO: mega fusion blocked: ttnn.eq on int32 + typecast bool->bf16 +
        # broadcasted multiply + sum across topk dim. Lowering needs tt-lang
        # int32 compare and bool->bf16 cast.
        weights_4d = ttnn.reshape(weights_tt, [1, 1, 1, TOPK])
        if indices_tt.dtype != ttnn.int32:
            indices_int32 = ttnn.typecast(indices_tt, ttnn.int32)
        else:
            indices_int32 = indices_tt
        indices_4d = ttnn.reshape(indices_int32, [1, 1, 1, TOPK])
        match = ttnn.eq(indices_4d, chip_ids_4d_tt)
        match_bf16 = ttnn.typecast(match, ttnn.bfloat16)
        match_weighted = ttnn.multiply(weights_4d, match_bf16)
        mask = ttnn.sum(match_weighted, dim=-1, keepdim=True)

        # 6. Batched routed-expert path: w1, w3, swiglu, w2, mask, sum-experts.
        # TODO: mega: replace batched ttnn.matmul x3 with batched-SUMMA tt-lang
        # kernel; SwiGLU body (clamp + silu + multiply) also still in ttnn.
        x_4d = ttnn.reshape(x_tt, [1, 1, 1, DIM])
        x_grouped = ttnn.repeat(x_4d, [1, PER_CHIP, 1, 1])
        y1 = ttnn.matmul(x_grouped, w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y3 = ttnn.matmul(x_grouped, w3_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if SWIGLU_LIMIT > 0:
            ttnn.clamp(y1, max=SWIGLU_LIMIT, output_tensor=y1)
            ttnn.clamp(y3, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT, output_tensor=y3)
        ttnn.silu(y1, output_tensor=y1)
        ttnn.multiply(y1, y3, output_tensor=y1)
        y = ttnn.matmul(y1, w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.multiply(y, mask, output_tensor=y)
        return ttnn.sum(y, dim=1, keepdim=True)

    return lk_f_kernel


def reference(mesh, x_tt, gate_w_tt, gate_bias_tt,
              w1_tt, w2_tt, w3_tt, chip_ids_4d_tt, per_chip):
    raw = ttnn.matmul(x_tt, gate_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    scores = ttnn.sqrt(ttnn.softplus(raw))
    biased = ttnn.add(scores, gate_bias_tt)
    _, indices_tt = ttnn.topk(biased, k=TOPK, dim=-1, largest=True, sorted=True)
    gathered = ttnn.gather(scores, dim=-1, index=indices_tt)
    wsum = ttnn.sum(gathered, dim=-1, keepdim=True)
    normed = ttnn.div(gathered, wsum)
    weights_tt = ttnn.multiply(normed, ROUTE_SCALE)

    weights_4d = ttnn.reshape(weights_tt, [1, 1, 1, TOPK])
    indices_int32 = (
        indices_tt if indices_tt.dtype == ttnn.int32
        else ttnn.typecast(indices_tt, ttnn.int32))
    indices_4d = ttnn.reshape(indices_int32, [1, 1, 1, TOPK])
    match = ttnn.eq(indices_4d, chip_ids_4d_tt)
    match_bf16 = ttnn.typecast(match, ttnn.bfloat16)
    ttnn.multiply(weights_4d, match_bf16, output_tensor=match_bf16)
    mask = ttnn.sum(match_bf16, dim=-1, keepdim=True)

    x_4d = ttnn.reshape(x_tt, [1, 1, 1, DIM])
    x_grouped = ttnn.repeat(x_4d, [1, per_chip, 1, 1])
    y1 = ttnn.matmul(x_grouped, w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y3 = ttnn.matmul(x_grouped, w3_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if SWIGLU_LIMIT > 0:
        ttnn.clamp(y1, max=SWIGLU_LIMIT, output_tensor=y1)
        ttnn.clamp(y3, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT, output_tensor=y3)
    ttnn.silu(y1, output_tensor=y1)
    ttnn.multiply(y1, y3, output_tensor=y1)
    y = ttnn.matmul(y1, w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.multiply(y, mask, output_tensor=y)
    y_local = ttnn.sum(y, dim=1, keepdim=True)
    return y_local


def main():
    torch.manual_seed(0)
    mesh = open_mesh(shape=(1, 1))
    mesh_shape = tuple(mesh.shape)
    try:
        per_chip = PER_CHIP
        x = torch.randn(1, DIM, dtype=torch.bfloat16) * 0.1
        gate_w = torch.randn(DIM, N_ROUTED, dtype=torch.bfloat16) * 0.02
        # Rig gate_bias so topk always picks indices 0..PER_CHIP-1.
        gate_bias = torch.full((1, N_ROUTED), -2.0, dtype=torch.bfloat16)
        gate_bias[0, :PER_CHIP] = 2.0
        w1 = torch.randn(1, per_chip, DIM, INTER_DIM, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(1, per_chip, INTER_DIM, DIM, dtype=torch.bfloat16) * 0.02
        w3 = torch.randn(1, per_chip, DIM, INTER_DIM, dtype=torch.bfloat16) * 0.02
        chip_ids = torch.arange(per_chip, dtype=torch.int32).view(
            1, per_chip, 1, 1)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep)
        gate_w_tt = ttnn.as_tensor(gate_w.contiguous(),
                                   dtype=ttnn.bfloat16, **rep)
        gate_bias_tt = ttnn.as_tensor(gate_bias.contiguous(),
                                      dtype=ttnn.bfloat16, **rep)
        w1_tt = ttnn.as_tensor(w1.contiguous(), dtype=ttnn.bfloat16, **rep)
        w2_tt = ttnn.as_tensor(w2.contiguous(), dtype=ttnn.bfloat16, **rep)
        w3_tt = ttnn.as_tensor(w3.contiguous(), dtype=ttnn.bfloat16, **rep)
        chip_ids_tt = ttnn.as_tensor(chip_ids.contiguous(),
                                     dtype=ttnn.int32, **rep)

        ref_out_tt = reference(mesh, x_tt, gate_w_tt, gate_bias_tt,
                               w1_tt, w2_tt, w3_tt, chip_ids_tt, per_chip)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_f_kernel(mesh, gate_bias)
        kernel_out_tt = kernel(x_tt, gate_w_tt, gate_bias_tt,
                               w1_tt, w2_tt, w3_tt, chip_ids_tt)
        kernel_host = download_chip0(mesh, mesh_shape, kernel_out_tt)

        ok = report_pcc("Lk-F", ref_host, kernel_host)

        benchmark("Lk-F ref",
                  lambda: reference(mesh, x_tt, gate_w_tt, gate_bias_tt,
                                    w1_tt, w2_tt, w3_tt, chip_ids_tt, per_chip),
                  mesh)
        benchmark("Lk-F ttl",
                  lambda: kernel(x_tt, gate_w_tt, gate_bias_tt,
                                 w1_tt, w2_tt, w3_tt, chip_ids_tt),
                  mesh)

        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
