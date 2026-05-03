"""Lk-D1 PCC test: kv_norm + kv rotary + act_quant_block on nope half.

Reference is the attn.kv stage in DeviceAttention.forward_device:
- reshape kv → [B*S, head_dim]
- DeviceRMSNorm.forward_device(kv_norm)
- reshape → [B, S, head_dim]
- pick cos/sin via embedding(start_pos, table); reshape to [1, S, rd/2]
- slice nope/rope, rotary on rope, concat
- act_quant_block on nope half via the existing TTL kernel
- concat back

Boundaries: pre-CCL is wkv all_gather; post-CCL is whichever indexer/
compressor matmul comes next (or Lk-Dsparse's gather if no compressor).

All tt-lang kernel definitions are inlined in this file (rmsnorm +
act_quant_block). The reference path still calls into inference.py's
wrappers so the comparison is apples-to-apples.

TODO: mega the rotary helper `_device_apply_rotary_interleaved` is
still a pure-ttnn op chain (slice/transpose/multiply/concat). For the
optimization stage every op in the zone needs to live as @ttl.operation
in this file. Lower the rotary into a tt-lang kernel that takes
(kv_rope_tile, cos_tile, sin_tile) and does the interleave + multiply +
combine inline. ttnn.embedding for cos/sin lookup is also still ttnn —
revisit whether to fold the lookup into the rotary kernel or keep it as
the lone ttnn op (it depends on a device int32 tensor).
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0

from inference import (
    DeviceRMSNorm, _device_apply_rotary_interleaved,
    _get_ttl_act_quant_block_kernel, _RMS_TILE,
)


HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_HEAD_DIM   # 448
ACT_QUANT_BLOCK = 64
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


def _make_act_quant_block_kernel(M_pad: int, N: int, BLOCK: int = 64,
                                 fp8_max: float = 448.0, eps: float = 1e-4):
    """Per-block max-abs scale round-trip in bf16, inlined from inference.py.

        amax = max(|x|, dim=last(block))
        s    = max(amax, eps) / fp8_max
        out  = clamp(x / s, -fp8_max, fp8_max) * s
    """
    if N % BLOCK != 0:
        raise ValueError(f"N={N} not divisible by BLOCK={BLOCK}")
    if BLOCK % TILE != 0:
        raise ValueError(f"BLOCK={BLOCK} not multiple of TILE={TILE}")
    if M_pad % TILE != 0:
        raise ValueError(f"M_pad={M_pad} not multiple of TILE={TILE}")

    M_tiles = M_pad // TILE
    NB = N // BLOCK
    BLOCK_TILES = BLOCK // TILE
    total_work = M_tiles * NB
    inv_fp8_max = 1.0 / fp8_max

    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def act_quant_kernel(x, scaler, out):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        total_cores = grid_rows * grid_cols
        work_per_core = -(-total_work // total_cores)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BLOCK_TILES), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_TILES), block_count=2)

        abs_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BLOCK_TILES), block_count=2)
        amax_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        amax_eps_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        inv_s_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BLOCK_TILES), block_count=2)
        s_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, BLOCK_TILES), block_count=2)

        @ttl.compute()
        def compute():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col
            sc = sc_dfb.wait()

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    x_blk = x_dfb.wait()
                    abs_dfb.reserve().store(ttl.math.abs(x_blk))
                    amax_dfb.reserve().store(
                        ttl.math.reduce_max(abs_dfb.wait(), sc, dims=[1])
                    )
                    amax = amax_dfb.wait()
                    amax_eps_dfb.reserve().store(
                        amax + ttl.math.fill(amax, eps)
                    )
                    amax_eps = amax_eps_dfb.wait()

                    inv_s_dfb.reserve().store(
                        ttl.math.recip(amax_eps)
                        * ttl.math.fill(amax_eps, fp8_max)
                    )
                    inv_s = inv_s_dfb.wait()
                    inv_s_bc_blk = inv_s_bc_dfb.reserve()
                    inv_s_bc_blk.store(
                        ttl.math.broadcast(inv_s, inv_s_bc_blk, dims=[1])
                    )
                    inv_s_bc = inv_s_bc_dfb.wait()

                    s_dfb.reserve().store(
                        amax_eps * ttl.math.fill(amax_eps, inv_fp8_max)
                    )
                    s = s_dfb.wait()
                    s_bc_blk = s_bc_dfb.reserve()
                    s_bc_blk.store(
                        ttl.math.broadcast(s, s_bc_blk, dims=[1])
                    )
                    s_bc = s_bc_dfb.wait()

                    out_dfb.reserve().store(x_blk * inv_s_bc * s_bc)

        @ttl.datamovement()
        def dm_read():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    m = global_w // NB
                    nb = global_w % NB
                    c0 = nb * BLOCK_TILES
                    c1 = c0 + BLOCK_TILES
                    blk = x_dfb.reserve()
                    ttl.copy(x[m:m+1, c0:c1], blk).wait()

        @ttl.datamovement()
        def dm_write():
            core_col, core_row = ttl.node(dims=2)
            core_idx = core_row * grid_cols + core_col

            for local_w in range(work_per_core):
                global_w = core_idx * work_per_core + local_w
                if global_w < total_work:
                    m = global_w // NB
                    nb = global_w % NB
                    c0 = nb * BLOCK_TILES
                    c1 = c0 + BLOCK_TILES
                    ttl.copy(out_dfb.wait(), out[m:m+1, c0:c1]).wait()

    return act_quant_kernel


def make_lk_d1_kernel(mesh, gamma_cpu):
    """Mega kernel for Lk-D1 = kv_norm → rotary(rope) → act_quant_block(nope).

    Two tt-lang dispatches (rmsnorm + act_quant_block), defined locally
    in this file. The rotary helper is a pure-ttnn op chain (not a
    tt-lang kernel) and stays imported.
    """
    rms_kernel = _make_rmsnorm_kernel(
        num_row_tiles=1, h_tiles=HEAD_DIM // TILE,
        rms_eps=NORM_EPS, inv_D=1.0 / HEAD_DIM)
    act_kernel = _make_act_quant_block_kernel(
        M_pad=TILE, N=NOPE_DIM, BLOCK=ACT_QUANT_BLOCK)

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    gamma_packed = gamma_cpu.flatten().to(torch.bfloat16) \
        .unsqueeze(0).expand(TILE, -1).contiguous()
    gamma_tt = ttnn.as_tensor(gamma_packed, dtype=ttnn.bfloat16, **rep)

    state: dict = {}

    def _alloc_replicated_zeros(shape):
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def lk_d1_kernel(kv, cos_full, sin_full, start_pos, kv_out):
        if "scratch" not in state:
            state["scaler_tt"] = ttnn.from_torch(
                torch.ones((TILE, TILE), dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state["normed_tt"] = _alloc_replicated_zeros((TILE, HEAD_DIM))
            state["nope_quant_tt"] = _alloc_replicated_zeros((TILE, NOPE_DIM))
            state["scratch"] = True

        rd_half = ROPE_HEAD_DIM // 2

        # kv_norm: pad → rmsnorm → slice/reshape.
        kv_2d = ttnn.reshape(kv, [B * S, HEAD_DIM])
        kv_padded = ttnn.pad(
            kv_2d, padding=[(0, TILE - B * S), (0, 0)], value=0.0)
        rms_kernel(kv_padded, gamma_tt, state["scaler_tt"], state["normed_tt"])
        kv_normed_2d = ttnn.slice(state["normed_tt"], [0, 0], [B * S, HEAD_DIM])
        kv_normed = ttnn.reshape(kv_normed_2d, [B, S, HEAD_DIM])

        # Rotary on rope-half via the existing ttnn-based helper.
        cos = ttnn.embedding(start_pos, cos_full, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(start_pos, sin_full, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, S, rd_half])
        sin = ttnn.reshape(sin, [1, S, rd_half])
        kv_nope = ttnn.slice(kv_normed, [0, 0, 0], [B, S, NOPE_DIM])
        kv_rope = ttnn.slice(kv_normed, [0, 0, NOPE_DIM], [B, S, HEAD_DIM])
        kv_rope = _device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)

        # act_quant_block on nope-half.
        nope_2d = ttnn.reshape(kv_nope, [B * S, NOPE_DIM])
        nope_padded = ttnn.pad(
            nope_2d, padding=[(0, TILE - B * S), (0, 0)], value=0.0)
        act_kernel(nope_padded, state["scaler_tt"], state["nope_quant_tt"])
        kv_nope_q_2d = ttnn.slice(state["nope_quant_tt"], [0, 0], [B * S, NOPE_DIM])
        kv_nope_q = ttnn.reshape(kv_nope_q_2d, [B, S, NOPE_DIM])

        # Concat nope_q + rope and copy into the test-provided out.
        merged = ttnn.concat([kv_nope_q, kv_rope], dim=-1)
        ttnn.copy(merged, kv_out)

    return lk_d1_kernel


def reference(mesh, kv_tt, gamma_cpu, cos_full_tt, sin_full_tt, start_pos_tt):
    rmsn = DeviceRMSNorm(mesh=mesh, cpu_gamma=gamma_cpu, eps=NORM_EPS)

    # Mirror DeviceAttention.forward_device attn.kv:
    kv_2d = ttnn.reshape(kv_tt, [B * S, HEAD_DIM])
    Mpad = -(-(B * S) // _RMS_TILE) * _RMS_TILE
    if (B * S) < Mpad:
        kv_2d = ttnn.pad(kv_2d, padding=[(0, Mpad - (B * S)), (0, 0)], value=0.0)
    kv_padded = rmsn.forward_device(kv_2d, B * S)
    kv_2d = ttnn.slice(kv_padded, [0, 0], [B * S, HEAD_DIM])
    kv_tt = ttnn.reshape(kv_2d, [B, S, HEAD_DIM])

    rd_half = ROPE_HEAD_DIM // 2
    cos = ttnn.embedding(start_pos_tt, cos_full_tt, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(start_pos_tt, sin_full_tt, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.reshape(cos, [1, S, rd_half])
    sin = ttnn.reshape(sin, [1, S, rd_half])
    kv_nope = ttnn.slice(kv_tt, [0, 0, 0], [B, S, NOPE_DIM])
    kv_rope = ttnn.slice(kv_tt, [0, 0, NOPE_DIM], [B, S, HEAD_DIM])
    kv_rope = _device_apply_rotary_interleaved(ttnn, kv_rope, cos, sin, inverse=False)
    kv_tt = ttnn.concat([kv_nope, kv_rope], dim=-1)

    # act_quant_block on nope half via the existing TTL kernel.
    act_kernel = _get_ttl_act_quant_block_kernel(_RMS_TILE, NOPE_DIM, ACT_QUANT_BLOCK)
    nope_2d = ttnn.reshape(
        ttnn.slice(kv_tt, [0, 0, 0], [B, S, NOPE_DIM]), [B * S, NOPE_DIM])
    act_quant_out = ttnn.zeros(
        shape=(_RMS_TILE, NOPE_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    act_quant_sc = ttnn.as_tensor(
        torch.ones(_RMS_TILE, _RMS_TILE, dtype=torch.bfloat16),
        device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    act_kernel(nope_2d, act_quant_sc, act_quant_out)
    kv_nope_q = ttnn.reshape(
        ttnn.slice(act_quant_out, [0, 0], [B * S, NOPE_DIM]),
        [B, S, NOPE_DIM])
    kv_rope_only = ttnn.slice(kv_tt, [0, 0, NOPE_DIM], [B, S, HEAD_DIM])
    return ttnn.concat([kv_nope_q, kv_rope_only], dim=-1)


def main():
    torch.manual_seed(0)
    mesh = open_mesh()
    mesh_shape = tuple(mesh.shape)
    try:
        kv = torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16) * 0.1
        gamma = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
        cos_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        sin_full = torch.randn(MAX_SEQ_LEN, ROPE_HEAD_DIM // 2, dtype=torch.bfloat16) * 0.5
        start_pos = torch.tensor([[1]], dtype=torch.int32)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        kv_tt = ttnn.as_tensor(kv.contiguous(), dtype=ttnn.bfloat16, **rep)
        cos_full_tt = ttnn.as_tensor(cos_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        sin_full_tt = ttnn.as_tensor(sin_full.contiguous(), dtype=ttnn.bfloat16, **rep)
        start_pos_tt = ttnn.from_torch(
            start_pos, device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        ref_out_tt = reference(mesh, kv_tt, gamma, cos_full_tt, sin_full_tt, start_pos_tt)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_d1_kernel(mesh, gamma)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(kv_tt, cos_full_tt, sin_full_tt, start_pos_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-D1", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
