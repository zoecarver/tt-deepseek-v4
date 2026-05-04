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

Fused as a single ttl.operation. Grid is (Np=8, 1) with each core
owning two tile-columns of HEAD_DIM. The first NOPE_CORES=7 cores own
the nope half (cols 0..447) and run act_quant_block; the last core
owns the rope half (cols 448..511) and runs the swap-matmul rotary
combine. ssq is reduced across all cores via a PipeNet, and inv_rms
is broadcast back via a second PipeNet so every core normalizes its
own column shard before the leaf operation.

Rotary lowering trick (same as Lk-C): bake the rotate_half permutation
into a swap matrix P and the rotate_half sign into the sin table
(sin_signed). Then rotary becomes
  kv_rope_rot = kv_rope * cos_extended + (kv_rope @ P) * sin_signed
where cos_extended = pair-repeat(cos) and P = block_diag of
[[0,1],[1,0]] x rd/2. cos/sin tables are pre-replicated across TILE
rows on host so the kernel reads tile-aligned tensors.

ttnn glue (TODO: mega): ttnn.embedding(start_pos, ...) still depends on
a device uint32 index (no tt-lang gather primitive). Slice/pad/reshape
around the lookup stay in ttnn for the same reason.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0, benchmark

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


def _make_fused_lkd1_kernel(M: int, K: int, NOPE: int, ROPE: int,
                            block_cfg, part_cfg,
                            rms_eps: float,
                            fp8_max: float = 448.0, eps: float = 1e-4,
                            fp32_dest_acc_en: bool = True):
    """Fused rmsnorm + rotary(rope_half) + act_quant_block(nope_half) as
    a single ttl.operation.

    Grid: (Np, 1). Each core owns `bn` tile-columns of HEAD_DIM=NOPE+ROPE.
    Cores [0, NOPE_CORES) own the nope half and run act_quant_block on
    their normalized shard. The remaining cores own the rope half and
    run the swap-matmul rotary combine; ROPE is small (typically 1 tile-
    block wide) so a single rope core handles the whole rope half.

    All-reduce on ssq:
    - reduce: each non-root sends its ssq partial to (0, 0).
    - bcast:  root computes inv = rsqrt(ssq/D + eps) and mcasts it back
      to all cores via a second PipeNet.

    M is fixed at one bm-block (Mp = 1, Kp = 1).
    """
    bm, bn, bk = block_cfg
    Mp, Np, _Kp = part_cfg
    if Mp != 1 or _Kp != 1:
        raise ValueError(f"fused Lk-D1 assumes Mp=Kp=1, got Mp={Mp} Kp={_Kp}")
    HEAD = NOPE + ROPE
    if HEAD != K:
        raise ValueError(f"K={K} must equal NOPE+ROPE={HEAD}")
    Mt = M // TILE
    HEAD_t = HEAD // TILE
    if Mt != bm or HEAD_t % bn:
        raise ValueError(
            f"shape/block: Mt={Mt} bm={bm} HEAD_t={HEAD_t} bn={bn}")
    if HEAD_t // bn != Np:
        raise ValueError(f"Np={Np} must equal HEAD_t/bn = {HEAD_t // bn}")
    NOPE_t = NOPE // TILE
    NOPE_CORES = NOPE_t // bn
    if NOPE_CORES * bn != NOPE_t:
        raise ValueError(
            f"NOPE doesn't align with bn: NOPE_t={NOPE_t} bn={bn}")
    ROPE_t = ROPE // TILE
    if ROPE_t != bn:
        raise ValueError(
            f"rope core handles a single bn-block: ROPE_t={ROPE_t} bn={bn}")
    if NOPE_CORES + 1 != Np:
        raise ValueError(
            f"expected Np = NOPE_CORES + 1 (one rope core), "
            f"got Np={Np} NOPE_CORES={NOPE_CORES}")

    inv_D = 1.0 / float(HEAD)
    inv_fp8_max = 1.0 / fp8_max
    # Block_count for the receive CB at root: each non-root core sends one
    # ssq partial. min 2 to keep dataflow happy.
    SSQ_RECV_BC = max(2, Np - 1)

    @ttl.operation(grid=(Np, 1),
                   fp32_dest_acc_en=fp32_dest_acc_en,
                   options="--no-ttl-reduce-full-fp32")
    def fused_kernel(kv, gamma, P, cos_b, sin_b, scaler, out):
        # Reduce ssq partials (non-root → root) and broadcast inv_rms back.
        ssq_reduce_pipes = [ttl.Pipe(src=(n_p, 0), dst=(0, 0))
                            for n_p in range(1, Np)]
        ssq_reduce_net = ttl.PipeNet(ssq_reduce_pipes)
        inv_bcast_pipes = [ttl.Pipe(src=(0, 0), dst=(slice(0, Np), 0))]
        inv_bcast_net = ttl.PipeNet(inv_bcast_pipes)

        # Common CBs.
        x_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, bn), block_count=2)
        g_cb = ttl.make_dataflow_buffer_like(gamma, shape=(bm, bn), block_count=2)
        sc_cb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        xsq_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, bn), block_count=2)
        ssq_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, 1), block_count=2)
        ssq_recv_cb = ttl.make_dataflow_buffer_like(
            kv, shape=(bm, 1), block_count=SSQ_RECV_BC)
        inv_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, 1), block_count=2)
        inv_recv_cb = ttl.make_dataflow_buffer_like(
            kv, shape=(bm, 1), block_count=2)
        inv_bc_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, bn), block_count=2)
        normed_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, bn), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(bm, bn), block_count=2)

        # Nope-only (act_quant_block) CBs.
        abs_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, bn), block_count=2)
        amax_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, 1), block_count=2)
        aeps_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, 1), block_count=2)
        inv_s_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, 1), block_count=2)
        s_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, 1), block_count=2)
        inv_s_bc_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, bn), block_count=2)
        s_bc_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, bn), block_count=2)

        # Rope-only (swap-matmul + rotary combine) CBs.
        P_cb = ttl.make_dataflow_buffer_like(P, shape=(bn, bn), block_count=2)
        cos_cb = ttl.make_dataflow_buffer_like(cos_b, shape=(bm, bn), block_count=2)
        sin_cb = ttl.make_dataflow_buffer_like(sin_b, shape=(bm, bn), block_count=2)
        swap_cb = ttl.make_dataflow_buffer_like(kv, shape=(bm, bn), block_count=2)

        @ttl.compute()
        def compute():
            n_p, _ = ttl.node(dims=2)
            sc = sc_cb.wait()

            # Local ssq partial over this core's bn-tile column shard.
            x = x_cb.wait()
            xsq_cb.reserve().store(x * x)
            ssq_cb.reserve().store(
                ttl.math.reduce_sum(xsq_cb.wait(), sc, dims=[1])
            )

            if n_p == 0:
                # Root: receive Np-1 ssq partials, sum, finalize inv_rms.
                for _ in range(Np - 1):
                    prev = ssq_cb.wait()
                    r = ssq_recv_cb.wait()
                    ssq_cb.reserve().store(prev + r)
                sq = ssq_cb.wait()
                inv_t = inv_cb.reserve()
                inv_t.store(ttl.math.broadcast(
                    ttl.math.rsqrt(
                        sq * ttl.math.fill(sq, inv_D)
                        + ttl.math.fill(sq, rms_eps)
                    ),
                    inv_t, dims=[1],
                ))

            # All cores: receive inv from broadcast pipe, expand to (bm, bn).
            inv = inv_recv_cb.wait()
            inv_bc_t = inv_bc_cb.reserve()
            inv_bc_t.store(ttl.math.broadcast(inv, inv_bc_t, dims=[1]))
            inv_bc = inv_bc_cb.wait()

            # Compute normed = x * gamma * inv_bc into a scratch CB so it can
            # be reused twice (matmul + elementwise on the rope core, abs +
            # final mul on nope cores) without re-reading from DRAM.
            x2 = x_cb.wait()
            g = g_cb.wait()
            normed_cb.reserve().store(x2 * g * inv_bc)
            normed = normed_cb.wait()

            if n_p == NOPE_CORES:
                # Rope core: out = normed * cos + (normed @ P) * sin.
                P_blk = P_cb.wait()
                c = cos_cb.wait()
                si = sin_cb.wait()
                swap_cb.reserve().store(normed @ P_blk)
                swap_blk = swap_cb.wait()
                out_cb.reserve().store(normed * c + swap_blk * si)
            else:
                # Nope core: act_quant_block round-trip in bf16.
                #   amax = reduce_max(|normed|, dim=1)
                #   inv_s = fp8_max / (amax + eps)
                #   s     = (amax + eps) / fp8_max
                #   out   = normed * inv_s_bc * s_bc
                abs_cb.reserve().store(ttl.math.abs(normed))
                amax_cb.reserve().store(
                    ttl.math.reduce_max(abs_cb.wait(), sc, dims=[1])
                )
                amax = amax_cb.wait()
                aeps_cb.reserve().store(
                    amax + ttl.math.fill(amax, eps)
                )
                aeps = aeps_cb.wait()
                inv_s_cb.reserve().store(
                    ttl.math.recip(aeps) * ttl.math.fill(aeps, fp8_max)
                )
                inv_s = inv_s_cb.wait()
                inv_s_bc_t = inv_s_bc_cb.reserve()
                inv_s_bc_t.store(
                    ttl.math.broadcast(inv_s, inv_s_bc_t, dims=[1])
                )
                inv_s_bc = inv_s_bc_cb.wait()
                s_cb.reserve().store(
                    aeps * ttl.math.fill(aeps, inv_fp8_max)
                )
                s = s_cb.wait()
                s_bc_t = s_bc_cb.reserve()
                s_bc_t.store(
                    ttl.math.broadcast(s, s_bc_t, dims=[1])
                )
                s_bc = s_bc_cb.wait()
                out_cb.reserve().store(normed * inv_s_bc * s_bc)

        @ttl.datamovement()
        def dm_read():
            n_p, _ = ttl.node(dims=2)
            nc = n_p * bn

            ttl.copy(scaler[0, 0], sc_cb.reserve()).wait()

            # x for ssq.
            ttl.copy(kv[0:bm, nc:nc + bn], x_cb.reserve()).wait()

            # ssq all-reduce: non-root sends, root receives Np-1 partials.
            if n_p == 0:
                def recv_ssq(pipe):
                    r = ssq_recv_cb.reserve()
                    ttl.copy(pipe, r).wait()
                ssq_reduce_net.if_dst(recv_ssq)
            else:
                ssq = ssq_cb.wait()

                def send_ssq(pipe):
                    ttl.copy(ssq, pipe).wait()
                ssq_reduce_net.if_src(send_ssq)

            # inv_rms broadcast: root sends, all cores (incl. self) receive.
            if n_p == 0:
                inv = inv_cb.wait()

                def send_inv(pipe):
                    ttl.copy(inv, pipe).wait()
                inv_bcast_net.if_src(send_inv)

            def recv_inv(pipe):
                r = inv_recv_cb.reserve()
                ttl.copy(pipe, r).wait()
            inv_bcast_net.if_dst(recv_inv)

            # x and gamma for the normalize step.
            ttl.copy(kv[0:bm, nc:nc + bn], x_cb.reserve()).wait()
            ttl.copy(gamma[0:bm, nc:nc + bn], g_cb.reserve()).wait()

            # Rope core only: P / cos / sin for the rotary combine.
            if n_p == NOPE_CORES:
                ttl.copy(P[0:bn, 0:bn], P_cb.reserve()).wait()
                ttl.copy(cos_b[0:bm, 0:bn], cos_cb.reserve()).wait()
                ttl.copy(sin_b[0:bm, 0:bn], sin_cb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            n_p, _ = ttl.node(dims=2)
            nc = n_p * bn
            o = out_cb.wait()
            ttl.copy(o, out[0:bm, nc:nc + bn]).wait()

    return fused_kernel


def _build_rotary_tables(cos_full_cpu: torch.Tensor, sin_full_cpu: torch.Tensor,
                         inverse: bool):
    """Same shape as Lk-C: (max_seq_len, TILE * rd) bf16 packed cos/sin.

    cos_extended = pair-repeat(cos), then expanded across TILE rows.
    sin_signed   = sin * sign, where sign = [-1, +1, -1, +1, ...] for
                   forward rotary (so that x[2k+1] picks up -sin(k)).
    """
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
    """Block-diagonal swap matrix P [rd, rd] with 2x2 [[0,1],[1,0]] blocks."""
    if rd % 2:
        raise ValueError(f"rd={rd} must be even")
    P = torch.zeros(rd, rd, dtype=torch.bfloat16)
    for k in range(rd // 2):
        P[2 * k, 2 * k + 1] = 1.0
        P[2 * k + 1, 2 * k] = 1.0
    return P


def make_lk_d1_kernel(mesh, gamma_cpu, cos_full_cpu, sin_full_cpu):
    """Mega kernel for Lk-D1 = kv_norm → rotary(rope) → act_quant_block(nope).

    Single fused ttl.operation. The cos/sin row gather still happens via
    ttnn.embedding (no tt-lang scalar-indexed gather yet). The final
    slice/reshape into the [B, S, HEAD_DIM] caller buffer is also ttnn,
    same as every other mega test.
    """
    # M=TILE=32, K=HEAD_DIM=512. Mt=1, HEAD_t=16.
    # block=(1, 2, _) part=(1, 8, 1) → 8 cores, 7 nope + 1 rope.
    fused_kernel = _make_fused_lkd1_kernel(
        M=TILE, K=HEAD_DIM, NOPE=NOPE_DIM, ROPE=ROPE_HEAD_DIM,
        block_cfg=(1, 2, 2), part_cfg=(1, 8, 1),
        rms_eps=NORM_EPS,
    )

    rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
               memory_config=ttnn.DRAM_MEMORY_CONFIG,
               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    gamma_packed = gamma_cpu.flatten().to(torch.bfloat16) \
        .unsqueeze(0).expand(TILE, -1).contiguous()
    gamma_tt = ttnn.as_tensor(gamma_packed, dtype=ttnn.bfloat16, **rep)

    cos_ext_packed, sin_signed_packed = _build_rotary_tables(
        cos_full_cpu, sin_full_cpu, inverse=False)
    cos_ext_tt = ttnn.as_tensor(cos_ext_packed, dtype=ttnn.bfloat16, **rep)
    sin_signed_tt = ttnn.as_tensor(sin_signed_packed, dtype=ttnn.bfloat16, **rep)
    P_cpu = _build_swap_matrix(ROPE_HEAD_DIM)
    P_tt = ttnn.as_tensor(P_cpu.contiguous(), dtype=ttnn.bfloat16, **rep)

    state: dict = {}

    def lk_d1_kernel(kv, cos_full, sin_full, start_pos, kv_out):
        if "scratch" not in state:
            state["scaler_tt"] = ttnn.from_torch(
                torch.ones((TILE, TILE), dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state["out_padded_tt"] = ttnn.from_torch(
                torch.zeros((TILE, HEAD_DIM), dtype=torch.bfloat16),
                device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state["scratch"] = True

        # 1. Pad input [1, 1, K] -> [TILE, K].
        kv_2d = ttnn.reshape(kv, [B * S, HEAD_DIM])
        kv_padded = ttnn.pad(
            kv_2d, padding=[(0, TILE - B * S), (0, 0)], value=0.0)

        # 2. Gather cos/sin row for start_pos. ttnn.embedding stays here
        # because tt-lang has no scalar-indexed row gather yet (TODO: mega).
        cos_b_2d = ttnn.embedding(start_pos, cos_ext_tt, layout=ttnn.TILE_LAYOUT)
        sin_b_2d = ttnn.embedding(start_pos, sin_signed_tt, layout=ttnn.TILE_LAYOUT)
        cos_b = ttnn.reshape(cos_b_2d, [TILE, ROPE_HEAD_DIM])
        sin_b = ttnn.reshape(sin_b_2d, [TILE, ROPE_HEAD_DIM])

        # 3. Single fused ttl.operation: rmsnorm + rotary(rope) + act_quant(nope).
        fused_kernel(kv_padded, gamma_tt, P_tt, cos_b, sin_b,
                     state["scaler_tt"], state["out_padded_tt"])

        # 4. Slice + reshape + copy into the test-provided [1, 1, HEAD_DIM] out.
        out_2d = ttnn.slice(state["out_padded_tt"], [0, 0], [B * S, HEAD_DIM])
        out_3d = ttnn.reshape(out_2d, [B, S, HEAD_DIM])
        ttnn.copy(out_3d, kv_out)

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
    # Fused kernel program is large; grow the kernel-config buffer at the
    # cost of per-core L1. See mega/README.md "Large fused programs".
    mesh = open_mesh(kernel_config_extra_bytes=128 * 1024)
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

        kernel = make_lk_d1_kernel(mesh, gamma, cos_full, sin_full)
        out_tt = ttnn.from_torch(
            torch.zeros(1, 1, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(kv_tt, cos_full_tt, sin_full_tt, start_pos_tt, out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, out_tt)

        ok = report_pcc("Lk-D1", ref_host, kernel_host)

        benchmark("Lk-D1 ref",
                  lambda: reference(mesh, kv_tt, gamma, cos_full_tt,
                                    sin_full_tt, start_pos_tt),
                  mesh)
        benchmark("Lk-D1 ttl",
                  lambda: kernel(kv_tt, cos_full_tt, sin_full_tt,
                                 start_pos_tt, out_tt),
                  mesh)

        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
