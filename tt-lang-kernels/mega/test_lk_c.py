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


def make_lk_c_kernel():
    """Mega kernel for Lk-C = q_rsqrt_norm + q rotary + wkv matmul.

    The q-stack tail uses ttnn helpers (q_rsqrt_norm + rotary); the wkv
    matmul is a tt-lang SUMMA dispatch. Mirrors the wq_b → wkv slice of
    DeviceAttention.forward_device exactly.
    """
    matmul_kernel = _make_summa_matmul_kernel(
        M=TILE, K=DIM, N=HEAD_DIM,
        block_cfg=(1, 4, 8), part_cfg=(1, 4, 1))
    state: dict = {}

    def _alloc_replicated_zeros(mesh, shape):
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def lk_c_kernel(q_full, x, cos_full, sin_full, start_pos, wkv_w,
                    q_out, wkv_out):
        if "scratch" not in state:
            mesh = q_full.device()
            state["wkv_padded_tt"] = _alloc_replicated_zeros(mesh, (TILE, HEAD_DIM))
            state["scratch"] = True

        rd_half = ROPE_HEAD_DIM // 2

        # q-stack tail (ttnn helpers).
        q_tt = ttnn.reshape(q_full, [B, S, N_HEADS, HEAD_DIM])
        q_tt = _device_q_rsqrt_norm(ttnn, q_tt, NORM_EPS)
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

        kernel = make_lk_c_kernel()
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
