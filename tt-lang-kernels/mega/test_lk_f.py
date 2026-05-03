"""Lk-F PCC test: gate post-process + routed experts (sans routed all_reduce).

Reference covers everything from the shared-expert all_reduce up to the
routed-expert all_reduce. Mirrors `MoE.forward_device` and
`MoE._forward_device_routed_cached`, sans the closing all_reduce.

For the test we run on a single-card mesh with replicated routed-expert
weights to keep the harness simple — the per-chip selection-mask logic
still exercises with `_chip_local_ids_tt` covering the full expert set.

Boundaries: pre-CCL is shared all_reduce (we receive `x_tt` replicated);
post-CCL is the routed-expert all_reduce on `y_local`.
"""
from __future__ import annotations

import sys
import torch
import ttnn
import ttl

import _refs  # noqa: F401
from _refs import open_mesh, close_mesh, report_pcc, download_chip0


DIM = 4096
INTER_DIM = 2048
N_ROUTED = 256
TOPK = 8
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
B = 1


def make_lk_f_kernel():
    """Placeholder mega kernel for Lk-F.

    Inputs:
      x:                 [1, dim] bf16  — post-shared-all_reduce
      gate_w:            [n_routed, dim] bf16  — replicated
      gate_bias:         [n_routed] bf16        — replicated (non-hash branch)
      w1, w2, w3:        per-chip bfp4_b weight slices (here we use bf16 for the test)
                         w1: [1, per_chip, dim, inter] bf16
                         w2: [1, per_chip, inter, dim] bf16
                         w3: [1, per_chip, dim, inter] bf16
      chip_ids_4d:       [1, per_chip, 1, 1] int32  — chip-local expert ids
    Output:
      y_local:           [1, 1, 1, dim] bf16  — pre-routed-all_reduce sum
    """
    @ttl.operation(grid="auto")
    def lk_f_kernel(x, gate_w, gate_bias, w1, w2, w3, chip_ids_4d, y_local_out):
        @ttl.compute()
        def compute():
            pass

    return lk_f_kernel


def reference(mesh, x_tt, gate_w_tt, gate_bias_tt,
              w1_tt, w2_tt, w3_tt, chip_ids_4d_tt, per_chip):
    # Gate body — sqrt(softplus(x @ Wᵀ)).
    raw = ttnn.matmul(x_tt, gate_w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    scores = ttnn.sqrt(ttnn.softplus(raw))
    biased = ttnn.add(scores, gate_bias_tt)
    _, indices_tt = ttnn.topk(biased, k=TOPK, dim=-1, largest=True, sorted=True)
    gathered = ttnn.gather(scores, dim=-1, index=indices_tt)
    wsum = ttnn.sum(gathered, dim=-1, keepdim=True)
    normed = ttnn.div(gathered, wsum)
    weights_tt = ttnn.multiply(normed, ROUTE_SCALE)

    # Path D body (mirror of MoE._forward_device_routed_cached, sans all_reduce).
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
    y_local = ttnn.sum(y, dim=1, keepdim=True)        # [1, 1, 1, dim]
    return y_local


def main():
    torch.manual_seed(0)
    # 1×1 mesh so the test models the per-chip path with the full expert set.
    mesh = open_mesh(shape=(1, 1))
    mesh_shape = tuple(mesh.shape)
    try:
        per_chip = N_ROUTED  # all experts on the one chip for the test
        x = torch.randn(1, DIM, dtype=torch.bfloat16) * 0.1
        gate_w = torch.randn(DIM, N_ROUTED, dtype=torch.bfloat16) * 0.02
        gate_bias = torch.randn(1, N_ROUTED, dtype=torch.bfloat16) * 0.01
        w1 = torch.randn(1, per_chip, DIM, INTER_DIM, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(1, per_chip, INTER_DIM, DIM, dtype=torch.bfloat16) * 0.02
        w3 = torch.randn(1, per_chip, DIM, INTER_DIM, dtype=torch.bfloat16) * 0.02
        chip_ids = torch.arange(per_chip, dtype=torch.int32).view(1, per_chip, 1, 1)

        rep = dict(device=mesh, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        x_tt = ttnn.as_tensor(x.contiguous(), dtype=ttnn.bfloat16, **rep)
        gate_w_tt = ttnn.as_tensor(gate_w.contiguous(), dtype=ttnn.bfloat16, **rep)
        gate_bias_tt = ttnn.as_tensor(gate_bias.contiguous(), dtype=ttnn.bfloat16, **rep)
        w1_tt = ttnn.as_tensor(w1.contiguous(), dtype=ttnn.bfloat16, **rep)
        w2_tt = ttnn.as_tensor(w2.contiguous(), dtype=ttnn.bfloat16, **rep)
        w3_tt = ttnn.as_tensor(w3.contiguous(), dtype=ttnn.bfloat16, **rep)
        chip_ids_tt = ttnn.as_tensor(chip_ids.contiguous(), dtype=ttnn.int32, **rep)

        ref_out_tt = reference(mesh, x_tt, gate_w_tt, gate_bias_tt,
                               w1_tt, w2_tt, w3_tt, chip_ids_tt, per_chip)
        ref_host = download_chip0(mesh, mesh_shape, ref_out_tt)

        kernel = make_lk_f_kernel()
        y_local_out_tt = ttnn.from_torch(
            torch.zeros(1, 1, 1, DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, **rep)
        kernel(x_tt, gate_w_tt, gate_bias_tt, w1_tt, w2_tt, w3_tt,
               chip_ids_tt, y_local_out_tt)
        kernel_host = download_chip0(mesh, mesh_shape, y_local_out_tt)

        ok = report_pcc("Lk-F", ref_host, kernel_host)
        sys.exit(0 if ok else 1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
