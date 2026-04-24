"""Standalone test: column-parallel lm_head on 1x4 Blackhole mesh.

Tests the plumbing (shard weight, replicate input, matmul, concat output)
against a CPU torch reference. No HF weights needed — uses synthetic data.

Run on the remote: python3 test_lm_head.py
"""
import time
import torch
import torch.nn.functional as F
import ttnn

VOCAB = 129280
DIM = 4096
SEQ = 1
MESH_SHAPE = (1, 4)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    a_m, b_m = a - a.mean(), b - b.mean()
    num = (a_m * b_m).sum()
    den = (a_m.norm() * b_m.norm()).clamp_min(1e-12)
    return (num / den).item()


def main():
    torch.manual_seed(0)

    # Pad vocab to mesh multiple so the shard dim divides evenly.
    assert VOCAB % MESH_SHAPE[1] == 0, f"vocab {VOCAB} must be divisible by {MESH_SHAPE[1]}"
    # Also round DIM/VOCAB to tile if they aren't already. 4096 and 129280 are both tile-multiples.
    assert DIM % 32 == 0 and VOCAB % 32 == 0

    # CPU reference tensors
    # Match torch.nn.Linear convention: weight is [vocab, dim], y = x @ w.T
    w_cpu = torch.randn(VOCAB, DIM, dtype=torch.bfloat16) * 0.02
    x_cpu = torch.randn(1, SEQ, DIM, dtype=torch.bfloat16)
    y_cpu = F.linear(x_cpu, w_cpu)  # [1, SEQ, VOCAB]
    print(f"[cpu] y shape: {y_cpu.shape}, dtype: {y_cpu.dtype}")

    print(f"[mesh] opening {MESH_SHAPE} ...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE))
    try:
        # Transpose weight to [dim, vocab] so matmul is x @ W (no .T on device).
        w_dv = w_cpu.transpose(0, 1).contiguous()  # [dim, vocab]
        print(f"[mesh] sharding weight {tuple(w_dv.shape)} column-parallel on mesh axis 1 ...")
        t0 = time.time()
        w_tt = ttnn.as_tensor(
            w_dv,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, MESH_SHAPE, dims=(None, -1)),
        )
        print(f"[mesh] weight sharded in {time.time()-t0:.2f}s")

        # Replicate x across the mesh (tiny tensor, each chip gets a copy)
        x_tt = ttnn.as_tensor(
            x_cpu,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print("[mesh] x replicated")

        # Distributed matmul
        t0 = time.time()
        y_tt = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"[mesh] matmul dispatched in {time.time()-t0:.2f}s")

        # Gather back: each chip has [B, SEQ, VOCAB/4]; concat along last dim.
        y_host = ttnn.to_torch(
            y_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, MESH_SHAPE, dims=(0, -1)),
        )
        # ConcatMesh2d with dim0=0 concats the row-axis as batch (1-wide so no-op in our case).
        # We only care about the column-parallel concat on axis=-1.
        # Result should be [1, SEQ, VOCAB].
        print(f"[mesh] y_host shape: {tuple(y_host.shape)}")

        # Trim any padding on batch axis (1×4 mesh with row-dim=0 replicated means no row concat expected,
        # but ConcatMesh2d always concats). Take the first batch.
        if y_host.shape[0] != 1:
            y_host = y_host[:1]

        p = pcc(y_cpu, y_host)
        max_abs = (y_cpu.float() - y_host.float()).abs().max().item()
        print(f"[check] PCC = {p:.6f}, max abs diff = {max_abs:.4f}")
        ok = p > 0.999
        print(f"[check] {'PASS' if ok else 'FAIL'}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
