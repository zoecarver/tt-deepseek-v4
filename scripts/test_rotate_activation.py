"""PCC test: device rotate_activation as a single matmul against the
Sylvester-Hadamard matrix.

Run via tt-connect-remote-device:
  scripts/run-test.sh --hw scripts/test_rotate_activation.py
"""
import sys
import os

# Allow `import inference` regardless of where the runner copies us to.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/tmp")

import torch
import importlib.util

inf_path = "/tmp/inference.py" if os.path.exists("/tmp/inference.py") else os.path.join(
    os.path.dirname(HERE), "inference.py"
)
spec = importlib.util.spec_from_file_location("inf", inf_path)
inf = importlib.util.module_from_spec(spec)
sys.modules["inf"] = inf
spec.loader.exec_module(inf)

import ttnn

torch.manual_seed(0)

B, S, H_heads, D = 1, 1, 16, 128

x_q = torch.randn(B, S, H_heads, D, dtype=torch.bfloat16)
x_kv = torch.randn(B, S, D, dtype=torch.bfloat16)

ref_q = inf.rotate_activation(x_q.clone())
ref_kv = inf.rotate_activation(x_kv.clone())

H_mat = (inf._sylvester_hadamard(D) * (D ** -0.5)).to(torch.bfloat16)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
H_tt = ttnn.as_tensor(
    H_mat, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
)


def run_device(x):
    x_tt = ttnn.as_tensor(
        x.contiguous(), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out_tt = inf._device_rotate_activation(ttnn, x_tt, H_tt)
    composer = ttnn.ConcatMesh2dToTensor(mesh, (1, 4), dims=(1, 0))
    return ttnn.to_torch(out_tt, mesh_composer=composer)[:B]


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


dev_q = run_device(x_q)
dev_kv = run_device(x_kv)
print(f"PCC q  device vs cpu_butterfly: {pcc(dev_q, ref_q):.6f}")
print(f"PCC kv device vs cpu_butterfly: {pcc(dev_kv, ref_kv):.6f}")
assert pcc(dev_q, ref_q) > 0.999, "q PCC regression"
assert pcc(dev_kv, ref_kv) > 0.999, "kv PCC regression"
ttnn.close_mesh_device(mesh)
print("OK")
