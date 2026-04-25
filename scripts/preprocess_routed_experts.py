"""Preprocess routed-expert weights into a sharded ttnn.bfloat4_b cache.

Reads an HF-derived state_dict.pt that has fp4-packed routed-expert weights
(layers.{L}.ffn.experts.{E}.w[1|3|2].weight as float4_e2m1fn_x2,
plus the matching .scale as float8_e8m0fnu) and writes a sibling cache
directory of pre-sharded .tensorbin files. At inference, the runtime calls
ttnn.load_tensor on each file with the live mesh device and the shards
distribute automatically — zero per-element host work, zero PCIe staging.

Output layout (mirrors tt-metal/models/demos/deepseek_v3_b1/prepare_weights.py):

    <out_dir>/
      manifest.json           # global metadata: mesh shape, n_layers, dims, model
      layer_003/              # one dir per MoE layer (skips hash layers 0..n_hash_layers-1)
        w1.tensorbin          # bfp4_b sharded [rows, cols, per_chip, K, N]
        w1_scale.tensorbin    # bf16   sharded [rows, cols, per_chip, Kb, N], Kb = K/32
        w3.tensorbin
        w3_scale.tensorbin
        w2.tensorbin
        w2_scale.tensorbin
      ...

Note on bit preservation: the bfp4_b file holds the original fp4 nibbles
bit-for-bit (achieved by pre-staging into a bf16 lattice tensor whose values
land exactly on bfp4's representable face). At kernel time we run an
algebraic remap from the bfp4 lattice {0, ±0.25, ..., ±1.75} back to the
fp4 e2m1 magnitudes {0, ±0.5, ..., ±6}, then multiply by the per-K-block
scale. See tt-lang-kernels/fp4_gemm.py for the kernel.

Usage on the remote (Galaxy):

    cd /home/ubuntu/deepseek
    python3 -u scripts/preprocess_routed_experts.py \\
        --state-dict /home/ubuntu/hf/state_dict.pt \\
        --out /home/ubuntu/hf/state_dict_bfp4_routed \\
        --mesh-rows 4 --mesh-cols 8 \\
        --validate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch

# We import fp4_gemm helpers for the lattice gather and scale expansion.
# When run from the repo: scripts/preprocess_routed_experts.py finds it via
# sibling tt-lang-kernels/. When run from /tmp on the remote: copy fp4_gemm.py
# next to this script (or to /tmp/) and the bare /tmp will be on sys.path.
THIS_DIR = Path(__file__).resolve().parent
for candidate in (THIS_DIR.parent / "tt-lang-kernels", THIS_DIR, Path("/tmp")):
    if (candidate / "fp4_gemm.py").exists():
        sys.path.insert(0, str(candidate))
        break
from fp4_gemm import (  # noqa: E402
    BFP4_MAGS,
    FP4_BLOCK_K,
    TILE,
    fp4_bytes_to_fp4_dequant_bf16,
    fp4_gemm_via_bfp4,
)

import ttnn  # noqa: E402  (after path so optional but expected available)


# -----------------------------------------------------------------------------
# State-dict introspection
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertKey:
    layer: int
    expert: int
    wname: str  # "w1" | "w2" | "w3"


def _scan_routed_experts(state: dict) -> dict[int, list[ExpertKey]]:
    """Return {layer_idx: [ExpertKey, ...]} for all routed-expert weights in the
    state dict. Order is stable: by (layer, expert, wname)."""
    found: dict[int, dict[tuple[int, str], ExpertKey]] = {}
    pat_prefix = "layers."
    pat_mid = ".ffn.experts."
    for k in state.keys():
        if not k.startswith(pat_prefix) or pat_mid not in k:
            continue
        if not k.endswith(".weight"):
            continue
        rest = k[len(pat_prefix):]                              # "{L}.ffn.experts.{E}.{w}.weight"
        L_str, _, after = rest.partition(".")
        E_str = after.split(".experts.")[1].split(".")[0]
        wname = after.split(".")[-2]
        if wname not in ("w1", "w2", "w3"):
            continue
        L, E = int(L_str), int(E_str)
        found.setdefault(L, {})[(E, wname)] = ExpertKey(L, E, wname)
    out: dict[int, list[ExpertKey]] = {}
    for L, m in found.items():
        out[L] = [m[k] for k in sorted(m.keys())]
    return out


def _validate_layer_keys(state: dict, layer: int, n_experts: int) -> None:
    """Sanity check that every (E, wname) is present with weight + scale."""
    missing = []
    for E in range(n_experts):
        for wname in ("w1", "w2", "w3"):
            for suf in (".weight", ".scale"):
                k = f"layers.{layer}.ffn.experts.{E}.{wname}{suf}"
                if k not in state:
                    missing.append(k)
    if missing:
        raise KeyError(
            f"layer {layer}: {len(missing)} routed-expert keys missing, "
            f"first 5: {missing[:5]}"
        )


# -----------------------------------------------------------------------------
# Bulk lattice gather (the core "30k -> 1 op" reduction)
# -----------------------------------------------------------------------------


def _signed_bfp4_lattice() -> torch.Tensor:
    """[16] float32 lookup: indices 0..7 = +mag, 8..15 = -mag of bfp4 lattice."""
    return torch.tensor(list(BFP4_MAGS) + [-v for v in BFP4_MAGS], dtype=torch.float32)


def _bulk_fp4_to_bfp4_lattice_bf16(packed_bytes: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] uint8 fp4 -> [E, N, K] bf16 in bfp4 lattice. One op,
    fully vectorized; no per-expert Python loop."""
    if packed_bytes.dim() != 3:
        raise ValueError(
            f"expected [E, N, K/2], got shape {tuple(packed_bytes.shape)}")
    E, N, Kh = packed_bytes.shape
    K = Kh * 2
    low = (packed_bytes & 0x0F).to(torch.long)
    high = ((packed_bytes >> 4) & 0x0F).to(torch.long)
    lat = _signed_bfp4_lattice()
    out = torch.empty(E, N, K, dtype=torch.bfloat16)
    out[..., 0::2] = lat[low].to(torch.bfloat16)
    out[..., 1::2] = lat[high].to(torch.bfloat16)
    return out


def _bulk_e8m0_to_bf16_compact(scales_e8m0: torch.Tensor, *, K: int, N: int) -> torch.Tensor:
    """[E, N, K/block_k] e8m0 -> [E, Kb, N] bf16, compact (no K-axis expansion).

    The K→Kb expansion is deferred to the device kernel (fp4_gemm_via_bfp4
    does ttnn.repeat_interleave at multiply time), which keeps disk and
    on-device storage 32× smaller.
    """
    if scales_e8m0.dim() != 3:
        raise ValueError(
            f"expected [E, N, K/block_k], got shape {tuple(scales_e8m0.shape)}")
    if scales_e8m0.dtype == torch.float8_e8m0fnu:
        sf_u = scales_e8m0.view(torch.uint8)
    elif scales_e8m0.dtype == torch.uint8:
        sf_u = scales_e8m0
    else:
        raise TypeError(f"unexpected scale dtype {scales_e8m0.dtype}")
    Kb = K // FP4_BLOCK_K
    if sf_u.shape[-1] != Kb:
        raise ValueError(
            f"scale last-dim {sf_u.shape[-1]} != K/block_k = {Kb}")
    if sf_u.shape[-2] != N:
        raise ValueError(
            f"scale dim -2 {sf_u.shape[-2]} != N = {N}")
    sf_f32 = (sf_u.to(torch.int32) << 23).contiguous().view(torch.float32)  # [E, N, Kb]
    sf_kn = sf_f32.transpose(-2, -1).contiguous()                            # [E, Kb, N]
    return sf_kn.to(torch.bfloat16)


# -----------------------------------------------------------------------------
# Per-(layer, weight) pipeline
# -----------------------------------------------------------------------------


@dataclass
class WeightShape:
    """For matmul order: x[..., K] @ w[K, N] -> y[..., N]."""
    K: int
    N: int


def _shape_for(wname: str, *, dim: int, inter_dim: int) -> WeightShape:
    """w1, w3: [out=inter_dim, in=dim]   so K=dim, N=inter_dim
       w2:     [out=dim,       in=inter] so K=inter, N=dim"""
    if wname in ("w1", "w3"):
        return WeightShape(K=dim, N=inter_dim)
    if wname == "w2":
        return WeightShape(K=inter_dim, N=dim)
    raise ValueError(f"unknown weight name {wname}")


def _stack_layer_weights(state: dict, layer: int, wname: str, n_experts: int):
    """Return (packed [E, N, Kh] uint8, scale [E, N, K/blk] e8m0)."""
    weights = []
    scales = []
    for E in range(n_experts):
        kw = f"layers.{layer}.ffn.experts.{E}.{wname}.weight"
        ks = f"layers.{layer}.ffn.experts.{E}.{wname}.scale"
        w = state[kw]
        s = state[ks]
        if w.dtype == torch.float4_e2m1fn_x2:
            w = w.view(torch.uint8)
        elif w.dtype not in (torch.int8, torch.uint8):
            raise TypeError(f"layer {layer} expert {E} {wname}.weight dtype {w.dtype}")
        weights.append(w)
        scales.append(s)
    return torch.stack(weights, dim=0), torch.stack(scales, dim=0)


def _shard_dump(
    bf16_kn_per_expert: torch.Tensor,    # [E, K, N] bf16
    *,
    mesh,
    rows: int,
    cols: int,
    out_path: Path,
    target_dtype,
):
    """Reshape an [E, K, N] tensor to (rows, cols, per_chip, K, N), build a
    host ttnn tensor with the right mesh sharding (dims 0, 1) and target dtype,
    then dump_tensor to out_path. Frees intermediates as we go."""
    E, K, N = bf16_kn_per_expert.shape
    if E != rows * cols * (E // (rows * cols)):
        raise ValueError(f"E={E} not divisible by rows*cols={rows*cols}")
    per_chip = E // (rows * cols)

    reshaped = bf16_kn_per_expert.view(rows, cols, per_chip, K, N).contiguous()
    del bf16_kn_per_expert

    host_tt = ttnn.from_torch(
        reshaped,
        dtype=target_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (rows, cols), dims=(0, 1)),
    )
    del reshaped

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ttnn.dump_tensor(str(out_path), host_tt)
    del host_tt


def _process_one_weight(
    state: dict,
    layer: int,
    wname: str,
    *,
    mesh,
    rows: int,
    cols: int,
    n_experts: int,
    dim: int,
    inter_dim: int,
    layer_dir: Path,
    profile: bool = False,
) -> dict:
    """Pipeline for one (layer, wname): stack -> lattice -> shard -> dump
    for both weight and scale. Pure-host until dump_tensor; thread-safe to
    run multiple wnames in parallel within a layer."""
    ws = _shape_for(wname, dim=dim, inter_dim=inter_dim)
    K, N = ws.K, ws.N
    t0 = time.perf_counter()

    packed, scales = _stack_layer_weights(state, layer, wname, n_experts)
    if packed.shape != (n_experts, N, K // 2):
        raise ValueError(
            f"layer {layer} {wname}.weight stack shape "
            f"{tuple(packed.shape)} != expected {(n_experts, N, K // 2)}"
        )
    if scales.shape != (n_experts, N, K // FP4_BLOCK_K):
        raise ValueError(
            f"layer {layer} {wname}.scale stack shape "
            f"{tuple(scales.shape)} != expected "
            f"{(n_experts, N, K // FP4_BLOCK_K)}"
        )
    t_stack = time.perf_counter()

    lat_nk = _bulk_fp4_to_bfp4_lattice_bf16(packed)
    del packed
    lat_kn = lat_nk.transpose(-2, -1).contiguous()
    del lat_nk
    t_lat = time.perf_counter()

    # Split _shard_dump into from_torch + dump_tensor for finer timing.
    if profile:
        E = n_experts
        per_chip = E // (rows * cols)
        reshaped = lat_kn.view(rows, cols, per_chip, K, N).contiguous()
        del lat_kn
        host_tt = ttnn.from_torch(
            reshaped, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
            device=None, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (rows, cols), dims=(0, 1)),
        )
        del reshaped
        t_w_ft = time.perf_counter()
        out_path = layer_dir / f"{wname}.tensorbin"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ttnn.dump_tensor(str(out_path), host_tt)
        del host_tt
        t_w_dump = time.perf_counter()
    else:
        _shard_dump(
            lat_kn,
            mesh=mesh, rows=rows, cols=cols,
            out_path=layer_dir / f"{wname}.tensorbin",
            target_dtype=ttnn.bfloat4_b,
        )
        t_w_ft = t_w_dump = time.perf_counter()

    scale_kn = _bulk_e8m0_to_bf16_compact(scales, K=K, N=N)
    del scales
    t_scale_lat = time.perf_counter()
    _shard_dump(
        scale_kn,
        mesh=mesh, rows=rows, cols=cols,
        out_path=layer_dir / f"{wname}_scale.tensorbin",
        target_dtype=ttnn.bfloat16,
    )

    dt = time.perf_counter() - t0
    wsize = (layer_dir / f"{wname}.tensorbin").stat().st_size
    ssize = (layer_dir / f"{wname}_scale.tensorbin").stat().st_size
    if profile:
        print(f"  layer {layer:>3d} {wname:<3s}  K={K:<5d} N={N:<5d}  "
              f"stack={t_stack-t0:.1f}s  lat={t_lat-t_stack:.1f}s  "
              f"w_ft={t_w_ft-t_lat:.1f}s  w_dump={t_w_dump-t_w_ft:.1f}s  "
              f"s_lat={t_scale_lat-t_w_dump:.1f}s  s_dump={time.perf_counter()-t_scale_lat:.1f}s  "
              f"total={dt:.1f}s",
              flush=True)
    else:
        print(f"  layer {layer:>3d} {wname:<3s}  K={K:<5d} N={N:<5d}  "
              f"wbytes={wsize:>10d}  sbytes={ssize:>10d}  ({dt:.1f}s)",
              flush=True)
    return {
        "wname": wname,
        "weight": f"{wname}.tensorbin",
        "scale":  f"{wname}_scale.tensorbin",
        "K": K, "N": N,
        "weight_bytes": wsize, "scale_bytes": ssize,
    }


def _process_layer(
    state: dict,
    layer: int,
    *,
    mesh,
    rows: int,
    cols: int,
    n_experts: int,
    dim: int,
    inter_dim: int,
    layer_dir: Path,
    workers: int,
    profile: bool = False,
) -> dict:
    """Build and dump the 6 .tensorbin files for one MoE layer. Returns a
    summary dict for the manifest. With workers>1, runs w1/w3/w2 concurrently
    on a thread pool (host pipeline is independent across the three)."""
    summary: dict = {"layer": layer, "files": {}}
    wnames = ("w1", "w3", "w2")
    if workers <= 1:
        results = [
            _process_one_weight(
                state, layer, w,
                mesh=mesh, rows=rows, cols=cols, n_experts=n_experts,
                dim=dim, inter_dim=inter_dim, layer_dir=layer_dir,
                profile=profile,
            )
            for w in wnames
        ]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(wnames))) as ex:
            futures = [
                ex.submit(
                    _process_one_weight, state, layer, w,
                    mesh=mesh, rows=rows, cols=cols, n_experts=n_experts,
                    dim=dim, inter_dim=inter_dim, layer_dir=layer_dir,
                    profile=profile,
                )
                for w in wnames
            ]
            results = [f.result() for f in futures]
    for r in results:
        wname = r.pop("wname")
        summary["files"][wname] = r
    return summary


# -----------------------------------------------------------------------------
# End-to-end validation (load back, run gemm, compare to reference)
# -----------------------------------------------------------------------------


def _validate_one_expert(
    state: dict, *, mesh, rows: int, cols: int,
    layer: int, expert_idx: int, wname: str, dim: int, inter_dim: int,
    cache_dir: Path,
) -> float:
    """Load back the cache file for (layer, wname), pull out a single expert's
    slice, and PCC-compare a small matmul against the reference fp4 dequant."""
    ws = _shape_for(wname, dim=dim, inter_dim=inter_dim)
    K, N = ws.K, ws.N
    Kb = K // FP4_BLOCK_K
    M = 32

    # Reference: lossless host dequant + bf16 matmul on this one expert.
    kw = f"layers.{layer}.ffn.experts.{expert_idx}.{wname}.weight"
    ks = f"layers.{layer}.ffn.experts.{expert_idx}.{wname}.scale"
    w_packed = state[kw]
    w_scale = state[ks]
    if w_packed.dtype == torch.float4_e2m1fn_x2:
        w_packed = w_packed.view(torch.uint8)
    w_ref_nk = fp4_bytes_to_fp4_dequant_bf16(w_packed, w_scale)
    w_ref_kn = w_ref_nk.transpose(0, 1).contiguous()
    torch.manual_seed(layer * 1000 + expert_idx)
    x = (torch.randn(M, K, dtype=torch.bfloat16) * 0.25).bfloat16()
    y_ref = (x.float() @ w_ref_kn.float()).to(torch.bfloat16)

    # Loaded cache: read full sharded tensor, gather to host, slice this expert.
    layer_dir = cache_dir / f"layer_{layer:03d}"
    w_tt = ttnn.load_tensor(str(layer_dir / f"{wname}.tensorbin"), device=mesh)
    sf_tt = ttnn.load_tensor(str(layer_dir / f"{wname}_scale.tensorbin"), device=mesh)

    composer = ttnn.ConcatMesh2dToTensor(mesh, (rows, cols), dims=(0, 1))
    w_back = ttnn.to_torch(w_tt, mesh_composer=composer)
    sf_back = ttnn.to_torch(sf_tt, mesh_composer=composer)

    # We packed experts as [rows, cols, per_chip, ...]. Decode the linear
    # expert index back to (r, c, p).
    if w_back.dim() != 5 or sf_back.dim() != 5:
        raise RuntimeError(
            f"unexpected loaded shapes w={tuple(w_back.shape)} sf={tuple(sf_back.shape)}"
        )
    rows_g, cols_g, per_chip_g, _, _ = w_back.shape
    flat_idx = expert_idx
    r = flat_idx // (cols_g * per_chip_g)
    c = (flat_idx // per_chip_g) % cols_g
    p = flat_idx % per_chip_g
    w_one = w_back[r, c, p].to(torch.bfloat16)            # [K, N]
    sf_one = sf_back[r, c, p].to(torch.bfloat16)          # [Kb, N]
    if w_one.shape != (K, N) or sf_one.shape != (Kb, N):
        raise RuntimeError(
            f"loaded slice shape mismatch: w={tuple(w_one.shape)} expected ({K},{N}); "
            f"sf={tuple(sf_one.shape)} expected ({Kb},{N})"
        )

    # Replicate the slice across the mesh and run gemm. Use mesh_composer=None
    # on read so we get a single replica back, not rows*cols stacked copies.
    common = dict(layout=ttnn.TILE_LAYOUT, device=mesh,
                  memory_config=ttnn.DRAM_MEMORY_CONFIG,
                  mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, **common)
    w_one_tt = ttnn.from_torch(w_one, dtype=ttnn.bfloat4_b, **common)
    sf_one_tt = ttnn.from_torch(sf_one, dtype=ttnn.bfloat16, **common)
    y_tt = fp4_gemm_via_bfp4(x_tt, w_one_tt, sf_one_tt)
    y_back_all = ttnn.to_torch(
        y_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, (rows, cols), dims=(0, 1)),
    )
    # Replicated tensor concatenated as (rows, cols) tiles -> [M*rows, N*cols].
    # Take the (0, 0) replica.
    if y_back_all.dim() != 2:
        raise RuntimeError(f"unexpected y_back shape {tuple(y_back_all.shape)}")
    y_back = y_back_all[:M, :N].to(torch.bfloat16)

    a, b = y_ref.float().flatten(), y_back.float().flatten()
    am, bm = a - a.mean(), b - b.mean()
    pcc = ((am * bm).sum() / (am.norm() * bm.norm()).clamp_min(1e-12)).item()
    return pcc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", required=True,
                   help="Path to existing state_dict.pt (HF-derived; fp4 expert weights).")
    p.add_argument("--out", required=True,
                   help="Output cache directory (will be created).")
    p.add_argument("--mesh-rows", type=int, default=4)
    p.add_argument("--mesh-cols", type=int, default=8)
    p.add_argument("--n-experts", type=int, default=256)
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=2048)
    p.add_argument("--n-hash-layers", type=int, default=3)
    p.add_argument("--validate", action="store_true",
                   help="At end, load back and PCC-check one expert per layer.")
    p.add_argument("--limit-layers", type=int, default=0,
                   help="If >0, only process this many layers (for testing).")
    p.add_argument("--workers", type=int, default=3,
                   help="Threads per layer (1=serial w1->w3->w2; 3=full intra-layer parallel).")
    p.add_argument("--profile", action="store_true",
                   help="Print fine-grained per-stage timing for each (layer, wname).")
    p.add_argument("--resume", action="store_true",
                   help="Skip layers whose 6 .tensorbin files all already exist "
                        "(checked by file presence, not size). Manifest is "
                        "rebuilt from disk for skipped layers.")
    args = p.parse_args()

    state_path = Path(args.state_dict)
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {state_path}", flush=True)
    t0 = time.perf_counter()
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
    print(f"[load] read {len(state)} keys in {time.perf_counter()-t0:.1f}s",
          flush=True)

    layers = sorted(_scan_routed_experts(state).keys())
    if args.n_hash_layers > 0:
        layers = [L for L in layers if L >= args.n_hash_layers]
    if args.limit_layers:
        layers = layers[:args.limit_layers]
    print(f"[scan] {len(layers)} MoE layers to process: "
          f"{layers[0]}..{layers[-1]}",
          flush=True)

    print(f"[mesh] opening ({args.mesh_rows}, {args.mesh_cols}) FABRIC_1D",
          flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(args.mesh_rows, args.mesh_cols),
        trace_region_size=10_000_000,
    )

    manifest = {
        "format_version": 1,
        "model": "DeepSeek-V4-Flash routed-experts (lossless bfp4_b storage of fp4 e2m1 nibbles)",
        "mesh_shape": [args.mesh_rows, args.mesh_cols],
        "n_experts": args.n_experts,
        "dim": args.dim,
        "inter_dim": args.inter_dim,
        "n_hash_layers": args.n_hash_layers,
        "fp4_block_size": FP4_BLOCK_K,
        "tile_size": TILE,
        "weight_dtype": "bfloat4_b",
        "scale_dtype": "bfloat16_compact_KbN",
        "layers": [],
    }

    try:
        wall0 = time.perf_counter()
        for L in layers:
            _validate_layer_keys(state, L, args.n_experts)
            layer_dir = out_dir / f"layer_{L:03d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            expected_files = [
                f"{w}{suf}.tensorbin"
                for w in ("w1", "w3", "w2")
                for suf in ("", "_scale")
            ]
            if args.resume and all((layer_dir / f).is_file() for f in expected_files):
                summary = {"layer": L, "files": {}}
                for w in ("w1", "w3", "w2"):
                    ws = _shape_for(w, dim=args.dim, inter_dim=args.inter_dim)
                    summary["files"][w] = {
                        "weight": f"{w}.tensorbin",
                        "scale":  f"{w}_scale.tensorbin",
                        "K": ws.K, "N": ws.N,
                        "weight_bytes": (layer_dir / f"{w}.tensorbin").stat().st_size,
                        "scale_bytes":  (layer_dir / f"{w}_scale.tensorbin").stat().st_size,
                    }
                summary["walltime_s"] = 0.0
                manifest["layers"].append(summary)
                with open(out_dir / "manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)
                print(f"[layer {L}] skipped (resume; cache hit)", flush=True)
                continue
            t0 = time.perf_counter()
            summary = _process_layer(
                state, L,
                mesh=mesh,
                rows=args.mesh_rows, cols=args.mesh_cols,
                n_experts=args.n_experts,
                dim=args.dim, inter_dim=args.inter_dim,
                layer_dir=layer_dir,
                workers=args.workers,
                profile=args.profile,
            )
            summary["walltime_s"] = round(time.perf_counter() - t0, 2)
            manifest["layers"].append(summary)
            with open(out_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"[layer {L}] done in {summary['walltime_s']}s "
                  f"(cumulative {time.perf_counter()-wall0:.1f}s)",
                  flush=True)

        if args.validate:
            # Sample first/middle/last layer × {w1, w3, w2} × first/last expert.
            # load_tensor is ~0.3s/file, so this stays well under a minute.
            print("\n[validate] sampling layers x weights x experts", flush=True)
            sample_layers = sorted({layers[0], layers[len(layers)//2], layers[-1]})
            sample_experts = (0, args.n_experts - 1)
            for L in sample_layers:
                for wname in ("w1", "w3", "w2"):
                    for E in sample_experts:
                        pcc = _validate_one_expert(
                            state, mesh=mesh,
                            rows=args.mesh_rows, cols=args.mesh_cols,
                            layer=L, expert_idx=E, wname=wname,
                            dim=args.dim, inter_dim=args.inter_dim,
                            cache_dir=out_dir,
                        )
                        print(f"  layer {L:>3d} expert {E:>3d} {wname}: PCC={pcc:.6f}", flush=True)
                        if pcc < 0.999:
                            raise AssertionError(
                                f"layer {L} expert {E} {wname}: PCC {pcc:.6f} < 0.999")

        print(f"\n[done] total wall {time.perf_counter()-wall0:.1f}s", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
