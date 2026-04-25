"""Smoke test: instantiate the inference.py consolidated DeviceCompressor +
DeviceIndexer once each (no_overlap and overlap+rotate), do one decode step,
and check it runs without error. Numerical PCC validation lives in s0/s1/s2/s3.
"""
from __future__ import annotations

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import ttnn

from harness import (
    load_inference, make_test_args, fresh_compressor,
    replicated, to_torch_replicated, report, open_mesh,
)

inf = load_inference()


def build_indexer():
    args = make_test_args(inf, max_seq_len=64)
    g = torch.Generator().manual_seed(11)
    ix = inf.Indexer(args, compress_ratio=4).eval()
    for p in ix.parameters():
        with torch.no_grad():
            p.copy_(torch.empty_like(p).normal_(generator=g) * 0.02)
    ix.kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len // 4, ix.head_dim)
    rd = ix.rope_head_dim
    ix.freqs_cis = inf.precompute_freqs_cis(
        rd, args.max_seq_len, args.original_seq_len, args.compress_rope_theta,
        args.rope_factor, args.beta_fast, args.beta_slow,
    )
    ix.compressor.kv_cache = ix.kv_cache
    ix.compressor.freqs_cis = ix.freqs_cis
    return args, ix


def main() -> int:
    torch.set_default_dtype(torch.bfloat16)
    print("=" * 72)
    print("Smoke: inference.DeviceCompressor + DeviceIndexer (consolidated)")
    print("=" * 72)

    # ratio=128 needs at least 128 decode steps to fire one compress; size
    # max_seq_len so kv_cache has at least one row.
    args = make_test_args(inf, max_seq_len=256)
    cpu_no, dev_no = fresh_compressor(inf, args, compress_ratio=128,
                                      head_dim=args.head_dim, rotate=False, seed=2)
    args_ix, ix = build_indexer()

    with open_mesh(ttnn) as mesh:
        # No-overlap variant.
        wkv = inf.DeviceColLinear(mesh, dev_no.wkv.weight)
        wgate = inf.DeviceColLinear(mesh, dev_no.wgate.weight)
        norm = inf.DeviceRMSNorm(mesh, dev_no.norm.weight, dev_no.norm.eps)
        dc_no = inf.DeviceCompressor(mesh, dev_no, wkv, wgate, norm)
        # Run all 128 steps so a compress fires at the end.
        for step in range(128):
            x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)
            cpu_out = cpu_no(x.clone(), step)
            x_tt = replicated(ttnn, mesh, x)
            dev_out = dc_no.forward_device(x_tt, 1, step)
            if (step + 1) % 128 == 0:
                dev_torch = to_torch_replicated(ttnn, mesh, dev_out)[:1]
                ok = report("no_overlap kv@127", cpu_out, dev_torch, pcc_min=0.99)
                if not ok: return 1

        # Indexer (which exercises overlap=True, rotate=True compressor).
        ix_wkv = inf.DeviceColLinear(mesh, ix.compressor.wkv.weight)
        ix_wgate = inf.DeviceColLinear(mesh, ix.compressor.wgate.weight)
        ix_norm = inf.DeviceRMSNorm(mesh, ix.compressor.norm.weight, ix.compressor.norm.eps)
        ix_dc = inf.DeviceCompressor(mesh, ix.compressor, ix_wkv, ix_wgate, ix_norm)
        wq_b = inf.DeviceColLinear(mesh, ix.wq_b.weight)
        weights_proj = inf.DeviceColLinear(mesh, ix.weights_proj.weight)
        di = inf.DeviceIndexer(mesh, ix, ix_dc, wq_b, weights_proj)

        # 8 steps -> 2 compresses; topk should match.
        all_ok = True
        for step in range(8):
            x = torch.randn(1, 1, args_ix.dim, dtype=torch.bfloat16)
            qr = torch.randn(1, 1, args_ix.q_lora_rank, dtype=torch.bfloat16)
            cpu_idxs = ix(x.clone(), qr.clone(), step, 0)
            x_tt = replicated(ttnn, mesh, x)
            qr_tt = replicated(ttnn, mesh, qr)
            score_tt = di.forward_device_score(x_tt, qr_tt, 1, step)
            T = (step + 1) // 4
            if T == 0: continue
            sc = to_torch_replicated(ttnn, mesh, score_tt)[:1]
            sc = sc.view(1, ix_dc.kv_cache_T_pad)[:, :T]
            k = min(di.index_topk, T)
            dev_idxs = sc.topk(k, dim=-1)[1]
            cpu_set = set(cpu_idxs.flatten().tolist())
            dev_set = set(dev_idxs.flatten().tolist())
            ok = cpu_set == dev_set
            print(f"  [{'OK' if ok else 'FAIL'}] indexer@step{step:02d} (T={T})  cpu={sorted(cpu_set)}  dev={sorted(dev_set)}")
            all_ok &= ok

        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
