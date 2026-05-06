"""Compare seam dumps from legacy and mega.

Usage: python3 compare_seams.py /path/legacy /path/mega
Walks both dirs, prints PCC + max|Δ| per matching file, sorted by
(token, layer, name). First row with PCC < 0.99 or shape mismatch is
likely the smoking gun.
"""
import sys
from pathlib import Path
import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if a.numel() != b.numel():
        return float("nan")
    am = a - a.mean()
    bm = b - b.mean()
    denom = (am.norm() * bm.norm()).item()
    if denom == 0.0:
        if (am.norm().item() == 0.0 and bm.norm().item() == 0.0):
            return 1.0
        return 0.0
    return (am @ bm).item() / denom


def parse_key(stem: str):
    parts = stem.split("_", 2)
    if len(parts) < 3 or not parts[0].startswith("T") or not parts[1].startswith("L"):
        return None
    return int(parts[0][1:]), int(parts[1][1:]), parts[2]


def main(a_dir: str, b_dir: str) -> int:
    A = Path(a_dir)
    B = Path(b_dir)
    if not A.is_dir() or not B.is_dir():
        print(f"missing dir: {A} or {B}", file=sys.stderr)
        return 2
    a_files = {f.name: f for f in A.glob("T*_L*.pt")}
    b_files = {f.name: f for f in B.glob("T*_L*.pt")}
    only_a = sorted(set(a_files) - set(b_files))
    only_b = sorted(set(b_files) - set(a_files))
    common = sorted(set(a_files) & set(b_files), key=lambda n: parse_key(Path(n).stem) or (99, 99, n))
    if only_a:
        print(f"[only in {A.name}] {len(only_a)} files: {only_a[:5]}{' ...' if len(only_a) > 5 else ''}")
    if only_b:
        print(f"[only in {B.name}] {len(only_b)} files: {only_b[:5]}{' ...' if len(only_b) > 5 else ''}")
    print(f"{'tok':>3} {'lyr':>3} {'name':<32} {'shape_a':<24} {'shape_b':<24} {'pcc':>9} {'max|Δ|':>10}")
    print("-" * 110)
    first_bad_printed = False
    for n in common:
        a = torch.load(a_files[n], map_location="cpu", weights_only=True)
        b = torch.load(b_files[n], map_location="cpu", weights_only=True)
        key = parse_key(Path(n).stem)
        tok, lyr, name = key if key else (-1, -1, n)
        sa = "x".join(str(x) for x in a.shape)
        sb = "x".join(str(x) for x in b.shape)
        if a.shape != b.shape:
            print(f"{tok:>3} {lyr:>3} {name:<32} {sa:<24} {sb:<24} {'SHAPE!':>9} {'-':>10}")
            continue
        p = pcc(a, b)
        max_d = (a.float() - b.float()).abs().max().item()
        marker = " <<< FIRST_BAD" if (p < 0.99 and not first_bad_printed) else ""
        if p < 0.99 and not first_bad_printed:
            first_bad_printed = True
        print(f"{tok:>3} {lyr:>3} {name:<32} {sa:<24} {sb:<24} {p:>9.4f} {max_d:>10.3e}{marker}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: compare_seams.py <legacy_dir> <mega_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
