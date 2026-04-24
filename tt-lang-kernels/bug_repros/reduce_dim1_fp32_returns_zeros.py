"""Minimal reproducer: ttl.math.reduce_max/reduce_sum with dims=[1] returns all
zeros when the input tile is fp32, but returns correct per-row values when the
same test is run in bfloat16.

Pattern from `oasis-ttlang/tests/_smoke_reduce_after_compute.py::make_direct`
(which itself exercises `reduce_max(x, scaler, dims=[1])` followed by
`broadcast(..., dims=[1])`).

How to reproduce
----------------
    export TT_REMOTE_CONF=<your remote config>
    run-test.sh --hw reduce_dim1_fp32_returns_zeros.py

The script runs the same kernel twice, once with ttnn.float32 and once with
ttnn.bfloat16. Input tile has row r filled with the scalar value r, so
reduce_max/reduce_sum along dims=[1] should yield r for row r, and the
subsequent broadcast should produce an output tile where row r is r repeated
32 times.

Observed (sterling, ttlang 1.0.0.dev4, Blackhole):
    [fp32]   out[:, 0] = [0, 0, 0, ...]            <- WRONG (all zeros)
    [bf16]   out[:, 0] = [0, 1, 2, ..., 31]        <- correct

Expected:
    Both dtypes should produce [0, 1, 2, ..., 31] in out[:, 0].

Notes
-----
- dims=[0] reduce works correctly for fp32 (produces per-column results).
- dims=[0, 1] scalar reduce works correctly for fp32.
- Only dims=[1] is affected.
- Issue reproduces with both reduce_max and reduce_sum.
- Root cause is the fp32-accumulation lowering for reduce. Adding
  `options="--no-ttl-reduce-full-fp32"` to `@ttl.operation` is a
  working workaround (see `reduce_dim1_nested_with.py`).
- `fp32_dest_acc_en=True`, `dst_full_sync_en=False`, L1 memory config,
  and `--no-ttl-maximize-dst` do NOT fix it on their own.
"""
from __future__ import annotations

import torch
import ttnn
import ttl

TILE = 32


@ttl.operation(grid=(1, 1))
def reduce_max_kernel(x, scaler, out):
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
    red_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        sc = sc_dfb.wait()
        xb = x_dfb.wait()
        red_dfb.reserve().store(ttl.math.reduce_max(xb, sc, dims=[1]))
        mb = out_dfb.reserve()
        mb.store(ttl.math.broadcast(red_dfb.wait(), mb, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
        ttl.copy(x[0, 0], x_dfb.reserve()).wait()

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_dfb.wait(), out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def reduce_sum_kernel(x, scaler, out):
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
    red_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        sc = sc_dfb.wait()
        xb = x_dfb.wait()
        red_dfb.reserve().store(ttl.math.reduce_sum(xb, sc, dims=[1]))
        mb = out_dfb.reserve()
        mb.store(ttl.math.broadcast(red_dfb.wait(), mb, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
        ttl.copy(x[0, 0], x_dfb.reserve()).wait()

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_dfb.wait(), out[0, 0]).wait()


KERNELS = {"reduce_max": reduce_max_kernel, "reduce_sum": reduce_sum_kernel}


def run(device, dtype_torch: torch.dtype, dtype_ttnn, reduce_op_name: str) -> torch.Tensor:
    x = torch.zeros(TILE, TILE, dtype=dtype_torch)
    for r in range(TILE):
        x[r, :] = float(r)
    out = torch.zeros_like(x)
    sc = torch.ones(TILE, TILE, dtype=dtype_torch)

    common = dict(dtype=dtype_ttnn, layout=ttnn.TILE_LAYOUT,
                  device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_tt = ttnn.from_torch(x, **common)
    out_tt = ttnn.from_torch(out, **common)
    sc_tt = ttnn.from_torch(sc, **common)

    KERNELS[reduce_op_name](x_tt, sc_tt, out_tt)

    return ttnn.to_torch(out_tt).float()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        rows = torch.arange(TILE, dtype=torch.float32)
        expected = {"reduce_max": rows, "reduce_sum": rows * TILE}
        for op in ("reduce_max", "reduce_sum"):
            for label, dtype_torch, dtype_ttnn in (
                ("bf16", torch.bfloat16, ttnn.bfloat16),
                ("fp32", torch.float32, ttnn.float32),
            ):
                out = run(device, dtype_torch, dtype_ttnn, op)
                col0 = out[:, 0]
                ok = torch.allclose(col0, expected[op], atol=0.5)
                status = "OK  " if ok else "BUG "
                print(f"[{status}] {op:12s} dims=[1] dtype={label}  "
                      f"col0[:8]={col0[:8].tolist()}")
    finally:
        ttnn.close_device(device)
