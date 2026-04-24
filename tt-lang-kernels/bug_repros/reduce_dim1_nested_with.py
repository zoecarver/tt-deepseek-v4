"""Iteration-friendly reduce dims=[1] test using the nested-with pattern from
`tt-lang/test/python/simple_reduce_bcast.py`, with the working workaround
`options="--no-ttl-reduce-full-fp32"` applied to @ttl.operation.

Structure mirrors the upstream test exactly: the inputs are waited in one
outer `with`, the reduce stores into `red_dfb` inside a nested `reserve`, and
the broadcast waits `red_dfb` and reserves `out_dfb` in a follow-on `with`
block. The upstream test exercises dims=[0, 1] scalar reduce in bf16; this
file substitutes dims=[1] and runs both bf16 and fp32.

The companion file `reduce_dim1_fp32_returns_zeros.py` shows the same kernel
without the workaround and demonstrates the zeros output for fp32.
"""
from __future__ import annotations

import torch
import ttnn
import ttl

TILE = 32


@ttl.operation(grid=(1, 1), options="--no-ttl-reduce-full-fp32")
def reduce_max_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, sc_dfb.wait() as scaler_blk:
            with red_dfb.reserve() as red_blk:
                red_blk.store(ttl.math.reduce_max(inp_blk, scaler_blk, dims=[1]))
            with red_dfb.wait() as red_blk, out_dfb.reserve() as out_blk:
                out_blk.store(ttl.math.broadcast(red_blk, out_blk, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()
        with sc_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1), options="--no-ttl-reduce-full-fp32")
def reduce_sum_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, sc_dfb.wait() as scaler_blk:
            with red_dfb.reserve() as red_blk:
                red_blk.store(ttl.math.reduce_sum(inp_blk, scaler_blk, dims=[1]))
            with red_dfb.wait() as red_blk, out_dfb.reserve() as out_blk:
                out_blk.store(ttl.math.broadcast(red_blk, out_blk, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()
        with sc_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


KERNELS = {"reduce_max": reduce_max_kernel, "reduce_sum": reduce_sum_kernel}


def run(device, dtype_torch: torch.dtype, dtype_ttnn, reduce_op_name: str) -> torch.Tensor:
    x = torch.zeros(TILE, TILE, dtype=dtype_torch)
    for r in range(TILE):
        x[r, :] = float(r)
    out = torch.zeros_like(x)
    sc = torch.ones(TILE, TILE, dtype=dtype_torch)

    common = dict(dtype=dtype_ttnn, layout=ttnn.TILE_LAYOUT,
                  device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
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
