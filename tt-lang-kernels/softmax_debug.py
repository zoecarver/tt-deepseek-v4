"""Minimal row-softmax debug kernel: isolates the first step of sinkhorn.

Purpose: verify the reduce_max + exp(x - broadcast(max)) + reduce_sum + recip +
broadcast pipeline produces correct row-softmax values on a single 32x32 tile.
Compares against torch.softmax(x, dim=-1) on the top-left 4x4 valid region.

Run: TT_REMOTE_CONF=.../sterling-all.conf run-test.sh --hw softmax_debug.py
"""
from __future__ import annotations

import ttl

TILE = 32


@ttl.operation(grid=(1, 1))
def softmax_kernel(x, scaler, out):
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    inv_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        sc = sc_dfb.wait()
        x_in = x_dfb.wait()
        with out_dfb.reserve() as xs:
            red_dfb.reserve().store(ttl.math.reduce_max(x_in, sc, dims=[1]))
            xs.store(ttl.math.exp(x_in - ttl.math.broadcast(red_dfb.wait(), x_in, dims=[1])))
            red_dfb.reserve().store(ttl.math.reduce_sum(xs, sc, dims=[1]))
            inv_dfb.reserve().store(ttl.math.recip(red_dfb.wait()))
            xs.store(xs * ttl.math.broadcast(inv_dfb.wait(), x_in, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
        ttl.copy(x[0, 0], x_dfb.reserve()).wait()

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_dfb.wait(), out[0, 0]).wait()


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn
    from harness import PAD_SENTINEL, assert_pcc, pack_4x4_slices, scaler_tile, unpack_4x4_slices

    torch.manual_seed(0)
    x_ref = torch.randn((1, 4, 4), dtype=torch.float32)
    # Torch softmax on the valid 4x4 (no padding), per-row.
    y_ref = x_ref.softmax(dim=-1)

    x_packed = pack_4x4_slices(x_ref, pad_value=PAD_SENTINEL)
    out_packed = torch.zeros_like(x_packed)
    sc = scaler_tile()

    device = ttnn.open_device(device_id=0)
    try:
        common = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                      device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_tt = ttnn.from_torch(x_packed, **common)
        out_tt = ttnn.from_torch(out_packed, **common)
        sc_tt = ttnn.from_torch(sc, **common)

        softmax_kernel(x_tt, sc_tt, out_tt)

        out_packed = ttnn.to_torch(out_tt)
        y_tt = unpack_4x4_slices(out_packed, 1)

        print(f"x_in:\n{x_ref[0].numpy()}")
        print(f"y_ref (torch.softmax):\n{y_ref[0].numpy()}")
        print(f"y_tt:\n{y_tt[0].numpy()}")
        print(f"raw tile top-left 8x8:\n{out_packed[:8, :8].numpy()}")
        print(f"raw tile row 0 all 32:\n{out_packed[0, :].numpy()}")

        assert_pcc(y_ref, y_tt, threshold=0.9995)
        print("SOFTMAX PASSED")
    finally:
        ttnn.close_device(device)
