"""Stage 1 of sinkhorn: just the row-softmax, isolated.

Once this passes PCC against torch.softmax, we incrementally add the mask,
eps, col-normalize, and iteration loop to build up to the full sinkhorn
kernel. Keeping each stage standalone makes it obvious which step regresses.

Pattern:
  - Each reduce/broadcast result goes into its own DFB (no aliasing with
    persistent state).
  - fp32 tiles require options="--no-ttl-reduce-full-fp32" (tt-lang#533).
"""
from __future__ import annotations

import ttl

TILE = 32


def make_kernel(num_slices: int):

    @ttl.operation(grid=(1, 1), options="--no-ttl-reduce-full-fp32")
    def softmax_kernel(x, scaler, out):
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        rmax_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rmax_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        # exp_dfb is consumed twice: once by reduce_sum, once by the final
        # multiply. We reserve a copy into exp_copy_dfb right after the reduce
        # pops the first block, FA-style (test_fa_simple.py).
        exp_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        exp_copy_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        rsum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        rinv_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            sc = sc_dfb.wait()
            for _ in range(num_slices):
                with x_dfb.wait() as x_in:
                    with rmax_dfb.reserve() as rmax:
                        rmax.store(ttl.math.reduce_max(x_in, sc, dims=[1]))
                    with rmax_dfb.wait() as rmax, rmax_bc_dfb.reserve() as rmx:
                        rmx.store(ttl.math.broadcast(rmax, rmx, dims=[1]))
                    with rmax_bc_dfb.wait() as rmx, exp_dfb.reserve() as ex:
                        ex.store(ttl.math.exp(x_in - rmx))
                with exp_dfb.wait() as ex:
                    with rsum_dfb.reserve() as rsum:
                        rsum.store(ttl.math.reduce_sum(ex, sc, dims=[1]))
                    with exp_copy_dfb.reserve() as ex_copy:
                        ex_copy.store(ex)
                with rsum_dfb.wait() as rsum, rinv_bc_dfb.reserve() as rinv:
                    rinv.store(ttl.math.broadcast(ttl.math.recip(rsum), rinv, dims=[1]))
                with exp_copy_dfb.wait() as ex, rinv_bc_dfb.wait() as rinv, out_dfb.reserve() as o:
                    o.store(ex * rinv)

        @ttl.datamovement()
        def dm_read():
            ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
            for t in range(num_slices):
                ttl.copy(x[t, 0], x_dfb.reserve()).wait()

        @ttl.datamovement()
        def dm_write():
            for t in range(num_slices):
                ttl.copy(out_dfb.wait(), out[t, 0]).wait()

    return softmax_kernel


def solve(x_tt, scaler_tt, out_tt):
    num_slices = x_tt.shape[0] // TILE
    kernel = make_kernel(num_slices)
    kernel(x_tt, scaler_tt, out_tt)


if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import torch
    import ttnn
    from harness import PAD_SENTINEL, assert_pcc, pack_4x4_slices, scaler_tile, unpack_4x4_slices

    torch.manual_seed(0)
    # Single 4x4 slice. torch.softmax(dim=-1) is the reference.
    x_ref = torch.randn((1, 4, 4), dtype=torch.float32)
    y_ref = x_ref.softmax(dim=-1)

    x_packed = pack_4x4_slices(x_ref, pad_value=PAD_SENTINEL, dtype=torch.float32)
    out_packed = torch.zeros_like(x_packed)
    sc = scaler_tile(dtype=torch.float32)

    device = ttnn.open_device(device_id=0)
    try:
        common = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                      device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_tt = ttnn.from_torch(x_packed, **common)
        out_tt = ttnn.from_torch(out_packed, **common)
        sc_tt = ttnn.from_torch(sc, **common)

        solve(x_tt, sc_tt, out_tt)

        out_packed = ttnn.to_torch(out_tt)
        y_tt = unpack_4x4_slices(out_packed, 1)

        print(f"x_in:\n{x_ref[0].numpy()}")
        print(f"y_ref (torch.softmax):\n{y_ref[0].numpy()}")
        print(f"y_tt:\n{y_tt[0].numpy()}")

        assert_pcc(y_ref, y_tt, threshold=0.9995)
        print("SOFTMAX PASSED")
    finally:
        ttnn.close_device(device)
