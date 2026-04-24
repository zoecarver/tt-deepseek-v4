# Priority TT-Lang kernels for DeepSeek-V4-Flash

Kernels where TT-Lang is expected to beat ttnn, either because (a) the op
doesn't exist in ttnn, or (b) fusion with neighboring ops avoids a DRAM
round-trip ttnn would incur. Each kernel is anchored to a DeepSeek reference
impl + test in `./TileKernels/`.

For each kernel below, the deliverable is:
- A TT-Lang kernel matching the listed signature.
- A PCC test (target PCC >0.999) over the listed input shapes, compared to a
  CPU torch reference. See `./scripts/test_lm_head.py` for the pattern.

## Priority ordering

1. **mHC Sinkhorn normalization** — not in ttnn; on critical path every block.
2. **mHC pre-big-fuse** — not in ttnn; huge fusion win (RMSNorm + linear + split + sinkhorn + apply).
3. **mHC post** — not in ttnn; companion to pre-big-fuse on the output side.
4. **Fused RMSNorm + FP8 per-token quant-cast** — both exist in ttnn separately, but fused saves one DRAM trip and hits every projection in V4-Flash.
5. **Fused SwiGLU forward + FP8 per-channel cast + transpose** — MoE FFN critical path; no ttnn equivalent.

Items 1-3 are the biggest wins because ttnn simply has no mHC kernel — without TT-Lang we'd fall back to many small elementwise ttnn ops per layer. Items 4-5 are pure fusion plays.

---

## 1. mHC Sinkhorn normalization

**Why TT-Lang:** ttnn has no hyper-connection ops. V4-Flash calls Sinkhorn once per block (43 blocks × 1 call = 43 invocations per token). It runs after the HC pre-split and normalizes the `comb_res_mix` tensor so rows and columns both sum to 1 via alternating row/col normalization for `repeat` iterations.

**Reference impl:** `TileKernels/tile_kernels/mhc/sinkhorn_kernel.py`
**Reference test:** `TileKernels/tests/mhc/test_sinkhorn.py`

**Signature:**
```python
def sinkhorn_normalize(
    comb_res_mix: torch.Tensor,  # shape [n0, n1, mhc, mhc], dtype fp32
    repeat: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:  # same shape, fp32
```

**Test shapes** (from `test_sinkhorn.py`):
- `(n0, n1, mhc) ∈ {(1, 1, 4), (1, 1024, 4), (1, 4096, 4), (2, 1, 4), (2, 1024, 4), (2, 4096, 4)}`
- In V4-Flash, `mhc_mult=4` always. `n1` is the sequence length; for decode `n1=1`, for prefill `n1=seq_len`.
- Only need forward. Backward is not on the inference path.

**Per-block work:** for each `(n0, n1)` slice, iterate `repeat` times alternating row-normalize (÷ row_sum) and col-normalize (÷ col_sum). Tiny 4×4 blocks so this is latency-bound; fusion with the caller (pre-big-fuse) is where the real win comes from.

---

## 2. mHC pre-big-fuse (RMSNorm + linear + split + sinkhorn + apply)

**Why TT-Lang:** Fuses five ops into one kernel:
1. RMSNorm over `hidden_size`.
2. Linear projection by `fn` to produce `mixes`.
3. Split `mixes` into `pre_mix` / `post_mix` / `comb_mix` with scale+base+sigmoid.
4. Sinkhorn on `comb_mix`.
5. Apply `pre_mix` to `residual` to produce `layer_input`.

In ttnn this would be 5+ separate ops with ~4 DRAM round-trips. Runs at the start of every block (43×).

**Reference impl:** `TileKernels/tile_kernels/mhc/pre_big_fuse_kernel.py`
**Reference test:** `TileKernels/tests/mhc/test_pre_big_fuse.py`

**Signature** (from `big_fuse_reference` in the test):
```python
def mhc_pre_big_fuse(
    residual: torch.Tensor,       # [1, n1, mhc_mult, hidden], bf16
    fn: torch.Tensor,             # [mhc_mult3, mhc_mult * hidden], fp32, where mhc_mult3 = 2*mhc_mult + mhc_mult**2
    mhc_scale: torch.Tensor,      # [3], fp32
    mhc_base: torch.Tensor,       # [mhc_mult3], fp32
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # post_mix:    [1, n1, mhc_mult], fp32
    # comb_mix:    [1, n1, mhc_mult, mhc_mult], fp32 (sinkhorn-normalized)
    # layer_input: [1, n1, hidden], bf16 (residual + pre_mix applied)
```

**Test shapes** (from `test_pre_big_fuse.py`):
- `n1 ∈ {512, 1024, 2048, 8192}`, `hidden_size ∈ {1280, 2560, 4096}`, `mhc_mult=4`.
- V4-Flash is `hidden=4096`, `mhc_mult=4`. `n1` varies with prefill length.

**Sub-kernels** this composes:
- `mhc_pre_norm_fn` — `tile_kernels/mhc/norm_fn_kernel.py`
- `mhc_pre_split_mixes` — `tile_kernels/mhc/pre_split_mixes_kernel.py`
- `sinkhorn_normalize` — kernel #1 above
- `mhc_pre_apply_mix` — `tile_kernels/mhc/pre_apply_mix_kernel.py`

If bringing up in stages: build #1 (sinkhorn) first, then build each sub-kernel independently with its own PCC test (`test_norm_fn.py`, `test_pre_split_mixes.py`, `test_pre_apply_mix.py`), then fuse.

---

## 3. mHC post

**Why TT-Lang:** Companion to pre-big-fuse on the output side. Merges the HC streams back with the residual after attention/FFN. ttnn has none of this composed. Runs at the end of every block (43×).

**Reference impl:** `TileKernels/tile_kernels/mhc/post_kernel.py`
**Reference test:** `TileKernels/tests/mhc/test_post.py`

**Signature:** Read the test file for the exact form; expect it to take the attention/FFN output plus `post_mix` and `comb_mix` from pre-big-fuse and produce the new residual stream. Same shape class as pre-big-fuse.

---

## 4. Fused RMSNorm + FP8 per-token quant-cast

**Why TT-Lang:** ttnn has `rms_norm` and fp8 quant separately. Every attention projection and FFN linear in V4-Flash does `x → norm → quant → matmul`. Fusing norm and quant into one kernel eliminates the bf16 intermediate write-then-read from DRAM. V4-Flash hits this pattern ~6 times per block × 43 blocks = ~260 times per token in decode.

**Reference quant impl:** `TileKernels/tile_kernels/quant/per_token_cast_kernel.py`
**Reference quant test:** `TileKernels/tests/quant/test_per_token_cast.py`

**Signature:**
```python
def rmsnorm_fp8_per_token_cast(
    x: torch.Tensor,        # [num_tokens, hidden], bf16
    gamma: torch.Tensor,    # [hidden], bf16 (RMSNorm weight)
    rms_eps: float = 1e-6,
    num_per_channels: int = 128,   # quant block size along hidden
    round_sf: bool = True,
    use_packed_ue8m0: bool = True, # V4-Flash uses ue8m0 scales
) -> tuple[torch.Tensor, torch.Tensor]:
    # x_fp8:  [num_tokens, hidden], float8_e4m3fn
    # x_sf:   [num_tokens, hidden // num_per_channels], float8_e8m0fnu (ue8m0)
```

**Test shapes:**
- `num_tokens ∈ {1, 32, 128, 1024, 4096}`, `hidden ∈ {1280, 2560, 4096}`, `num_per_channels ∈ {32, 128}`.
- For V4-Flash, the common shape is `hidden=4096`, `num_per_channels=128` (block size 128 on the contraction dim).

**Reference:** take the per-token quant test in `test_per_token_cast.py`, prepend a torch `RMSNorm` (weight `gamma`, eps), and assert the fused kernel matches.

---

## 5. Fused SwiGLU forward + FP8 per-channel cast + transpose

**Why TT-Lang:** MoE expert FFN. Computes `SwiGLU(x)` on a `[num_tokens, 2*hidden]` input (the gate+up concat), then quantizes along the token dimension and transposes for the next FP8 matmul. In ttnn this is 3+ ops; fused, it's one kernel. Critical path in every MoE expert invocation — each token routes to ~6 experts, and each expert hits this kernel.

**Reference impl:** `TileKernels/tile_kernels/quant/swiglu_forward_and_per_channel_cast_and_transpose_kernel.py`
**Reference test:** `TileKernels/tests/quant/test_swiglu_forward_and_per_channel_cast_and_transpose.py`

**Signature** (from test):
```python
def swiglu_forward_and_per_channel_cast_and_transpose(
    x: torch.Tensor,              # [num_tokens, 2 * hidden], bf16
    fmt: str = 'e4m3',
    num_per_tokens: int = 128,    # quant block size along num_tokens
    round_sf: bool = True,
    without_transpose: bool = False,
    swiglu_clamp_value: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # x_fp8: [num_tokens, hidden] fp8_e4m3 if without_transpose else [hidden, num_tokens]
    # x_sf:  per-channel scales
```

**Test shapes:**
- `num_tokens ∈ {128, ..., large}`, `hidden ∈ {1280, 2560, 4096}`, `num_per_tokens ∈ {32, 128}`, `swiglu_clamp_value ∈ {None, 10.0, 0.5}`.
- V4-Flash MoE inter-dim is `2048`, so `hidden=2048` is the primary target shape. Per-expert `num_tokens` varies with routing.

---

## Out of scope for this batch

These are important but I'd pick ttnn first or defer:

- **Dense FlashAttention** — ttnn has working FA demos (gpt_oss, llama). Only needed for prefill.
- **Top-k / argtop-k** — ttnn has `ttnn.topk`. Caveat: for non-power-of-2 widths the Blackhole bitonic path falls back and tanks; worth benchmarking before rewriting.
- **FP8 block-scale GEMM** — ttnn matmul with block-scale is the existing path; writing a new GEMM is a huge lift for questionable win unless profiling shows it's the bottleneck.
- **Sparse gather-attention** — biggest decode perf unlock but not yet scoped; plan to tackle after the mHC ops land so the full block executes end-to-end on device.
- **Engram kernels** (`TileKernels/tile_kernels/engram/`) — not on the V4-Flash inference path (they're for DeepSeek's memory system). Skip.

## PCC test pattern

All tests should match the shape of `./scripts/test_lm_head.py`:

1. Build a CPU torch reference (use the existing `tile_kernels.torch.mhc.*_ref` or the `*_reference` functions in each TileKernels test).
2. Run the TT-Lang kernel on `--hw` with matching inputs.
3. Compare via `pcc()` (target >0.999 for fp16/bf16, >0.9995 for fp32) and `max_abs_diff`.
4. Test shapes: at minimum the V4-Flash production shape; ideally sweep the shape matrix from the reference test.
