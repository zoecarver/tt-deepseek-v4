# Fast Weight Loading to TT Devices

## TL;DR

Use `ttnn.as_tensor(..., cache_file_name=...)` to auto-generate a `.tensorbin`
flatbuffer cache on first load. Subsequent loads skip tilization/host-side
conversion and hit a fast flatbuffer path (`load_tensor_flatbuffer`).

For a 149 GB checkpoint (V4-Flash), the expected win is **huge**: first pass
converts HF safetensors → torch → tilized TT tensor and dumps a `.tensorbin`;
every subsequent run maps that cache directly onto the device, bypassing
safetensors parsing, FP4/FP8 dequant, and re-tilization.

## How the caching actually works

Source: `/Users/zcarver/Developer/tt-metal/ttnn/ttnn/operations/core.py:549`

`ttnn.as_tensor(tensor, dtype=..., layout=..., device=..., memory_config=...,
cache_file_name=...)` behaves like `ttnn.from_torch` when `cache_file_name` is
`None`. When `cache_file_name` is set:

1. File name is normalized as
   `{cache_file_name}_dtype_{dtype}_layout_{layout}.tensorbin`
   (so changing dtype or layout invalidates the cache automatically).
2. If that file does not exist, the tensor is tilized on the host and then
   `ttnn._ttnn.tensor.dump_tensor_flatbuffer(...)` writes it to disk. The
   returned tensor is then transferred to the device.
3. If the file exists, `ttnn._ttnn.tensor.load_tensor_flatbuffer(cache_file_name,
   device=device)` reads the flatbuffer straight onto the device.
4. If the cache load raises `RuntimeError`, it silently falls back to the slow
   path and overwrites the cache.

This means **per-tensor caching**. One file per weight is the norm. The
mesh_mapper is forced to the unsharded form when caching
`ReplicateTensorToMeshWrapper` so a single cache is portable across devices.

## Reference patterns in the workspace

### 1. Gemma (`/Users/zcarver/Developer/gemma/prepare_weights.py`)

Pattern: **prepare offline → save single `.pt` bundle → load on device**.

- `prepare_weights.py` reads HF safetensors without needing the `safetensors`
  library (custom JSON header parser), transposes weight matrices to the
  `x @ w` layout, expands 1D RMSNorm weights to `(TILE=32, dim)`, precomputes
  RoPE tables, and saves everything as a flat `.pt` bundle.
- At inference time, the runner loads the bundle once with `torch.load` and
  then pushes each tensor with `ttnn.from_torch` (no per-tensor cache). The
  upfront `.pt` packaging is what makes subsequent runs fast.

Good reference for: offline preprocessing, tile-padding 1D norms, RoPE
precompute, splitting up shards without safetensors dependency.

### 2. LingBot-World DiT (`/Users/zcarver/Developer/lingbot-world/tt/prepare_weights.py`)

Same "bundle once" strategy as Gemma, but uses the `safetensors` library
directly and handles sharded safetensors via the HF index JSON. Preprocessing
transposes all Linear weights, expands norms, and saves `bundle` as a
`torch.save` dict.

Good reference for: sharded safetensors index traversal and per-block weight
organization (useful for DeepSeek's 43 blocks).

### 3. TT-Metal's own llama and gemma demos

`ttnn.as_tensor(cache_file_name=...)` is the blessed path inside `tt-metal`
demos. Look for calls of the form:

```python
cache_name = weight_cache_path / f"layer{i}.wq"
w = ttnn.as_tensor(
    torch_w,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_shape),
    cache_file_name=str(cache_name),
)
```

The flatbuffer cache captures the tilized bits, so first-run cost (dequant,
transpose, tilize) is amortized across every future run.

## Recommended approach for DeepSeek V4-Flash

Do both layers of caching, for two different audiences:

### Layer 1: offline HF → preprocessed torch bundle

Write `prepare_weights.py` modeled on Gemma/LingBot. One-time work:

- Walk `model.safetensors.index.json`.
- For each target tensor:
  - FP8 weight + scale → dequant to bf16 (the `wo_a` path already implemented
    in `inference.py`).
  - FP4 expert weights: keep FP4 nibble-packed (don't dequant — 4x size
    penalty) but still transpose / reorder for the layout we want on device.
  - FP8 scales: keep in their source dtype; they are tiny.
  - RMSNorm weights: expand to `(TILE=32, dim)`.
  - Linear weights: transpose to `x @ w` layout.
  - RoPE: precompute cos/sin tables.
- Save as `{shard_id}.pt` chunks so we can mmap selectively (149 GB won't fit
  in one `torch.load` comfortably; chunk by layer).

This step alone skips safetensors parsing and FP8 dequant on every subsequent
run.

### Layer 2: per-tensor `.tensorbin` cache

At device bring-up, wrap every `ttnn.from_torch` with `ttnn.as_tensor(...,
cache_file_name=...)`. First device run populates the cache; second run is
essentially "mmap the flatbuffers to L1/DRAM."

Pattern:

```python
def load_weight(name, torch_tensor, dtype, mesh_mapper=None):
    return ttnn.as_tensor(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
        cache_file_name=str(cache_dir / name),
    )
```

Put `cache_dir` on a fast local disk on the remote (not NFS), e.g.
`/tmp/deepseek_v4_flash_cache/`.

## Gotchas

- `ttnn.as_tensor` bakes `dtype` and `layout` into the cache filename. Changing
  either invalidates automatically, so there's no risk of stale cache after a
  dtype flip.
- The `dump_tensor_flatbuffer` assertion requires `storage_type == HOST`.
  Don't pass `device=` to the inner `from_torch` call; the outer
  `as_tensor` handles transfer.
- With sharded meshes, `ReplicateTensorToMeshWrapper` is intentionally
  unwrapped during caching so the on-disk blob is device-count agnostic.
- 149 GB on-disk cache is a lot; verify there's enough scratch space before
  the first run populates it.

## Files to consult

- `/Users/zcarver/Developer/tt-metal/ttnn/ttnn/operations/core.py:549-651`
  (authoritative `ttnn.as_tensor` impl, incl. cache naming and load fallback)
- `/Users/zcarver/Developer/gemma/prepare_weights.py` (bundle-once offline pass,
  RoPE precompute, no safetensors dep)
- `/Users/zcarver/Developer/lingbot-world/tt/prepare_weights.py` (sharded
  safetensors via index JSON, per-block organization)
