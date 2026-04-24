"""DeviceMoEGate: on-device V4-Flash MoE gate (standalone v1).

Full on-device pipeline for one gate:
  1. tt-lang `matmul_softmax`  ->  row-softmax(x @ w^T)           (fused)
  2. ttnn.add(bias)            ->  biased scores
  3. ttnn.topk                 ->  indices (biased top-k)
  4. ttnn.gather               ->  original-softmax scores at those indices
  5. ttnn.multiply(route_scale)

Matches the non-hash branch of `Gate.forward` in `inference.py:840`; the
gather pulls from the un-biased softmax, not the biased scores (this is
what PyTorch does too: `original_scores.gather(-1, indices)`).

v1 is single-device (one card). No mesh sharding. No hash-gate (raises).
Primary thread integrates this into `inference.py` when the attention
work is settled.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Optional, Tuple

import torch
import ttnn

# tt-lang kernel lives one level up in tt-lang-kernels/.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tt-lang-kernels"))

from matmul_softmax import TILE, solve as matmul_softmax_solve
from harness import scaler_tile


class DeviceMoEGate:
    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        topk: int,
        route_scale: float,
    ):
        """
        device: ttnn device (from ttnn.open_device).
        weight: [n_experts, dim] torch tensor (nn.Linear order).
        bias: [n_experts] torch tensor, or None.
        topk: number of experts to route to.
        route_scale: post-gather multiplier.
        """
        if weight.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"gate weight must be bf16/fp32, got {weight.dtype}")
        n_experts, dim = weight.shape
        if dim % TILE or n_experts % TILE:
            raise ValueError(
                f"gate weight dims must be tile-aligned ({TILE}); got {weight.shape}"
            )
        self.device = device
        self.n_experts = n_experts
        self.dim = dim
        self.topk = topk
        self.route_scale = float(route_scale)

        # Weight stored transposed for matmul order: x[M, dim] @ w[dim, n_experts].
        w_kn = weight.to(torch.bfloat16).transpose(0, 1).contiguous()
        common = dict(
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w_tt = ttnn.from_torch(w_kn, **common)

        # 32x32 scaler tile of 1.0s for the tt-lang softmax reductions.
        self.sc_tt = ttnn.from_torch(scaler_tile(dtype=torch.bfloat16), **common)

        # Bias reshaped to [1, n_experts] so ttnn.add broadcasts across rows.
        if bias is not None:
            b_row = bias.to(torch.bfloat16).view(1, n_experts).contiguous()
            self.bias_tt = ttnn.from_torch(b_row, **common)
        else:
            self.bias_tt = None

    def forward(
        self,
        x: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        x: [M, dim] torch tensor. M in [1, TILE] (v1: one tile-row only).
        return_scores: if True also returns pre-bias softmax scores [M, n_experts]
                       (test hook; not used by MoE.forward).
        Returns (weights, indices) or (scores, weights, indices).
          weights: [M, topk] float32
          indices: [M, topk] int64
          scores:  [M, n_experts] float32
        """
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must be [M, dim={self.dim}], got {tuple(x.shape)}")
        M = x.shape[0]
        if M < 1 or M > TILE:
            raise ValueError(f"v1 requires 1 <= M <= {TILE}, got M={M}")

        # Pad x to a full tile row.
        x_padded = torch.zeros(TILE, self.dim, dtype=torch.bfloat16)
        x_padded[:M] = x.to(torch.bfloat16)

        common = dict(
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_tt = ttnn.from_torch(x_padded.contiguous(), **common)
        scores_tt = ttnn.from_torch(
            torch.zeros(TILE, self.n_experts, dtype=torch.bfloat16), **common
        )

        # 1. fused matmul + row-softmax
        matmul_softmax_solve(x_tt, self.w_tt, self.sc_tt, scores_tt)

        # 2. bias add
        if self.bias_tt is not None:
            biased_tt = ttnn.add(scores_tt, self.bias_tt)
        else:
            biased_tt = scores_tt

        # 3. topk on biased scores (values discarded; only indices matter)
        _, indices_tt = ttnn.topk(
            biased_tt, k=self.topk, dim=-1, largest=True, sorted=True
        )

        # 4. gather ORIGINAL (un-biased) softmax at those indices
        gathered_tt = ttnn.gather(scores_tt, dim=-1, index=indices_tt)

        # 5. scale
        weights_tt = ttnn.multiply(gathered_tt, self.route_scale)

        weights = ttnn.to_torch(weights_tt)[:M].to(torch.float32)
        indices = ttnn.to_torch(indices_tt)[:M].to(torch.int64)
        if return_scores:
            scores = ttnn.to_torch(scores_tt)[:M].to(torch.float32)
            return scores, weights, indices
        return weights, indices

    def __call__(self, x, return_scores: bool = False):
        return self.forward(x, return_scores=return_scores)
