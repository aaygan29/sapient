"""Loss functions. Per spec §4.1 (decision #15).

masked_mse_loss:
  - per-vertex squared error
  - mean over the 20,484 vertices
  - mask-weighted mean over time
"""

from __future__ import annotations

import torch


def masked_mse_loss(
    pred: torch.Tensor,    # (B, T, V)
    target: torch.Tensor,  # (B, T, V)
    mask: torch.Tensor,    # (B, T) — 1 where valid, 0 where padded
) -> torch.Tensor:
    """Per-vertex MSE, mean over vertices, then mask-weighted mean over time."""
    err = ((pred - target) ** 2).mean(dim=-1)  # (B, T)
    denom = mask.sum().clamp(min=1.0)
    return (err * mask).sum() / denom
