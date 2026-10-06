"""Eval metrics. Per spec §5.2."""

from __future__ import annotations

import numpy as np
import torch


def _pearson_along_time(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-feature Pearson r along axis 0.

    pred, target: (N, F) — N timepoints, F features (vertices or parcels).
    Returns: (F,) Pearson r per feature. NaNs (zero-variance features) → 0.
    """
    p = pred.float()
    t = target.float()
    p = p - p.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)
    num = (p * t).sum(dim=0)
    denom = torch.sqrt((p ** 2).sum(dim=0) * (t ** 2).sum(dim=0)).clamp(min=1e-8)
    r = num / denom
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def vertex_pearson(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Vertex-level Pearson r.

    pred, target: (B, T, 20484) — same shape, same time axis.
    Returns mean + std + top-10% mean across vertices.
    """
    p = pred.reshape(-1, pred.shape[-1])
    t = target.reshape(-1, target.shape[-1])
    r = _pearson_along_time(p, t)
    k = max(1, int(round(0.10 * r.numel())))
    return {
        "vertex_pearson_mean": r.mean().item(),
        "vertex_pearson_std":  r.std().item(),
        "vertex_pearson_top10pct_mean": r.topk(k).values.mean().item(),
    }


def parcel_pearson(
    pred: torch.Tensor,           # (B, T, 20484)
    target: torch.Tensor,         # (B, T, 20484)
    parcel_matrix: np.ndarray,    # (n_parcels, 20484) sparse averaging matrix
) -> dict[str, float]:
    """Project vertices → Schaefer parcels, then Pearson per parcel."""
    # parcel_matrix is scipy.sparse; densify for the (small) parcel dim.
    M = torch.from_numpy(parcel_matrix.toarray()).to(pred.device).float()  # (P, V)
    # (B, T, V) → (B, T, P) via einsum / matmul.
    pp = pred.float() @ M.T
    tp = target.float() @ M.T
    pp = pp.reshape(-1, pp.shape[-1])
    tp = tp.reshape(-1, tp.shape[-1])
    r = _pearson_along_time(pp, tp)
    return {
        "parcel_pearson_mean": r.mean().item(),
        "parcel_pearson_std":  r.std().item(),
    }
