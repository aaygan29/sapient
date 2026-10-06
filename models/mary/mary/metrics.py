"""Eval metrics for Mary. Reuses the sapient1 family shapes (sapient1/metrics.py).

Exposes hooks the eval harness imports (CONTRACTS.md §6 / guide §5):
  - `vertex_pearson(pred, target)`  — per-vertex r, mean/std/top-10%
  - `parcel_pearson(pred, target, parcel_matrix)` — Schaefer-parcel r
  - `pearson_r(pred, bold)` — harness-facing wrapper returning summary + per-vertex vector
"""

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


def vertex_pearson(
    pred: torch.Tensor,
    target: torch.Tensor,
    responsive_mask: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Vertex-level Pearson r.

    pred, target: (B, T, 20484) — same shape, same time axis.
    Returns mean + std + top-10% mean across vertices (whole-brain).

    When `responsive_mask` (bool/0-1 tensor of shape (V,), typically ISC > 0.05) is
    given, ALSO returns `vertex_pearson_mean_responsive` — the mean r restricted to
    responsive vertices. This is the honest dial: ~90% of vertices are noise (ISC≈0)
    and dilute the whole-brain mean 10:1. Whole-brain stats are always returned.
    """
    p = pred.reshape(-1, pred.shape[-1])
    t = target.reshape(-1, target.shape[-1])
    r = _pearson_along_time(p, t)
    k = max(1, int(round(0.10 * r.numel())))
    out = {
        "vertex_pearson_mean": r.mean().item(),
        "vertex_pearson_std": r.std().item(),
        "vertex_pearson_top10pct_mean": r.topk(k).values.mean().item(),
    }
    if responsive_mask is not None:
        m = responsive_mask
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(np.asarray(m))
        m = m.to(r.device).bool()
        if m.any():
            out["vertex_pearson_mean_responsive"] = r[m].mean().item()
            out["n_responsive"] = int(m.sum().item())
        else:
            out["vertex_pearson_mean_responsive"] = float("nan")
            out["n_responsive"] = 0
    return out


def parcel_pearson(
    pred: torch.Tensor,           # (B, T, 20484)
    target: torch.Tensor,         # (B, T, 20484)
    parcel_matrix: np.ndarray,    # (n_parcels, 20484) averaging matrix (dense or scipy.sparse)
) -> dict[str, float]:
    """Project vertices → Schaefer parcels, then Pearson per parcel."""
    M = parcel_matrix.toarray() if hasattr(parcel_matrix, "toarray") else np.asarray(parcel_matrix)
    M = torch.from_numpy(M).to(pred.device).float()          # (P, V)
    pp = (pred.float() @ M.T).reshape(-1, M.shape[0])
    tp = (target.float() @ M.T).reshape(-1, M.shape[0])
    r = _pearson_along_time(pp, tp)
    return {
        "parcel_pearson_mean": r.mean().item(),
        "parcel_pearson_std": r.std().item(),
    }


def pearson_r(pred: torch.Tensor, bold: torch.Tensor) -> dict[str, object]:
    """Harness-facing per-vertex Pearson r (CONTRACTS.md §6).

    Returns mean / median / top-10% summary plus the full per-vertex r vector (numpy).
    """
    p = pred.reshape(-1, pred.shape[-1])
    t = bold.reshape(-1, bold.shape[-1])
    r = _pearson_along_time(p, t)
    k = max(1, int(round(0.10 * r.numel())))
    return {
        "mean": r.mean().item(),
        "median": r.median().item(),
        "top10pct_mean": r.topk(k).values.mean().item(),
        "per_vertex": r.detach().cpu().numpy(),
    }
