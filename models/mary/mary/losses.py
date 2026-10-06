"""Composite loss for Mary. Per ORCLE-Nano §5 and CONTRACTS.md §6.

    L = 1.0 · MSE  +  0.5 · NegCorr  +  0.1 · InfoNCE(τ=0.07)

| Term     | Weight | Why |
|----------|--------|-----|
| MSE      | 1.0    | absolute accuracy of predicted vs actual BOLD. |
| NegCorr  | 0.5    | negative mean per-vertex Pearson r along time — directly optimizes the eval metric. |
| InfoNCE  | 0.1    | contrastive across TR timepoints (positive = same-TR pred/target, negatives = other TRs); prevents the degenerate "predict the mean every TR" collapse. τ=0.07. |

`composite_loss(pred, target, mask) -> (scalar, {mse, negcorr, infonce})`.
`mask` is `(B, T)` with 1 where the TR is valid, 0 where padded (matches sapient1's
masked_mse_loss convention).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def masked_mse_loss(
    pred: torch.Tensor,    # (B, T, V)
    target: torch.Tensor,  # (B, T, V)
    mask: torch.Tensor,    # (B, T) — 1 valid, 0 padded
    vertex_weight: Optional[torch.Tensor] = None,   # (V,) — per-vertex weight
) -> torch.Tensor:
    """Per-vertex MSE, mean over vertices, then mask-weighted mean over time.

    When `vertex_weight` (shape (V,)) is given, the vertex reduction is a weighted
    mean (same responsive-vertex weighting as `negcorr_loss`); None keeps the
    original plain mean over vertices.
    """
    sq = (pred - target) ** 2                         # (B, T, V)
    if vertex_weight is None:
        err = sq.mean(dim=-1)                         # (B, T)
    else:
        w = vertex_weight.to(sq.dtype).to(sq.device).clamp(min=0.0)     # (V,)
        err = (sq * w).sum(dim=-1) / w.sum().clamp(min=1e-8)            # (B, T)
    denom = mask.sum().clamp(min=1.0)
    return (err * mask).sum() / denom


def negcorr_loss(
    pred: torch.Tensor,    # (B, T, V)
    target: torch.Tensor,  # (B, T, V)
    mask: torch.Tensor,    # (B, T)
    vertex_weight: Optional[torch.Tensor] = None,   # (V,) — per-vertex weight
) -> torch.Tensor:
    """Negative mean per-vertex Pearson r along the time axis (over valid TRs).

    Computed per (batch, vertex) over time, then averaged over vertices and batch.
    Zero-variance vertices contribute r=0 (nan_to_num).

    When `vertex_weight` (shape (V,)) is given, the vertex reduction becomes a
    WEIGHTED mean instead of a plain mean. This is the responsive-vertex fix: ~90%
    of the 20,484 fsaverage5 vertices are noise (ISC≈0) and dilute the signal 10:1,
    so we weight each vertex by its ISC noise-ceiling (clamped ≥0). When None the
    behavior is identical to the original plain mean.
    """
    p = pred.float()
    t = target.float()
    m = mask.float().unsqueeze(-1)                   # (B, T, 1)
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)    # (B, 1, 1)

    p_mean = (p * m).sum(dim=1, keepdim=True) / n
    t_mean = (t * m).sum(dim=1, keepdim=True) / n
    pc = (p - p_mean) * m
    tc = (t - t_mean) * m
    num = (pc * tc).sum(dim=1)                       # (B, V)
    # Numerically stable: add eps to EACH variance before the product (avoids the
    # exploding-gradient singularity when predictions are near-flat), then clamp r.
    var_p = (pc ** 2).sum(dim=1) + 1e-4
    var_t = (tc ** 2).sum(dim=1) + 1e-4
    r = (num / torch.sqrt(var_p * var_t)).clamp(-1.0, 1.0)
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)            # (B, V)
    if vertex_weight is None:
        return -r.mean()
    # Weighted mean over vertices (broadcast over batch), then mean over batch.
    w = vertex_weight.to(r.dtype).to(r.device).clamp(min=0.0)           # (V,)
    r_v = r.mean(dim=0)                              # (V,) — mean over batch first
    return -((r_v * w).sum() / w.sum().clamp(min=1e-8))


def infonce_loss(
    pred: torch.Tensor,    # (B, T, V)
    target: torch.Tensor,  # (B, T, V)
    mask: torch.Tensor,    # (B, T)
    temperature: float = 0.07,
) -> torch.Tensor:
    """Contrastive loss across TR timepoints (within each sample).

    For each sample, build a (T, T) similarity matrix between predicted and target TR
    vectors (cosine / temperature). The positive for predicted TR i is target TR i;
    negatives are all other target TRs. Symmetric InfoNCE, averaged over valid TRs and
    over the batch. Padded TRs are masked out of both rows and columns.
    """
    B, T, V = pred.shape
    p = F.normalize(pred.float(), dim=-1)            # (B, T, V)
    t = F.normalize(target.float(), dim=-1)
    logits = torch.bmm(p, t.transpose(1, 2)) / temperature   # (B, T, T)

    valid = mask.bool()                              # (B, T)
    # Mask out padded TRs as candidate keys (columns) by setting to -inf.
    col_mask = (~valid).unsqueeze(1).expand(B, T, T)
    logits = logits.masked_fill(col_mask, float("-inf"))

    labels = torch.arange(T, device=pred.device).unsqueeze(0).expand(B, T)  # (B, T)
    # Only compute loss on valid query rows.
    loss_pt = F.cross_entropy(
        logits.reshape(B * T, T),
        labels.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    loss_pt = torch.nan_to_num(loss_pt, nan=0.0, posinf=0.0, neginf=0.0)
    denom = valid.float().sum().clamp(min=1.0)
    row_loss = (loss_pt * valid.float()).sum() / denom

    # Symmetric: also target→pred direction.
    logits_t = torch.bmm(t, p.transpose(1, 2)) / temperature
    logits_t = logits_t.masked_fill(col_mask, float("-inf"))
    loss_tp = F.cross_entropy(
        logits_t.reshape(B * T, T),
        labels.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    loss_tp = torch.nan_to_num(loss_tp, nan=0.0, posinf=0.0, neginf=0.0)
    col_loss = (loss_tp * valid.float()).sum() / denom

    return 0.5 * (row_loss + col_loss)


def composite_loss(
    pred: torch.Tensor,             # (B, T, V)
    target: torch.Tensor,           # (B, T, V)
    mask: Optional[torch.Tensor] = None,   # (B, T); defaults to all-valid
    mse_weight: float = 1.0,
    negcorr_weight: float = 0.5,
    infonce_weight: float = 0.1,
    infonce_temperature: float = 0.07,
    vertex_weight: Optional[torch.Tensor] = None,   # (V,) — per-vertex weight (ISC)
) -> tuple[torch.Tensor, dict[str, float]]:
    """Composite loss = 1.0·MSE + 0.5·NegCorr + 0.1·InfoNCE(τ=0.07).

    Returns `(scalar_loss, {"mse": ..., "negcorr": ..., "infonce": ...})`. The dict
    holds the **unweighted** term values (floats) for logging.

    `vertex_weight` (shape (V,)) is the responsive-vertex fix: when provided it
    re-weights the MSE and NegCorr vertex reductions toward responsive (high-ISC)
    vertices so the ~90% noise vertices stop diluting the signal. InfoNCE is left
    unweighted (it contrasts whole-TR vectors, not per-vertex). None = original
    whole-brain behavior.
    """
    if mask is None:
        mask = torch.ones(pred.shape[0], pred.shape[1], device=pred.device)

    mse = masked_mse_loss(pred, target, mask, vertex_weight)
    negcorr = negcorr_loss(pred, target, mask, vertex_weight)
    infonce = infonce_loss(pred, target, mask, infonce_temperature)

    total = mse_weight * mse + negcorr_weight * negcorr + infonce_weight * infonce
    terms = {
        "mse": mse.item(),
        "negcorr": negcorr.item(),
        "infonce": infonce.item(),
    }
    return total, terms
