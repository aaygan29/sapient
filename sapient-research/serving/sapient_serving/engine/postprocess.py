"""Paper-faithful post-processing: vertices -> Schaefer-1000 parcels -> ROI summaries.

This module encodes the *methodology* the papers describe, independent of the
specific model serving it. Faithfulness anchors (verified against the code +
'Cortex of One' Methods §2.x):

  * Output space: 20,484 fsaverage5 cortical-surface vertices (10,242 / hemisphere),
    predicted at 1 Hz over 100-TR windows. (matches sapient1 model.py n_vertices=20484)
  * Parcellation: Schaefer-1000. We reuse sapient1.vertices_to_parcels / load_parcellation
    when the real package is present, so the projection matrix is THE one built by
    the repo's data/prepare_fmri.py — not a re-derivation. (matches sapient1 inference.py)
  * ROI decomposition: the papers report a Yeo-7 network breakdown
    (Visual, Somatomotor, DorsalAttention, VentralAttention, Limbic, Frontoparietal, Default).

OPEN ITEMS (flagged, not fabricated — to confirm once the repo read resumes):
  * The internal TRIBE/Sapient work elsewhere references "8 ROIs". The papers use Yeo-7.
    The exact canonical ROI set + the Schaefer-1000 -> ROI assignment must come from the
    repo's atlas/eval code (engineering-outline / eval.py). We load that mapping from a
    file (roi_map) rather than hardcode a guess; YEO7_NETWORKS below is the documented default.

  * METRIC SEMANTICS (important honesty point): the papers' headline number is a Pearson
    correlation between predicted and *measured* fMRI. An inference API has no measured
    fMRI for novel client content, so it CANNOT return that correlation. What we return
    per ROI is the model's *predicted activation* (mean over the window). /v1/info states
    this distinction explicitly so a client never mistakes it for a paper-style r value.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..schemas import ParcelHighlight, ParcelSummary

N_VERTICES = 20484
N_PARCELS = 1000

YEO7_NETWORKS = [
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Limbic",
    "Frontoparietal",
    "Default",
]

# Optional: reuse the audited projection from the real package when available.
try:  # pragma: no cover - depends on the private pkg being installed
    from sapient1 import vertices_to_parcels as _sapient_v2p  # type: ignore
    _HAVE_SAPIENT1 = True
except Exception:
    _HAVE_SAPIENT1 = False


def vertices_to_parcels(vertex_pred: np.ndarray, parcel_matrix) -> np.ndarray:
    """(T, 20484) -> (T, 1000). Prefer the repo's audited adapter; fall back to a matmul."""
    if _HAVE_SAPIENT1:
        return np.asarray(_sapient_v2p(vertex_pred, parcel_matrix))
    pm = parcel_matrix.toarray() if hasattr(parcel_matrix, "toarray") else np.asarray(parcel_matrix)
    flat = vertex_pred.reshape(-1, vertex_pred.shape[-1])
    out = flat @ pm.T
    return out.reshape(*vertex_pred.shape[:-1], pm.shape[0])


def parcel_means_over_time(parcels: np.ndarray) -> np.ndarray:
    """(T, 1000) -> (1000,) mean predicted activation per parcel over the window."""
    return np.asarray(parcels, dtype=float).mean(axis=0)


def parcels_to_rois(
    parcel_mean: np.ndarray,
    network_ids: Optional[np.ndarray],
    network_names: list[str] = YEO7_NETWORKS,
) -> dict[str, float]:
    """Aggregate the 1000 parcels into ROI/network means.

    `network_ids` is a (1000,) int array assigning each parcel to a network index.
    It MUST come from the canonical atlas mapping (loaded from file), not invented.
    If it's unavailable we return an empty dict rather than a fabricated breakdown.
    """
    if network_ids is None:
        return {}
    parcel_mean = np.asarray(parcel_mean, dtype=float)
    network_ids = np.asarray(network_ids, dtype=int)
    scores: dict[str, float] = {}
    for idx, name in enumerate(network_names):
        sel = network_ids == idx
        if sel.any():
            scores[name] = float(parcel_mean[sel].mean())
    return scores


def summarize_parcels(
    parcel_mean: np.ndarray,
    *,
    top_k: int = 20,
    include_full: bool = False,
    name_fn=lambda i: f"Parcel_{i + 1:04d}",
) -> ParcelSummary:
    """Coarse, IP-protective summary: stats + top-K highlights; full vector only on opt-in."""
    parcel_mean = np.asarray(parcel_mean, dtype=float)
    order = np.argsort(parcel_mean)[::-1][:top_k]
    top = [
        ParcelHighlight(parcel_id=int(i), name=name_fn(int(i)), value=float(parcel_mean[i]))
        for i in order
    ]
    return ParcelSummary(
        n_parcels=int(parcel_mean.shape[0]),
        mean=float(parcel_mean.mean()),
        std=float(parcel_mean.std()),
        top=top,
        values=[float(x) for x in parcel_mean] if include_full else None,
    )


# Schaefer-2018 7-network label token -> index into YEO7_NETWORKS.
_SCHAEFER_TOKEN_TO_IDX = {
    "Vis": 0, "SomMot": 1, "DorsAttn": 2, "SalVentAttn": 3, "Limbic": 4, "Cont": 5, "Default": 6,
}


def load_schaefer_network_ids(n_parcels: int = N_PARCELS) -> np.ndarray:  # pragma: no cover - real path
    """(n_parcels,) array assigning each Schaefer-1000 parcel to a Yeo-7 network index.

    Requires nilearn (the [real] extra). This is the canonical atlas mapping the real
    buy/sell readout needs; load once and cache on the engine.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(n_rois=n_parcels, yeo_networks=7)
    ids = []
    for label in atlas.labels:
        lab = label.decode() if isinstance(label, bytes) else label
        token = next((t for t in _SCHAEFER_TOKEN_TO_IDX if t in lab), "SalVentAttn")
        ids.append(_SCHAEFER_TOKEN_TO_IDX[token])
    return np.asarray(ids, dtype=int)
