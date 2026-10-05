"""Atlas helpers: map a Schaefer-1000 parcellation to Yeo-7 networks, and aggregate
parcel activation into per-network activation.
"""
from __future__ import annotations

import numpy as np

from .constructs import YEO7_NETWORKS

# Schaefer-2018 7-network label token -> index into YEO7_NETWORKS.
_SCHAEFER_TOKEN_TO_IDX = {
    "Vis": 0, "SomMot": 1, "DorsAttn": 2, "SalVentAttn": 3, "Limbic": 4, "Cont": 5, "Default": 6,
}


def load_schaefer_network_ids(n_parcels: int = 1000) -> np.ndarray:  # pragma: no cover - needs nilearn
    """(n_parcels,) Yeo-7 network index per Schaefer parcel. Requires the [atlas] extra (nilearn)."""
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(n_rois=n_parcels, yeo_networks=7)
    ids = []
    for label in atlas.labels:
        lab = label.decode() if isinstance(label, bytes) else label
        token = next((t for t in _SCHAEFER_TOKEN_TO_IDX if t in lab), "SalVentAttn")
        ids.append(_SCHAEFER_TOKEN_TO_IDX[token])
    return np.asarray(ids, dtype=int)


def parcels_to_networks(parcel_activation, network_ids) -> dict[str, float]:
    """(n_parcels,) or (T, n_parcels) activation + (n_parcels,) Yeo-7 ids -> {network: mean activation}."""
    pa = np.asarray(parcel_activation, dtype=float)
    if pa.ndim == 2:
        pa = pa.mean(axis=0)  # average over time / trials
    elif pa.ndim != 1:
        raise ValueError(f"parcel_activation must be 1-D or 2-D, got shape {pa.shape}")
    ids = np.asarray(network_ids, dtype=int)
    if pa.shape[0] != ids.shape[0]:
        raise ValueError(f"parcel_activation has {pa.shape[0]} parcels but network_ids has {ids.shape[0]}")
    out: dict[str, float] = {}
    for i, name in enumerate(YEO7_NETWORKS):
        sel = ids == i
        if np.any(sel):
            out[name] = float(pa[sel].mean())
    return out
