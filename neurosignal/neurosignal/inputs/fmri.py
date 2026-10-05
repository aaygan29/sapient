"""Convenience: detect directly from Schaefer-1000 parcellated fMRI (auto-loads the atlas)."""
from __future__ import annotations

from ..types import DetectionResult


def detect_from_schaefer(parcel_activation, *, n_parcels: int = 1000,
                         source: str = "Schaefer-1000 fMRI") -> DetectionResult:
    """parcel_activation: (n_parcels,) or (T, n_parcels) on the Schaefer-1000 atlas. Needs the [atlas] extra."""
    from ..atlases import load_schaefer_network_ids
    from .parcels import detect_from_parcels

    return detect_from_parcels(parcel_activation, load_schaefer_network_ids(n_parcels), source=source)
