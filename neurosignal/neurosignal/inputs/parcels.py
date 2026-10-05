"""Primary measured/predicted-fMRI entry: parcellated activation -> detection."""
from __future__ import annotations

from ..atlases import parcels_to_networks
from ..detect import detect_from_networks
from ..types import DetectionResult


def detect_from_parcels(parcel_activation, network_ids, *, source: str = "parcellated fMRI") -> DetectionResult:
    """Detect constructs + buy/sell from parcel-level activation.

    parcel_activation : (n_parcels,) or (T, n_parcels) — e.g. Schaefer-1000 betas / z-stats.
    network_ids       : (n_parcels,) Yeo-7 index per parcel (see atlases.load_schaefer_network_ids).
    """
    return detect_from_networks(parcels_to_networks(parcel_activation, network_ids), source=source)
