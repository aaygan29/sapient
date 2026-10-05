"""Core detection: brain-network activations -> per-construct scores + buy/sell signal.

Pure NumPy, deterministic, input-validated. The approach-avoidance composite follows
Knutson et al. (2007): reward/value drives buying, conflict/insula drives selling.
Constructs whose regions are absent from the input are reported as not-covered (never
silently fabricated), and `coverage` + `confidence` reflect that.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .constructs import CONSTRUCTS, YEO7_NETWORKS
from .types import ConstructActivation, DetectionResult

BUY_THRESHOLD = 60.0
SELL_THRESHOLD = 40.0


def _validate(network_activation: Mapping[str, float]) -> dict[str, float]:
    if not network_activation:
        raise ValueError("network_activation is empty; provide activation for at least one Yeo-7 network.")
    clean: dict[str, float] = {}
    for k, v in network_activation.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"activation for {k!r} is not a number: {v!r}")
        if not math.isfinite(fv):
            raise ValueError(f"activation for {k!r} is not finite: {fv}")
        clean[str(k)] = fv
    unknown = sorted(set(clean) - set(YEO7_NETWORKS))
    if unknown:
        raise ValueError(f"unknown network label(s) {unknown}; expected a subset of {list(YEO7_NETWORKS)}")
    return clean


def _normalize(values: dict[str, float]) -> dict[str, float]:
    """Min-max across the provided networks so relative activation is comparable."""
    arr = np.array(list(values.values()), dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def detect_from_networks(
    network_activation: Mapping[str, float],
    *,
    source: str = "Yeo-7 network activations",
    normalize: bool = True,
) -> DetectionResult:
    """Detect construct activations + buy/sell signal from Yeo-7 network activations.

    `network_activation` maps Yeo-7 network names (a subset of constructs.YEO7_NETWORKS)
    to activation values. With `normalize=True` (default) they are min-max normalized
    internally (single-stimulus use). Pass `normalize=False` when the caller has already
    scaled activations on a common axis (e.g. globally across a timeline) so values stay
    comparable across frames.
    """
    clean = _validate(network_activation)
    norm = _normalize(clean) if normalize else {k: float(np.clip(v, 0.0, 1.0)) for k, v in clean.items()}

    constructs: list[ConstructActivation] = []
    approach_num = approach_den = avoid_num = avoid_den = 0.0
    conflict = 0.5
    covered = 0

    for c in CONSTRUCTS:
        present = [n for n in c.yeo7_networks if n in norm]
        is_covered = len(present) > 0
        score = float(np.mean([norm[n] for n in present])) * 100.0 if is_covered else 0.0
        if is_covered:
            covered += 1
            frac = score / 100.0
            w = abs(c.polarity)
            if c.polarity > 0:
                approach_num += w * frac
                approach_den += w
            else:
                avoid_num += w * frac
                avoid_den += w
            if c.key == "conflict_risk":
                conflict = frac
        constructs.append(ConstructActivation(
            key=c.key, label=c.label, score=round(score, 1),
            covered=is_covered, cortical_proxy=(c.subcortical and is_covered),
            regions=list(c.regions), citations=list(c.citations),
        ))

    approach = approach_num / approach_den if approach_den else 0.5
    avoid = avoid_num / avoid_den if avoid_den else 0.5
    buy_sell = float(np.clip(100.0 * (0.5 + (approach - 0.5) - 0.5 * (avoid - 0.5)), 0.0, 100.0))
    recommendation = "Buy" if buy_sell >= BUY_THRESHOLD else ("Sell" if buy_sell <= SELL_THRESHOLD else "Hold")

    coverage = covered / len(CONSTRUCTS)
    confidence = float(np.clip((0.5 + 0.5 * coverage) - 0.3 * conflict, 0.2, 0.95))

    notes: list[str] = []
    missing = [c.label for c in CONSTRUCTS if not any(n in norm for n in c.yeo7_networks)]
    if missing:
        notes.append("Constructs NOT covered by this input (regions absent): " + ", ".join(missing) + ".")
    if any(c.subcortical and any(n in norm for n in c.yeo7_networks) for c in CONSTRUCTS):
        notes.append(
            "Subcortical regions (nucleus accumbens, amygdala, hippocampus) are approximated by "
            "cortical proxies; for direct subcortical detection use a volumetric subcortical atlas."
        )

    return DetectionResult(
        constructs=constructs, buy_sell_score=round(buy_sell, 1), recommendation=recommendation,
        confidence=round(confidence, 2), coverage=round(coverage, 2), source=source, notes=notes,
    )
