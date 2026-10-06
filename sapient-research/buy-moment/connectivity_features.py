"""
connectivity_features.py — CPM-style (connectome-based predictive modeling)
rolling-window functional-connectivity fingerprint, computed as a pure
post-processing step on the same (T, 400) Schaefer-400 parcel-activation
matrix `detect_buy_moments` already consumes. No retraining, no change to
the base encoder or `05_finetune_sapient.py`.

Motivation / citations
-----------------------
`buy_moment_detector.py`'s composite score is built entirely from per-timepoint
REGIONAL activation amplitude (a weighted mean over parcels at each second).
Connectome-based predictive modeling shows individual differences in behavior
are often better captured by EDGE-level functional connectivity — covariance
between distributed parcel pairs over a time window — than by activation
amplitude alone:

    Finn et al. 2015 — "Functional connectome fingerprinting: identifying
        individuals using patterns of brain connectivity." Nat Neurosci
        18(11):1664-1671.
    Shen et al. 2017 — "Using connectome-based predictive modeling to predict
        individual behavior from brain connectivity." Nat Protoc
        12(3):506-518. https://www.nature.com/articles/nprot.2016.178

This module adds a compact "connectivity fingerprint" feature: a rolling-
window parcel-by-parcel Pearson correlation matrix, restricted to the parcel
indices already defined in `buy_moment_detector.ROI_GROUPS` (reusing the same
literature-motivated ROI selection rather than computing all 400x400 = 160k
pairs, which is both unnecessary — CPM's edge selection step exists precisely
because most edges are uninformative — and slow for a per-window operation).

Honest gap noted
-----------------
This feature is NOT wired into `detect_buy_moments`'s composite score, and it
has NOT been validated against any real behavioral outcome. The validation
below is synthetic-only: we generate synthetic (T, 400) parcel data with a
KNOWN, INJECTED covariance structure between two ROI groups and confirm the
feature recovers that injected correlation within a stated tolerance. This
establishes only that the *arithmetic* is correct — it says nothing about
whether connectivity-fingerprint edges here actually predict any real
purchase/engagement outcome. Real-outcome validation (e.g. against NeuroEngage
engagement labels, if/when a real behavioral channel exists in the pipeline —
see `validate_against_behavior.py`) is future work.

Run as a script (synthetic self-test)
--------------------------------------
    python connectivity_features.py
"""

from __future__ import annotations
from typing import Any

import numpy as np

from buy_moment_detector import ROI_GROUPS


def _group_parcel_index_map() -> dict[str, np.ndarray]:
    """ROI group name -> sorted unique parcel column indices."""
    return {name: np.asarray(sorted(set(spec["parcels"])), dtype=int)
            for name, spec in ROI_GROUPS.items()}


def rolling_connectivity_fingerprint(
    parcel_act: np.ndarray,
    *,
    window_sec: int = 15,
    step_sec: int = 5,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute a rolling-window parcel-by-parcel correlation ("connectivity
    fingerprint") over a literature-motivated subset of parcels.

    Parameters
    ----------
    parcel_act : np.ndarray of shape (T, 400)
        Same Schaefer-400 parcellated input `detect_buy_moments` takes.
    window_sec : int
        Window length in seconds (timepoints) for each correlation matrix.
    step_sec : int
        Stride between consecutive window starts.
    groups : list[str] | None
        Which ROI_GROUPS to include parcels from. Defaults to all groups
        defined in `buy_moment_detector.ROI_GROUPS` (still far fewer than the
        full 400 parcels — see module docstring for why we don't do 400x400).

    Returns
    -------
    dict with keys:
        parcel_index   : (P,) the parcel column indices used, sorted
        parcel_group   : (P,) which ROI group each of the P parcels came from
        window_starts  : (W,) start timepoint of each window
        window_sec     : int, echoed back
        fingerprints   : (W, P, P) per-window Pearson correlation matrices
                         (NaN-safe: windows with zero-variance parcels get 0
                         for those entries rather than NaN)
    """
    if parcel_act.ndim != 2 or parcel_act.shape[1] != 400:
        raise ValueError(f"parcel_act must be (T, 400); got {parcel_act.shape}")

    T = parcel_act.shape[0]
    group_map = _group_parcel_index_map()
    use_groups = groups if groups is not None else list(group_map.keys())

    parcel_index: list[int] = []
    parcel_group: list[str] = []
    for g in use_groups:
        for p in group_map[g]:
            parcel_index.append(int(p))
            parcel_group.append(g)
    # De-duplicate while preserving first-seen group label, then sort by index
    # so the resulting matrix has a stable, reproducible parcel ordering.
    seen: dict[int, str] = {}
    for p, g in zip(parcel_index, parcel_group):
        seen.setdefault(p, g)
    ordered = sorted(seen.items())
    parcel_index_arr = np.asarray([p for p, _ in ordered], dtype=int)
    parcel_group_arr = np.asarray([g for _, g in ordered], dtype=object)

    if window_sec > T:
        raise ValueError(f"window_sec={window_sec} exceeds n_timepoints={T}")

    window_starts = list(range(0, T - window_sec + 1, step_sec))
    P = parcel_index_arr.size
    fingerprints = np.zeros((len(window_starts), P, P), dtype=np.float32)

    sub = parcel_act[:, parcel_index_arr]
    for wi, start in enumerate(window_starts):
        seg = sub[start:start + window_sec, :]        # (window_sec, P)
        std = seg.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(seg, rowvar=False)      # (P, P)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        # Force exact-zero-variance parcels to read 0 rather than whatever
        # corrcoef's degenerate-division artifact produces.
        zero_var = std == 0
        if zero_var.any():
            corr[zero_var, :] = 0.0
            corr[:, zero_var] = 0.0
        fingerprints[wi] = corr.astype(np.float32)

    return {
        "parcel_index": parcel_index_arr,
        "parcel_group": parcel_group_arr,
        "window_starts": np.asarray(window_starts, dtype=int),
        "window_sec": window_sec,
        "fingerprints": fingerprints,
    }


def mean_between_group_connectivity(
    fingerprint_result: dict[str, Any],
    group_a: str,
    group_b: str,
) -> np.ndarray:
    """
    Compact scalar summary per window: mean pairwise correlation between all
    parcels in `group_a` and all parcels in `group_b` (off-diagonal block
    mean). This is the kind of single-number "connectivity fingerprint"
    feature a downstream model would actually consume, e.g.
    OFC<->vmPFC coupling strength per window.
    """
    parcel_group = fingerprint_result["parcel_group"]
    fp = fingerprint_result["fingerprints"]
    idx_a = np.where(parcel_group == group_a)[0]
    idx_b = np.where(parcel_group == group_b)[0]
    if idx_a.size == 0 or idx_b.size == 0:
        raise ValueError(f"group not found in fingerprint result: "
                          f"{group_a if idx_a.size == 0 else group_b}")
    block = fp[:, idx_a[:, None], idx_b[None, :]]   # (W, |A|, |B|)
    return block.reshape(block.shape[0], -1).mean(axis=1)


# ---------------------------------------------------------------------------
# Synthetic self-test: inject a known covariance structure between two ROI
# groups and confirm the feature recovers it within tolerance.
# ---------------------------------------------------------------------------

def _synthetic_two_group_data(
    T: int = 300,
    injected_r: float = 0.7,
    group_a: str = "Limbic_OFC",
    group_b: str = "Default_PFC_vmPFC",
    noise_sd: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a synthetic (T, 400) parcel matrix where parcels in `group_a` and
    `group_b` share a common latent signal scaled to produce a KNOWN target
    Pearson correlation `injected_r` between any parcel in A and any parcel
    in B, plus independent noise everywhere else. This is disclosed synthetic
    data for arithmetic validation only — see module Honest-gap-noted section.
    """
    rng = np.random.default_rng(seed)
    parcel_act = rng.normal(0, noise_sd, size=(T, 400)).astype(np.float32)

    latent = rng.normal(0, 1, size=T).astype(np.float32)
    group_map = _group_parcel_index_map()
    cols_a = group_map[group_a]
    cols_b = group_map[group_b]

    # For X = a*latent + sqrt(1-a^2)*noise (unit-variance noise), corr(X, latent) = a.
    # We want corr(X_a, X_b) = injected_r when both share the same latent with
    # loading a: corr = a^2. So a = sqrt(injected_r) for injected_r >= 0.
    a = float(np.sqrt(max(injected_r, 0.0)))
    b = float(np.sqrt(max(1.0 - a * a, 0.0)))
    for cols in (cols_a, cols_b):
        noise = rng.normal(0, 1, size=(T, cols.size)).astype(np.float32)
        parcel_act[:, cols] = a * latent[:, None] + b * noise

    return parcel_act


def _run_synthetic_validation() -> bool:
    injected_r = 0.7
    tol = 0.12  # generous tolerance: window-level Pearson r on finite windows
                # is noisy even when the population-level target is exact.
    group_a, group_b = "Limbic_OFC", "Default_PFC_vmPFC"

    parcel_act = _synthetic_two_group_data(injected_r=injected_r,
                                            group_a=group_a, group_b=group_b)
    fp = rolling_connectivity_fingerprint(parcel_act, window_sec=30, step_sec=10)
    recovered = mean_between_group_connectivity(fp, group_a, group_b)
    recovered_mean = float(np.mean(recovered))

    # Negative control: a group pair that shares no latent signal in the
    # synthetic data should recover ~0 connectivity, not the injected value.
    control_group = "SalVentAttn_FrOperIns"
    recovered_control = mean_between_group_connectivity(fp, group_a, control_group)
    recovered_control_mean = float(np.mean(recovered_control))

    ok_signal = abs(recovered_mean - injected_r) <= tol
    ok_control = abs(recovered_control_mean) <= tol

    print("Connectivity-fingerprint synthetic validation")
    print("-" * 60)
    print(f"Injected  {group_a} <-> {group_b} correlation: {injected_r:+.3f}")
    print(f"Recovered {group_a} <-> {group_b} correlation: {recovered_mean:+.3f} "
          f"(tolerance ±{tol})  -> {'PASS' if ok_signal else 'FAIL'}")
    print(f"Negative control {group_a} <-> {control_group} "
          f"(no injected signal): {recovered_control_mean:+.3f}  "
          f"-> {'PASS' if ok_control else 'FAIL'}")
    return ok_signal and ok_control


if __name__ == "__main__":
    import sys
    passed = _run_synthetic_validation()
    sys.exit(0 if passed else 1)
