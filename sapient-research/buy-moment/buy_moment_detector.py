"""
buy_moment_detector.py — neural buy-moment detection from Schaefer-400 parcellated
brain activity produced by Sapient / TRIBE v2.

Pipeline
--------
1.  Input: (T, 400) per-second activation matrix, Schaefer 2018 7-network atlas,
    fsaverage5 surface (LH = cols 0..199, RH = cols 200..399).
2.  For each ROI group, average across the parcels in that group at each
    timepoint, multiply by its signed weight, and sum across groups to get a
    raw buy-moment score `s(t)`.
3.  z-score `s(t)` across the run, then find peaks with z >= Z_THRESHOLD and a
    refractory window of REFRACTORY_SEC seconds. Those are the buy moments.
4.  For each detected peak, emit a per-ROI breakdown showing which signals
    drove the score, so analysts can sanity-check the call against the video.

Run as a script
---------------
    python buy_moment_detector.py /path/to/parcel_activations.npy [out_dir]

Or import:

    from buy_moment_detector import detect_buy_moments
    result = detect_buy_moments(parcel_act)   # parcel_act shape (T, 400)

Honest gap noted (2026-08-25)
------------------------------
The single composite `score_raw`/`score_z` above is a hand-weighted sum over
parcel groups — structurally the same move as collapsing a TRIBE-style
whole-cortex encoder's output into one aggregate "engagement/drive" signal.
arXiv:2607.01400 ("A global predicted-fMRI drive signal from TRIBE does not
predict YouTube replay heatmaps") shows that move fails against a real
behavioral outcome (replay heatmaps) even when the underlying encoder hits
good held-out fMRI Pearson r — the aggregation step, not the encoder, is
where the claim breaks. This module has never been validated against a real
behavioral ground-truth channel; see `validate_against_behavior.py` in this
directory for the synthetic validation harness that stands in for that gate
until a real behavioral channel exists in the pipeline, and
`sapienteval/ACCEPTANCE_CRITERIA.md` §4 for the ship gate this motivates.
Separately, `generalizable_score` / `idiosyncratic_score` below (Genevsky &
Knutson 2018, 2025) address a related but distinct concern: even a validated
composite may only be safe to report at the individual level, not the
aggregate/market level, depending on which ROI groups drive it.
"""

from __future__ import annotations
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# ROI groups
#
# Indices are 0-based COLUMN indices into the (T, 400) Schaefer-400 7-Networks
# matrix.  LH parcels = cols 0..199, RH parcels = cols 200..399.  Groups were
# verified against the actual atlas labels — see header docstring.
#
# Connectome grounding (2026-08-25, see `connectome_validation.py`)
#   Each group below was checked two ways:
#     (1) REAL-DATA: mean within-group pairwise functional connectivity,
#         computed from 8 real adult subjects' Schaefer-400 parcel time series
#         (nilearn's `fetch_development_fmri`, public naturalistic-viewing
#         fMRI, no DUA), beats a null of 1000 random same-size parcel sets at
#         the 100th percentile for all 8 groups (observed r 0.22-0.40 vs.
#         random-null mean r ~0.09). This means every ROI_GROUPS bundle is
#         measurably more internally coherent in real human connectivity data
#         than a same-size random parcel draw — a necessary but NOT sufficient
#         precondition for treating the group as one meaningful channel. It is
#         NOT evidence the group predicts buying behavior.
#     (2) LITERATURE LABEL: cross-tabbed against the Schaefer atlas's own
#         published Yeo-7 network assignment (Schaefer et al. 2018, Cerebral
#         Cortex). All 8 groups sit at 100% purity inside a single Yeo-7
#         network (Limbic, Default, Cont, or SalVentAttn respectively) — i.e.
#         the label-string matching this module was built on did not
#         accidentally straddle unrelated canonical networks.
#   Full numbers + methodology: `connectome_validation.py` and
#   `sapienteval/SCIENCE.md` §12. This upgrades but does not replace the
#   "not anatomically curated" caveat below — real-data coherence and Yeo-7
#   purity are not the same as validation against buying behavior.
#
# Citations
#   Knutson  2007 — "Neural predictors of purchases."  Neuron 53(1):147-156.
#   Plassmann 2007 — "Orbitofrontal cortex encodes willingness to pay in
#                    everyday economic transactions."  J Neurosci 27(37):9984-8.
#   Tom      2007 — "The neural basis of loss aversion in decision-making
#                   under risk."  Science 315(5811):515-518.
#   Reimann  2010 — "Aesthetic package design: a behavioral, neural, and
#                   psychological investigation."  J Consumer Psych 20(4):431-441.
#   Venkatraman 2015 — "Predicting advertising success beyond traditional
#                      measures: new insights from neurophysiological methods
#                      and market response modelling."  J Marketing Res 52(4):436-452.
#   Yao      2021 — "Functional brain networks in marketing: a meta-analysis."
#                   (purchase-decision meta of 25+ studies)
#   Chan     2024 — DMN / social-cognition predictors of video-ad liking.
#   Genevsky & Knutson 2018 — "Neuroforecasting Aggregate Choice."  Reports
#                   that NAcc anticipatory-affect activity generalizes across
#                   people to predict AGGREGATE choice, while mPFC
#                   integrative-value activity does not.
#                   https://stanford.edu/~knutson/nfc/knutson18.pdf
#   Genevsky, Tong & Knutson 2025 — "Neuroforecasting reveals generalizable
#                   components of choice."  PNAS Nexus 4(2):pgaf029.  Two
#                   independent experiments (crowdfunding, video-viewing):
#                   NAcc ICC 0.408-0.441 (p<=0.009, significant) vs mPFC ICC
#                   0.198-0.273 (p>=0.08, NOT significant) — i.e. the
#                   NAcc/early-affective signal generalizes across people and
#                   is safe for aggregate/market-level claims; the
#                   mPFC/integrative signal stays idiosyncratic to the
#                   individual and should NOT be used for aggregate claims.
#                   https://academic.oup.com/pnasnexus/article/4/2/pgaf029/8016018
# ---------------------------------------------------------------------------
#
# Generalizable (aggregate-safe) vs idiosyncratic (individual-only) channels
# ---------------------------------------------------------------------------
# Per Genevsky & Knutson (2018, 2025) above, not every positively-weighted ROI
# group is equally safe to use for an aggregate/market-level claim (e.g. "this
# ad will convert broadly"). Early affective/anticipatory groups (NAcc-proxy
# OFC, amygdala-proxy temporal pole) generalize across people (ICC ~0.41-0.44,
# significant); later integrative/deliberative groups (vmPFC/mPFC, DLPFC) do
# NOT generalize (ICC ~0.20-0.27, non-significant) — they are idiosyncratic to
# the individual. Each group below carries a "channel" tag:
#   "generalizable" — aggregate/market-level claims are supported by the cited
#                      ICC evidence.
#   "idiosyncratic"  — individual-response claims only; do NOT use for
#                      aggregate/market-level claims per the cited ICC null.
#   "unclassified"   — the Genevsky/Knutson generalization dichotomy concerns
#                      positive anticipatory-affect vs integrative-value
#                      signals specifically; it does not speak to the
#                      negative/suppressive groups here, so they are left
#                      unclassified rather than guessed.
# See `generalizable_score` / `idiosyncratic_score` in `detect_buy_moments`.

ROI_GROUPS: dict[str, dict[str, Any]] = {
    # ----- POSITIVE: drive the buy moment up -----
    "Limbic_OFC": {
        "weight": +2.00,
        "channel": "generalizable",
        "parcels": [113, 114, 115, 116, 117,           # LH_Limbic_OFC_1..5
                    318, 319, 320, 321, 322, 323],     # RH_Limbic_OFC_1..6
        "rationale": "Orbitofrontal cortex — cortical proxy for the subcortical "
                     "NAcc reward signal Knutson 2007 reports; Plassmann 2007 "
                     "shows OFC encodes willingness-to-pay magnitude directly; "
                     "Yao 2021 finds OFC is the largest cluster in the purchase-"
                     "decision meta-analysis.  Channel = generalizable: our "
                     "NAcc/anticipatory-affect proxy, shown by Genevsky & "
                     "Knutson (2018, 2025) to generalize across people to "
                     "aggregate choice (ICC ~0.41-0.44, significant) — "
                     "appropriate for aggregate/market-level claims.",
    },
    "Default_PFC_vmPFC": {
        "weight": +1.50,
        "channel": "idiosyncratic",
        "parcels": list(range(165, 189)) + list(range(374, 391)),
        "rationale": "Ventromedial / medial PFC — value integration and "
                     "price-vs-desire matching.  Knutson 2007: mPFC up when "
                     "price feels fair predicts buy.  Tom 2007: vmPFC tracks "
                     "net value (gains − losses).  Yao 2021: vmPFC is the "
                     "single largest cortical cluster across purchase neuroimaging.  "
                     "Channel = idiosyncratic: our mPFC/integrative-value proxy, "
                     "shown by Genevsky & Knutson (2018, 2025) to NOT generalize "
                     "across people (ICC ~0.20-0.27, non-significant) — usable "
                     "only for individual-response claims, never aggregate/"
                     "market-level claims.",
    },
    "Default_Temp_SocCog": {
        "weight": +1.00,
        "channel": "unclassified",
        "parcels": (list(range(148, 158)) + list(range(189, 200)) +
                    list(range(361, 374)) + list(range(391, 400))),
        "rationale": "DMN temporal + dorsomedial PFC + posterior cingulate / "
                     "precuneus — social cognition and narrative absorption.  "
                     "Chan 2024: social-cognition parcels predicted ad liking "
                     "better than reward ROIs alone in video stimuli.  "
                     "Venkatraman 2015: DMN engagement predicts real-world sales lift.  "
                     "Channel = unclassified: outside the anticipatory-affect vs "
                     "integrative-value dichotomy Genevsky & Knutson tested.",
    },
    "Limbic_TempPole": {
        "weight": +0.75,
        "channel": "generalizable",
        "parcels": list(range(118, 126)) + list(range(324, 331)),
        "rationale": "Temporal pole — cortical proxy for amygdala-driven "
                     "emotional arousal.  Venkatraman 2015: amygdala / temporal-"
                     "pole activation predicts real sales.  Reimann 2010: "
                     "aesthetic packaging activates temporal pole.  "
                     "Channel = generalizable: early affective/arousal signal, "
                     "grouped with the NAcc/OFC proxy per Genevsky & Knutson's "
                     "'early affective responses generalize more broadly' "
                     "framing — not itself directly ICC-tested by those papers, "
                     "so treat this specific parcel group's generalizable "
                     "label as an extrapolation from the OFC finding, not a "
                     "separately measured ICC.",
    },
    "Cont_PFCl_DLPFC": {
        "weight": +0.50,
        "channel": "idiosyncratic",
        "parcels": list(range(134, 142)) + list(range(340, 355)),
        "rationale": "Dorsolateral PFC — deliberate valuation and "
                     "willingness-to-pay magnitude (Plassmann 2007).  "
                     "Channel = idiosyncratic: deliberative/control-network "
                     "valuation, grouped with the mPFC integrative proxy per "
                     "Genevsky & Knutson's 'integrative or behavioral "
                     "responses' framing that fails to generalize — again an "
                     "extrapolation from the mPFC finding specifically, not a "
                     "separately measured ICC for DLPFC.",
    },

    # ----- NEGATIVE: suppress / push the score toward no-buy -----
    "SalVentAttn_FrOperIns": {
        "weight": -1.00,
        "channel": "unclassified",
        "parcels": list(range(96, 105)) + list(range(301, 309)),
        "rationale": "Anterior insula — the 'pain of paying.'  Knutson 2007: "
                     "insula ↑ during price exposure predicts purchase "
                     "RESISTANCE; Tom 2007: insula tracks loss aversion.  "
                     "Buy moment = insula SUPPRESSED (negative weight).",
    },
    "Cont_Par_CogLoad": {
        "weight": -0.75,
        "channel": "unclassified",
        "parcels": list(range(126, 132)) + list(range(331, 337)),
        "rationale": "Control-network parietal — cognitive load / confusion.  "
                     "Chan 2024: high cognitive load suppresses purchase "
                     "intent during video ads.",
    },
    "SalVentAttn_ParOper": {
        "weight": -0.50,
        "channel": "unclassified",
        # NB: the RH range below sits in the SalVentAttn TempOccPar subdivision,
        # not ParOper proper, but is still a salience-network posterior region
        # so the aesthetic-aversion interpretation holds.
        "parcels": list(range(91, 95)) + list(range(293, 300)),
        "rationale": "Parietal operculum / posterior insula / posterior salience "
                     "— aesthetic aversion.  Reimann 2010: unaesthetic packaging "
                     "activates this region.",
    },
}


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

Z_THRESHOLD = 1.5            # z-score on the composite to call a peak
REFRACTORY_SEC = 6           # minimum seconds between consecutive peaks
                             # (Knutson-style fMRI HRF is ~6s wide)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

@dataclass
class BuyMoment:
    t_sec: int
    score_z: float
    score_raw: float
    rationale: dict[str, float]      # per-ROI signed contribution to the score


def _group_means(parcel_act: np.ndarray) -> dict[str, np.ndarray]:
    """For each ROI group, return its (T,) within-group mean over time."""
    out: dict[str, np.ndarray] = {}
    T = parcel_act.shape[0]
    for name, spec in ROI_GROUPS.items():
        cols = np.asarray(spec["parcels"], dtype=int)
        if cols.size == 0:
            out[name] = np.zeros(T, dtype=np.float32)
            continue
        out[name] = parcel_act[:, cols].mean(axis=1).astype(np.float32)
    return out


def _compose_score(group_means: dict[str, np.ndarray],
                    names: list[str] | None = None) -> np.ndarray:
    """Signed weighted sum of group means -> raw score time series.

    `names` restricts the sum to a subset of ROI_GROUPS (used for the
    generalizable/idiosyncratic channel scores below). Defaults to all groups
    (the original composite behavior).
    """
    names = names if names is not None else list(group_means.keys())
    parts = [ROI_GROUPS[name]["weight"] * group_means[name] for name in names]
    if not parts:
        T = next(iter(group_means.values())).shape[0]
        return np.zeros(T, dtype=np.float32)
    return np.sum(parts, axis=0)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def detect_buy_moments(
    parcel_act: np.ndarray,
    *,
    z_threshold: float = Z_THRESHOLD,
    refractory_sec: int = REFRACTORY_SEC,
) -> dict[str, Any]:
    """
    Run the detector end-to-end.

    Parameters
    ----------
    parcel_act : np.ndarray of shape (T, 400)
        Schaefer-400 parcellated activations, one row per second.

    Returns
    -------
    dict with keys:
        score_raw   : (T,) raw weighted-sum signal (all ROI groups combined —
                      the original composite, unchanged for existing callers)
        score_z     : (T,) z-scored signal (same composite as above)
        generalizable_score : (T,) z-scored signal from "generalizable"-channel
                      groups only (NAcc/OFC + amygdala/temporal-pole proxies).
                      Appropriate for aggregate/market-level claims per
                      Genevsky & Knutson (2018, 2025) — see ROI_GROUPS header.
        idiosyncratic_score : (T,) z-scored signal from "idiosyncratic"-channel
                      groups only (vmPFC/mPFC + DLPFC proxies). Appropriate
                      ONLY for individual-response claims; do NOT use for
                      aggregate/market-level claims per the same citations.
        channel_config : which ROI groups feed each of the two channel scores
                      above, for auditability.
        group_means : dict[name -> (T,)] per-ROI-group means (raw units)
        buy_moments : list[BuyMoment] sorted by score_z descending (computed
                      against the combined composite, as before)
        config      : the thresholds + weights used
    """
    if parcel_act.ndim != 2 or parcel_act.shape[1] != 400:
        raise ValueError(f"parcel_act must be (T, 400); got {parcel_act.shape}")

    group_means = _group_means(parcel_act)
    score_raw = _compose_score(group_means)
    score_z = _zscore(score_raw)

    # New: generalizable vs idiosyncratic channel scores (additive — does not
    # change score_raw/score_z/group_means/buy_moments for existing callers).
    generalizable_names = [n for n, s in ROI_GROUPS.items() if s["channel"] == "generalizable"]
    idiosyncratic_names = [n for n, s in ROI_GROUPS.items() if s["channel"] == "idiosyncratic"]
    generalizable_raw = _compose_score(group_means, generalizable_names)
    idiosyncratic_raw = _compose_score(group_means, idiosyncratic_names)
    generalizable_score = _zscore(generalizable_raw)
    idiosyncratic_score = _zscore(idiosyncratic_raw)

    # For the per-peak rationale, z-score each group within itself so the
    # contributions are comparable in standardised units.
    group_z = {name: _zscore(series) for name, series in group_means.items()}

    peaks, _ = find_peaks(score_z, height=z_threshold, distance=refractory_sec)

    buy_moments: list[BuyMoment] = []
    for t in peaks.tolist():
        rationale = {
            name: round(float(ROI_GROUPS[name]["weight"]) * float(group_z[name][t]), 4)
            for name in ROI_GROUPS
        }
        buy_moments.append(BuyMoment(
            t_sec=int(t),
            score_z=round(float(score_z[t]), 4),
            score_raw=round(float(score_raw[t]), 4),
            rationale=rationale,
        ))
    buy_moments.sort(key=lambda b: b.score_z, reverse=True)

    return {
        "score_raw": score_raw,
        "score_z": score_z,
        "generalizable_score": generalizable_score,
        "idiosyncratic_score": idiosyncratic_score,
        "channel_config": {
            "generalizable_groups": generalizable_names,
            "idiosyncratic_groups": idiosyncratic_names,
            "unclassified_groups": [n for n, s in ROI_GROUPS.items()
                                     if s["channel"] == "unclassified"],
            "note": "generalizable_score is appropriate for aggregate/"
                    "market-level claims; idiosyncratic_score is appropriate "
                    "only for individual-response claims. See ROI_GROUPS "
                    "header docstring for the Genevsky & Knutson (2018, 2025) "
                    "citations this is based on.",
        },
        "group_means": group_means,
        "buy_moments": buy_moments,
        "config": {
            "z_threshold": z_threshold,
            "refractory_sec": refractory_sec,
            "n_timepoints": int(parcel_act.shape[0]),
            "groups": {
                name: {"weight": spec["weight"], "n_parcels": len(spec["parcels"]),
                       "channel": spec["channel"]}
                for name, spec in ROI_GROUPS.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _serialise(result: dict[str, Any]) -> dict[str, Any]:
    """Make the result dict JSON-friendly."""
    return {
        "config": result["config"],
        "channel_config": result["channel_config"],
        "score_z":   [round(float(x), 5) for x in result["score_z"]],
        "score_raw": [round(float(x), 5) for x in result["score_raw"]],
        "generalizable_score": [round(float(x), 5) for x in result["generalizable_score"]],
        "idiosyncratic_score": [round(float(x), 5) for x in result["idiosyncratic_score"]],
        "group_means": {k: [round(float(x), 5) for x in v]
                        for k, v in result["group_means"].items()},
        "buy_moments": [asdict(b) for b in result["buy_moments"]],
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        print("\nUsage: python buy_moment_detector.py <parcel_activations.npy> [out_dir]",
              file=sys.stderr)
        return 2

    npy_path = Path(argv[1])
    out_dir = Path(argv[2]) if len(argv) > 2 else npy_path.parent / "buy_moments"
    out_dir.mkdir(parents=True, exist_ok=True)

    parcel_act = np.load(npy_path)
    result = detect_buy_moments(parcel_act)

    # JSON: full machine-readable output
    (out_dir / "buy_moments.json").write_text(
        json.dumps(_serialise(result), indent=2))

    # CSV: per-second time series with score + group means
    with open(out_dir / "buy_moments_timeseries.csv", "w", newline="") as f:
        w = csv.writer(f)
        groups = list(result["group_means"].keys())
        w.writerow(["t_sec", "score_raw", "score_z", "generalizable_score",
                    "idiosyncratic_score", *groups])
        T = result["score_z"].shape[0]
        for t in range(T):
            row = [t, round(float(result["score_raw"][t]), 5),
                      round(float(result["score_z"][t]), 5),
                      round(float(result["generalizable_score"][t]), 5),
                      round(float(result["idiosyncratic_score"][t]), 5)]
            row += [round(float(result["group_means"][g][t]), 5) for g in groups]
            w.writerow(row)

    # CSV: just the detected moments, sorted by score
    with open(out_dir / "buy_moments_peaks.csv", "w", newline="") as f:
        w = csv.writer(f)
        groups = list(result["group_means"].keys())
        w.writerow(["t_sec", "score_z", "score_raw", *groups])
        for b in result["buy_moments"]:
            w.writerow([b.t_sec, b.score_z, b.score_raw,
                        *(b.rationale[g] for g in groups)])

    # Console summary
    print(f"Parcel matrix: {parcel_act.shape}, dtype {parcel_act.dtype}")
    print(f"Z-threshold:   {Z_THRESHOLD}  |  Refractory: {REFRACTORY_SEC}s")
    print(f"Buy moments detected: {len(result['buy_moments'])}")
    print()
    if result["buy_moments"]:
        print(f"{'t_sec':>6} {'score_z':>9} {'score_raw':>10}   top contributing ROIs")
        print('-' * 90)
        for b in result["buy_moments"]:
            top = sorted(b.rationale.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            top_s = ", ".join(f"{k}({v:+.2f})" for k, v in top)
            print(f"{b.t_sec:>6} {b.score_z:>9.3f} {b.score_raw:>10.4f}   {top_s}")
    print()
    print(f"Wrote: {out_dir}/buy_moments.json")
    print(f"Wrote: {out_dir}/buy_moments_timeseries.csv")
    print(f"Wrote: {out_dir}/buy_moments_peaks.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
