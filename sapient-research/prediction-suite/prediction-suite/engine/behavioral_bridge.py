"""
behavioral_bridge.py — the brain → behavior layer, calibrated on the published
neuroforecasting literature.

WHY THIS EXISTS
---------------
`neuroforecasting.py` fits a data-driven map from a construct score to an outcome.
That is correct but agnostic: it does not encode *which* neural signals are known
to generalize from a small scanned sample to a whole market. Twenty years of
neuroforecasting work says the answer is specific and non-obvious:

  - Anticipatory-AFFECT signals (NAcc, MPFC) forecast AGGREGATE choice, sometimes
    better than the person's own behavior or self-report.
      Knutson et al., Neural predictors of purchases, Neuron 2007.
      Falk et al., From neural responses to population behavior, Psych Sci 2012.
  - The affective component generalizes ACROSS people; the integrative component is
    more idiosyncratic. So for market-level forecasting you weight affect, not
    deliberation.
      Genevsky & Knutson; Knutson & Genevsky, Neuroforecasting Aggregate Choice,
      Curr Dir Psychol Sci 2018.
  - Brain forecasts stay valid when the scanned sample is NOT demographically
    representative, where behavioral forecasts degrade — a real advantage of the
    neural signal.
      Genevsky et al., Neuroforecasting reveals generalizable components of choice,
      PNAS Nexus 2025.
  - Regions contribute with SIGNED direction: NAcc / medial OFC / amygdala / dmPFC
    push sales up; dlPFC and insula push them down.
      Kühn et al., Multiple "buy buttons" in the brain, NeuroImage 2016.
  - Neural + self-report COMBINED beats either alone, and the neural term only helps
    for content with real affective argument (content-dependence).
      Falk et al., Functional brain imaging predicts public health campaign success,
      SCAN 2016 (combined R^2 up to 0.65).

This module encodes those findings as a calibrated, auditable forecasting layer.
It is deliberately conservative: signed priors from the literature, a self-report
ensemble, and a representativeness-robustness read-out — and it defers the
"is this signal even affective or just sensory?" question to the specificity gate
in `mary_readouts.py`. It never claims a market prediction the neural signal cannot
carry.

INPUT CONTRACT
--------------
Everything is model-agnostic. You pass:
  - construct_readouts: dict of grounded construct scores from mary_readouts.py
    (value / reward / emotion / arousal / attention / memory), already global-signal
    and confound residualized.
  - optionally, region_means: signed ROI activations on fsaverage5 (NAcc proxy,
    mOFC, amygdala, dmPFC, dlPFC, insula) if you have vertexwise maps.
  - optionally, self_report: the audience's stated preference (0-100).
Output is a single calibrated market-forecast index with an uncertainty band and a
plain-language provenance string saying which evidence carried the prediction.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Literature-derived priors.
# ---------------------------------------------------------------------------
# Affect-Integration-Motivation (AIM) grouping. Weights are the RELATIVE prior
# emphasis for AGGREGATE (market) forecasting, where affect generalizes best.
# These are priors, not fitted coefficients: they set the starting direction and
# emphasis and are then reconciled against any in-sample calibration data.
AIM_CONSTRUCT_PRIORS = {
    # affect: generalizes across individuals -> highest aggregate weight
    "value":     +1.00,   # NAcc/vmPFC value signal; strongest single buy-signal (Knutson 2007)
    "reward":    +0.90,
    "arousal":   +0.55,   # amygdala/insula arousal; positive but noisier
    "emotion":   +0.55,
    # attention/memory: necessary for encoding, weaker direct aggregate weight
    "attention": +0.40,   # engagement/ISC (Dmochowski 2014) — supports, not drives
    "memory":    +0.35,   # MTL encoding -> recall/commercial success (Boksem & Smidts 2015)
}

# Signed regional contributions to point-of-sale behavior (Kühn et al. 2016).
# Positive regions raise predicted sales; negative regions lower it.
SIGNED_REGION_PRIORS = {
    "NAcc":     +1.00,   # nucleus accumbens — approach/anticipatory reward
    "mOFC":     +0.80,   # medial orbitofrontal — subjective value
    "amygdala": +0.45,   # salience/arousal
    "dmPFC":    +0.50,   # self-relevance (Falk self-localizer MPFC)
    "dlPFC":    -0.60,   # deliberative control — brakes purchase
    "insula":   -0.70,   # anticipated loss / price pain (Knutson 2007)
}

# Falk 2016: neural term and self-report are complementary; combined beats either.
# Default ensemble weight on the neural index when both are present.
DEFAULT_NEURAL_WEIGHT = 0.6   # 0.6 neural / 0.4 self-report; tune on held-out data


@dataclass
class MarketForecast:
    """A calibrated brain -> market-behavior forecast for one stimulus."""
    index: float                       # 0-100 calibrated market-forecast index
    band: tuple[float, float]          # 68% uncertainty interval
    neural_component: float            # neural-only index (pre-ensemble)
    self_report_component: float | None
    dominant_evidence: str             # which construct/region carried it
    representativeness_robust: bool     # is this driven by the generalizable (affective) signal?
    provenance: str                    # plain-language "why this number"
    caveats: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lo, hi = self.band
        return (f"Market-forecast index {self.index:.1f} "
                f"[{lo:.1f}-{hi:.1f}]  ·  driver: {self.dominant_evidence}  ·  "
                f"{'sample-robust' if self.representativeness_robust else 'sample-sensitive'}")


class BehavioralBridge:
    """
    Turns Mary's grounded construct read-outs into a calibrated market-behavior
    forecast, weighting the neural components the literature says generalize to
    aggregate choice.
    """

    def __init__(self,
                 construct_priors: dict | None = None,
                 region_priors: dict | None = None,
                 neural_weight: float = DEFAULT_NEURAL_WEIGHT):
        self.cw = dict(construct_priors or AIM_CONSTRUCT_PRIORS)
        self.rw = dict(region_priors or SIGNED_REGION_PRIORS)
        self.neural_weight = float(np.clip(neural_weight, 0.0, 1.0))
        # Calibration state (set by .calibrate); identity by default.
        self._scale = 1.0
        self._offset = 0.0
        self._resid_sd = 12.0   # prior uncertainty (index points) until calibrated

    # ---- 1. neural index from construct read-outs (affect-weighted) ----
    def _neural_index(self, construct_readouts: dict,
                      region_means: dict | None = None) -> tuple[float, str, bool]:
        """
        Weighted combination of grounded constructs (+ optional signed ROIs).
        Returns (raw_index, dominant_evidence, affect_driven).
        """
        contribs: dict[str, float] = {}
        for name, score in construct_readouts.items():
            if name.startswith("_"):          # skip confound channels
                continue
            w = self.cw.get(name, 0.0)
            contribs[name] = w * float(score)

        if region_means:
            for roi, act in region_means.items():
                w = self.rw.get(roi, 0.0)
                contribs[f"roi:{roi}"] = w * float(act)

        raw = float(sum(contribs.values()))

        # dominant evidence = largest-magnitude contributor
        dominant = max(contribs, key=lambda k: abs(contribs[k])) if contribs else "none"

        # affect-driven if the dominant contributor is an affective construct/region
        affect_keys = {"value", "reward", "arousal", "emotion",
                       "roi:NAcc", "roi:mOFC", "roi:amygdala"}
        affect_driven = dominant in affect_keys

        return raw, dominant, affect_driven

    # ---- 2. calibrate raw neural index to an outcome scale (optional) ----
    def calibrate(self, raw_neural: np.ndarray, outcomes: np.ndarray) -> dict:
        """
        Fit a single scale+offset mapping raw neural index -> outcome units
        (e.g. CTR, sales lift, call-volume change). Keeps the model interpretable:
        the *shape* comes from literature priors, the *scale* from your data.
        """
        raw_neural = np.asarray(raw_neural, float)
        outcomes = np.asarray(outcomes, float)
        A = np.column_stack([np.ones_like(raw_neural), raw_neural])
        (offset, scale), *_ = np.linalg.lstsq(A, outcomes, rcond=None)
        self._offset, self._scale = float(offset), float(scale)
        resid = outcomes - (self._offset + self._scale * raw_neural)
        self._resid_sd = float(np.std(resid)) if len(resid) > 1 else self._resid_sd
        r = float(np.corrcoef(raw_neural, outcomes)[0, 1]) if len(raw_neural) > 1 else float("nan")
        return {"scale": self._scale, "offset": self._offset,
                "resid_sd": self._resid_sd, "in_sample_r": r,
                "n": int(len(raw_neural))}

    @staticmethod
    def _to_0_100(x: float) -> float:
        # squashing map so an uncalibrated index is still reportable on a 0-100 scale
        return float(100.0 / (1.0 + np.exp(-x)))

    # ---- 3. full forecast ----
    def forecast(self,
                 construct_readouts: dict,
                 region_means: dict | None = None,
                 self_report: float | None = None,
                 specificity_passed: bool | None = None) -> MarketForecast:
        """
        Produce a calibrated market-forecast index for one stimulus.

        specificity_passed: pass the result of mary_readouts specificity_gate here.
            If it FAILED (signal is sensory/confound, not construct), we withhold the
            strong claim and flag it — mirroring Falk 2016 content-dependence.
        """
        raw, dominant, affect_driven = self._neural_index(construct_readouts, region_means)

        # calibrated or squashed neural index on 0-100
        if self._scale != 1.0 or self._offset != 0.0:
            neural_idx = float(self._offset + self._scale * raw)
        else:
            neural_idx = self._to_0_100(raw)

        caveats: list[str] = []

        # ensemble with self-report (Falk 2016: combined > either alone)
        if self_report is not None:
            idx = self.neural_weight * neural_idx + (1 - self.neural_weight) * float(self_report)
            sr_component = float(self_report)
        else:
            idx = neural_idx
            sr_component = None
            caveats.append("No self-report supplied; neural-only. Combined models "
                           "forecast better (Falk 2016).")

        # specificity gate integration
        if specificity_passed is False:
            caveats.append("Specificity gate FAILED: the contrast is better explained "
                           "by low-level sensory confound than by the named construct. "
                           "Market claim withheld — treat index as sensory salience only "
                           "(Falk 2016 content-dependence).")
        if not affect_driven:
            caveats.append("Forecast is carried by an integrative/attention signal, which "
                           "generalizes less across individuals than affect; expect the "
                           "aggregate forecast to be more sample-sensitive "
                           "(Knutson & Genevsky 2018).")

        # uncertainty band (68%)
        band = (max(0.0, idx - self._resid_sd), min(100.0, idx + self._resid_sd))

        provenance = self._provenance(dominant, affect_driven, self_report is not None,
                                      specificity_passed)

        return MarketForecast(
            index=float(np.clip(idx, 0, 100)),
            band=band,
            neural_component=float(np.clip(neural_idx, 0, 100)),
            self_report_component=sr_component,
            dominant_evidence=dominant,
            representativeness_robust=affect_driven and (specificity_passed is not False),
            provenance=provenance,
            caveats=caveats,
        )

    @staticmethod
    def _provenance(dominant: str, affect_driven: bool,
                    has_self_report: bool, spec: bool | None) -> str:
        parts = [f"Driven by {dominant}."]
        if affect_driven:
            parts.append("Anticipatory-affect signal — the component shown to "
                         "generalize to aggregate choice across individuals and to stay "
                         "valid on non-representative samples (Knutson 2007; Falk 2012; "
                         "Genevsky 2025).")
        else:
            parts.append("Integrative/attention signal — supports encoding and recall "
                         "(Boksem & Smidts 2015) but is a weaker aggregate driver.")
        if has_self_report:
            parts.append("Ensembled with self-report (Falk 2016: combined R^2 up to 0.65).")
        if spec is False:
            parts.append("Specificity gate failed — reported as sensory salience, not value.")
        return " ".join(parts)


# ---- batch convenience ----
def forecast_campaign_set(bridge: BehavioralBridge,
                          readouts_per_ad: list[dict],
                          self_reports: list[float] | None = None,
                          region_means_per_ad: list[dict] | None = None,
                          specificity_flags: list[bool] | None = None) -> list[MarketForecast]:
    """Forecast a whole set of ads and return them rank-orderable by .index."""
    n = len(readouts_per_ad)
    sr = self_reports or [None] * n
    rm = region_means_per_ad or [None] * n
    sp = specificity_flags or [None] * n
    return [bridge.forecast(readouts_per_ad[i], region_means=rm[i],
                            self_report=sr[i], specificity_passed=sp[i])
            for i in range(n)]
