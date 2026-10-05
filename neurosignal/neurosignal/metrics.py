"""The neuro-metric layer: network activation -> interpretable, theory-driven scores.

Headline metrics (beyond the 7 constructs and the buy/sell signal):
  * Affective Valence     — approach/withdrawal balance (limbic reward/emotion vs insula/ACC).
  * Arousal / Intensity   — limbic + salience activation magnitude.
  * Engagement            — attention + salience + reward.
  * Manipulation Index    — affective/reward drive relative to analytic (frontoparietal) control;
                            i.e. System-1 capture with suppressed deliberation (dual-process).
  * Sycophancy Index      — (text) social-reward + agreement/praise signaling with low epistemic
                            conflict; useful for auditing LLM-generated flattery.

These are transparent, theory-driven indices over (model-predicted or measured) activation —
not validated clinical instruments. Every score exposes its `basis`.
"""
from __future__ import annotations

import numpy as np

from .detect import detect_from_networks
from .types import Analysis, MetricScore


def _normalize(d: dict[str, float]) -> dict[str, float]:
    arr = np.array(list(d.values()), dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _band(x: float) -> str:
    return "low" if x < 33 else ("high" if x > 66 else "moderate")


def compute_metrics(network_activation, *, text_features=None,
                    modalities=("?",), source: str = "encoder",
                    normalize: bool = True) -> Analysis:
    det = detect_from_networks(network_activation, source=source, normalize=normalize)
    vals = {k: float(v) for k, v in network_activation.items()}
    nf = _normalize(vals) if normalize else {k: float(np.clip(v, 0.0, 1.0)) for k, v in vals.items()}
    f = lambda n: float(nf.get(n, 0.5))

    reward = (f("Limbic") + f("Default")) / 2.0
    emotion = f("Limbic")
    salience = f("VentralAttention")          # ACC / anterior insula (conflict)
    control = f("Frontoparietal")             # DLPFC analytic control
    arousal_frac = (f("Limbic") + f("VentralAttention")) / 2.0
    affect = (reward + emotion + arousal_frac) / 3.0

    valence = round(float(np.clip(100.0 * (((reward + emotion) / 2.0) - salience), -100, 100)), 1)
    arousal = round(100.0 * arousal_frac, 1)
    engagement = round(100.0 * ((f("DorsalAttention") + f("VentralAttention") + reward) / 3.0), 1)
    manipulation = round(float(np.clip(100.0 * (0.5 + (affect - control)), 0, 100)), 1)

    metrics = [
        MetricScore("valence", "Affective Valence", valence, "-100..100",
                    ("positive / approach" if valence > 10 else "negative / withdrawal" if valence < -10 else "neutral"),
                    "limbic (vmPFC) reward+emotion vs anterior-insula/ACC withdrawal"),
        MetricScore("arousal", "Arousal / Intensity", arousal, "0-100", _band(arousal),
                    "limbic (amygdala) + salience-network activation"),
        MetricScore("engagement", "Engagement", engagement, "0-100", _band(engagement),
                    "dorsal attention + salience + reward drive"),
        MetricScore("manipulation", "Manipulation Index", manipulation, "0-100", _band(manipulation),
                    "affective/reward drive relative to analytic (frontoparietal) control — System-1 capture"),
        MetricScore("buy_sell", "Buy / Sell Signal", det.buy_sell_score, "0-100", det.recommendation,
                    "approach-avoidance neuroforecasting (reward minus conflict)"),
    ]
    notes = list(det.notes)

    if text_features is not None:
        tf = text_features.as_dict() if hasattr(text_features, "as_dict") else dict(text_features)
        social = f("Default")
        syc = float(np.clip(0.55 * float(tf.get("sycophancy_lexical", 0.0))
                            + 0.45 * (0.5 + 0.5 * (social - salience)), 0.0, 1.0)) * 100.0
        metrics.append(MetricScore(
            "sycophancy", "Sycophancy / Flattery Index", round(syc, 1), "0-100", _band(syc),
            "social-reward + agreement/praise signaling with low epistemic conflict (text-mode)"))
    else:
        notes.append("Sycophancy index requires text input (text-mode metric).")

    networks_0_100 = {k: round(v * 100.0, 1) for k, v in nf.items()}
    return Analysis(
        metrics=metrics, networks=networks_0_100, constructs=det.constructs,
        modalities=list(modalities), coverage=det.coverage, confidence=det.confidence,
        source=source, timeline=None, notes=notes,
    )
