"""Layer 3 — the STATS layer.

Raw second-by-second frames from `timeline.py` are pushed here and compiled into the
clean, summarized forms the visualization consumes: per-metric time-series statistics,
the moments that matter (peak persuasion / arousal / buy pressure), network dynamics
(which areas move most), and segment roll-ups. Deliberately separate from encoding so
the raw read-out and its summarization never entangle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .timeline import Timeline

_METRIC_LABELS = {
    "valence": "Affective Valence", "arousal": "Arousal / Intensity",
    "engagement": "Engagement", "manipulation": "Persuasion Pressure",
    "sycophancy": "Sycophancy", "buy_sell": "Buy / Sell Signal",
}


@dataclass
class MetricStats:
    key: str
    label: str
    mean: float
    peak: float
    peak_t: float
    trough: float
    trough_t: float
    auc: float            # area under the curve (time-integrated exposure)
    volatility: float     # std of second-to-second change
    final: float


@dataclass
class Stats:
    metrics: dict                 # key -> MetricStats (as dict)
    peak_moments: list            # [{t, metric, label, score}] sorted by salience
    network_dynamics: dict        # network -> {mean, peak, range}
    segments: list                # coarse roll-ups (thirds): [{label, t0, t1, manipulation, buy_sell}]
    duration: float
    n_frames: int
    personalized: bool
    headline: str                 # one-line plain-English summary

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics, "peak_moments": self.peak_moments,
            "network_dynamics": self.network_dynamics, "segments": self.segments,
            "duration": self.duration, "n_frames": self.n_frames,
            "personalized": self.personalized, "headline": self.headline,
        }


def _series_stats(key, ts, vals) -> MetricStats:
    a = np.asarray(vals, dtype=float)
    t = np.asarray(ts, dtype=float)
    peak_i, trough_i = int(np.argmax(a)), int(np.argmin(a))
    _trapz = getattr(np, "trapezoid", np.trapz)   # numpy>=2 renamed trapz -> trapezoid
    vol = float(np.std(np.diff(a))) if a.size > 1 else 0.0
    auc = float(_trapz(a, t)) if a.size > 1 else (float(a[0]) if a.size else 0.0)
    return MetricStats(
        key=key, label=_METRIC_LABELS.get(key, key),
        mean=round(float(a.mean()), 2), peak=round(float(a[peak_i]), 2),
        peak_t=round(float(t[peak_i]), 2), trough=round(float(a[trough_i]), 2),
        trough_t=round(float(t[trough_i]), 2), auc=round(auc, 2),
        volatility=round(vol, 2), final=round(float(a[-1]), 2),
    )


def compile_stats(timeline: Timeline) -> Stats:
    if not timeline.frames:
        raise ValueError("empty timeline")
    keys = sorted({k for fr in timeline.frames for k in fr.metrics})
    metrics: dict[str, dict] = {}
    for k in keys:
        ts, vals = timeline.metric_series(k)
        metrics[k] = asdict(_series_stats(k, ts, vals))

    # moments that matter: top peaks across persuasion/arousal/engagement/buy
    salient = ["manipulation", "arousal", "engagement", "buy_sell"]
    peak_moments = []
    for k in salient:
        if k in metrics:
            ms = metrics[k]
            peak_moments.append({"t": ms["peak_t"], "metric": k, "label": ms["label"],
                                 "score": ms["peak"]})
    peak_moments.sort(key=lambda d: d["score"], reverse=True)

    # network dynamics: which areas move most over the clip
    nets = sorted({n for fr in timeline.frames for n in fr.networks})
    net_dyn = {}
    for n in nets:
        _, nv = timeline.network_series(n)
        arr = np.asarray(nv, dtype=float)
        net_dyn[n] = {"mean": round(float(arr.mean()), 1), "peak": round(float(arr.max()), 1),
                      "range": round(float(arr.max() - arr.min()), 1)}

    # coarse thirds roll-up (intro / build / payoff)
    n = len(timeline.frames)
    bounds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    labels = ["intro", "build", "payoff"]
    segments = []
    for (lo, hi), lab in zip(bounds, labels):
        fr = timeline.frames[lo:hi] or timeline.frames[lo:lo + 1]
        segments.append({
            "label": lab, "t0": round(fr[0].t, 2), "t1": round(fr[-1].t, 2),
            "manipulation": round(float(np.mean([f.metrics.get("manipulation", 0) for f in fr])), 1),
            "buy_sell": round(float(np.mean([f.buy_sell for f in fr])), 1),
        })

    top = peak_moments[0] if peak_moments else None
    most_dynamic = max(net_dyn.items(), key=lambda kv: kv[1]["range"]) if net_dyn else None
    headline = (
        f"Peak {top['label'].lower()} ({top['score']:.0f}) at {top['t']:.0f}s; "
        f"most dynamic network: {most_dynamic[0]} (Δ{most_dynamic[1]['range']:.0f})."
        if top and most_dynamic else "Timeline compiled."
    )

    return Stats(
        metrics=metrics, peak_moments=peak_moments, network_dynamics=net_dyn,
        segments=segments, duration=timeline.duration, n_frames=n,
        personalized=timeline.personalized, headline=headline,
    )
