"""Layer 2 — temporal neural encoding.

Turns uploaded media into a second-by-second `Timeline` of brain-network activations,
constructs, and neuro-behavioral metrics (valence, arousal, engagement, persuasion
pressure, buy/sell). This is what drives the live "digital brain" and the stats layer.

Two input paths:
  * per_second_networks : list of {Yeo-7 network: activation} — the LEARNED-MARY path
                          (Mary predicts cortical activation per second; feed it straight in).
  * per_second_features : list of {modality: feature_dict} — run through an encoder
                          (reference, or an enrolled personal brain via brain_file).

Architecture:  INGEST -> [ENCODE = this module] -> STATS (stats.py) -> VISUALIZE (viz.py)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .encoders import get_encoder
from .metrics import compute_metrics

# the metrics surfaced on the timeline (manipulation = persuasion/System-1-capture pressure)
TIMELINE_METRICS = ("valence", "arousal", "engagement", "manipulation", "sycophancy", "buy_sell")


@dataclass
class TimelineFrame:
    t: float                      # seconds from start
    networks: dict                # Yeo-7 network activation, 0..100
    metrics: dict                 # {valence, arousal, engagement, manipulation, buy_sell, ...}
    constructs: dict              # {reward_value, emotion, attention, ...} 0..100
    buy_sell: float
    recommendation: str


@dataclass
class Timeline:
    frames: list                  # list[TimelineFrame]
    fps: float = 1.0
    duration: float = 0.0
    source: str = "encoder"
    personalized: bool = False
    provenance: dict | None = None
    notes: list = field(default_factory=list)

    def times(self):
        return [f.t for f in self.frames]

    def metric_series(self, key: str):
        return [f.t for f in self.frames], [f.metrics.get(key, float("nan")) for f in self.frames]

    def network_series(self, network: str):
        return [f.t for f in self.frames], [f.networks.get(network, 0.0) for f in self.frames]

    def to_dict(self) -> dict:
        return {
            "fps": self.fps, "duration": self.duration, "source": self.source,
            "personalized": self.personalized, "provenance": self.provenance,
            "notes": self.notes,
            "frames": [
                {"t": fr.t, "networks": fr.networks, "metrics": fr.metrics,
                 "constructs": fr.constructs, "buy_sell": fr.buy_sell,
                 "recommendation": fr.recommendation}
                for fr in self.frames
            ],
        }


def _frame_from_analysis(t: float, analysis) -> TimelineFrame:
    return TimelineFrame(
        t=round(float(t), 3),
        networks=dict(analysis.networks),
        metrics={m.key: m.score for m in analysis.metrics},
        constructs={c.key: c.score for c in analysis.constructs},
        buy_sell=next((m.score for m in analysis.metrics if m.key == "buy_sell"), 0.0),
        recommendation=next((m.interpretation for m in analysis.metrics if m.key == "buy_sell"), "Hold"),
    )


def analyze_timeline(*, per_second_networks=None, per_second_features=None,
                     fps: float = 1.0, encoder: str = "reference", brain_file=None,
                     source: str | None = None) -> Timeline:
    """Produce a second-by-second Timeline.

    Provide exactly one of `per_second_networks` (already-predicted Yeo-7 activations,
    e.g. from Mary) or `per_second_features` (raw features to run through an encoder).
    `brain_file` personalizes the read-out to an enrolled subject (live per-person brain).
    """
    if (per_second_networks is None) == (per_second_features is None):
        raise ValueError("provide exactly one of per_second_networks or per_second_features")

    bf = None
    if brain_file is not None:
        from .enrollment import BrainFile
        bf = brain_file if isinstance(brain_file, BrainFile) else BrainFile.load(brain_file)

    # ---- 1. collect raw per-second network dicts (apply personalization) ----
    raw: list[dict] = []
    text_seq: list = []
    mod_seq: list = []
    enc_name = "predicted-networks"
    if per_second_networks is not None:
        head = (bf.network_head or {}) if bf else {}   # Mary path: per-subject head adjusts predicted nets
        for item in per_second_networks:
            d = {k: float(v) for k, v in item.items()}
            if head:
                d = {k: v + head.get(k, 0.0) for k, v in d.items()}
            raw.append(d); text_seq.append(None); mod_seq.append(["fmri"])
    else:
        enc = get_encoder(encoder)
        if bf is not None:
            from .enrollment import EnrolledEncoder
            enc = EnrolledEncoder(enc, bf)
        enc_name = getattr(enc, "name", "encoder")
        for item in per_second_features:
            raw.append(enc.encode(item)); text_seq.append(item.get("text")); mod_seq.append(list(item.keys()))

    raw = [d for d in raw if d]
    if not raw:
        raise ValueError("no network activation produced from the inputs")

    # ---- 2. GLOBAL scaling across the whole clip (so frames are comparable over time) ----
    all_vals = [v for d in raw for v in d.values()]
    lo, hi = min(all_vals), max(all_vals)
    span = (hi - lo) if (hi - lo) > 1e-9 else 1.0
    scale01 = lambda d: {k: (v - lo) / span for k, v in d.items()}

    # ---- 3. metrics per frame on the globally-scaled networks (normalize=False) ----
    frames: list[TimelineFrame] = []
    dt = 1.0 / fps if fps else 1.0
    for i, d in enumerate(raw):
        a = compute_metrics(scale01(d), text_features=text_seq[i], modalities=mod_seq[i],
                            source=source or "timeline", normalize=False)
        frames.append(_frame_from_analysis(i * dt, a))

    tl = Timeline(frames=frames, fps=fps, duration=len(frames) * dt,
                  source=source or enc_name)
    if bf is not None:
        tl.personalized = True
        tl.provenance = dict(bf.provenance or {}, subject=bf.subject,
                             brain_file_fingerprint=bf.fingerprint)
        tl.notes.append(f"Personalized live brain for enrolled subject '{bf.subject}'.")
    return tl
