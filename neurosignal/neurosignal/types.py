"""Typed results."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConstructActivation:
    key: str
    label: str
    score: float            # 0..100 normalized activation for this construct
    covered: bool           # was at least one of its regions present in the input?
    cortical_proxy: bool    # subcortical region approximated by a cortical proxy
    regions: list[str]
    citations: list[str]


@dataclass
class DetectionResult:
    constructs: list[ConstructActivation]
    buy_sell_score: float   # 0..100 approach-avoidance composite
    recommendation: str     # "Buy" | "Hold" | "Sell"
    confidence: float       # 0..1 (rises with coverage, falls with conflict)
    coverage: float         # fraction of constructs the input could measure
    source: str             # what produced the activations
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "buy_sell_score": self.buy_sell_score,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "source": self.source,
            "constructs": [
                {"key": c.key, "label": c.label, "score": c.score, "covered": c.covered,
                 "cortical_proxy": c.cortical_proxy, "regions": c.regions, "citations": c.citations}
                for c in self.constructs
            ],
            "notes": self.notes,
        }


@dataclass
class MetricScore:
    key: str
    label: str
    score: float            # 0..100 (or -100..100 for signed metrics like valence)
    unit: str               # "0-100" | "-100..100"
    interpretation: str     # short human reading (low/moderate/high or directional)
    basis: str              # one-line neuro basis
    ci95: tuple | None = None   # conformal interval, when a calibration set is supplied
    withheld: bool = False      # True if the specificity gate withheld this score (confounded)


@dataclass
class Analysis:
    """Full multimodal neuro-analysis: headline metrics + networks + constructs."""
    metrics: list["MetricScore"]
    networks: dict[str, float]                 # Yeo-7 network activation, 0..100
    constructs: list["ConstructActivation"]    # the 7 consumer-neuro constructs
    modalities: list[str]                      # which of text/audio/video contributed
    coverage: float
    confidence: float
    source: str
    timeline: dict | None = None
    notes: list[str] = field(default_factory=list)
    personalized: bool = False                 # True if a personal brain_file was applied
    provenance: dict | None = None             # enrollment/encoder manifest when personalized

    def to_dict(self) -> dict:
        return {
            "metrics": [m.__dict__.copy() for m in self.metrics],
            "networks": self.networks,
            "constructs": [
                {"key": c.key, "label": c.label, "score": c.score,
                 "covered": c.covered, "cortical_proxy": c.cortical_proxy}
                for c in self.constructs
            ],
            "modalities": self.modalities,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "source": self.source,
            "timeline": self.timeline,
            "notes": self.notes,
            "personalized": self.personalized,
            "provenance": self.provenance,
        }
