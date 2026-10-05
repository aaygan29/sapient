"""neurosignal — evidence-based, multimodal detection of neuro-behavioral constructs.

Two entry points:
  * analyze(text=/audio=/video=)  — media -> brain-network activation (via an encoder) ->
    metrics: valence, arousal, engagement, manipulation, sycophancy, buy/sell + the 7 constructs.
  * detect_from_networks / detect_from_parcels — score activation you already have
    (measured fMRI on an atlas, or model-predicted).

Honest by design: outputs report coverage/confidence, flag cortical proxies, and distinguish
the transparent reference encoder from a learned fMRI encoder.
"""
from .analyze import analyze
from .constructs import CONSTRUCTS, YEO7_NETWORKS, Construct
from .detect import detect_from_networks
from .enrollment import (
    AdditiveEncoder, BrainFile, EnrolledEncoder, MaryAdapter,
    enroll_head, enroll_network_head,
)
from .inputs.parcels import detect_from_parcels
from .metrics import compute_metrics
from .stats import Stats, compile_stats
from .timeline import Timeline, TimelineFrame, analyze_timeline
from .types import Analysis, ConstructActivation, DetectionResult, MetricScore
from .viz import render_report, save_report
from .validation import (
    Finding, Provenance, conformal_interval, empirical_coverage, gate,
    specificity_gate, split_conformal_regression,
)

__version__ = "0.3.0"
__all__ = [
    "analyze", "compute_metrics",
    "detect_from_networks", "detect_from_parcels",
    "CONSTRUCTS", "Construct", "YEO7_NETWORKS",
    "Analysis", "MetricScore", "ConstructActivation", "DetectionResult",
    # enrollment — personal digital brains
    "AdditiveEncoder", "MaryAdapter", "enroll_head", "enroll_network_head",
    "EnrolledEncoder", "BrainFile",
    # validation — the honesty layer
    "Finding", "gate", "Provenance", "specificity_gate",
    "split_conformal_regression", "empirical_coverage", "conformal_interval",
    # temporal pipeline + stats + viz (the live digital brain)
    "analyze_timeline", "Timeline", "TimelineFrame",
    "compile_stats", "Stats", "render_report", "save_report",
    "__version__",
]
