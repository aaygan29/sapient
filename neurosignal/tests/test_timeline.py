"""Tests for the temporal pipeline (timeline) + stats layer.
Run: python3 tests/test_timeline.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurosignal as ns  # noqa: E402
from neurosignal.stats import compile_stats  # noqa: E402
from neurosignal.timeline import analyze_timeline  # noqa: E402

_failures = []


def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    if not cond:
        _failures.append(name)


def _ad_arc():
    """12s of per-second Yeo-7 activations: calm intro -> reward build -> urgent CTA."""
    frames = []
    for t in range(12):
        if t < 4:      # intro: calm, visual
            f = {"Visual": 0.6, "Default": 0.6, "Somatomotor": 0.4, "DorsalAttention": 0.4,
                 "VentralAttention": 0.2, "Limbic": 0.3, "Frontoparietal": 0.5}
        elif t < 8:    # build: reward + attention rising
            f = {"Visual": 0.7, "Default": 0.5, "Somatomotor": 0.4, "DorsalAttention": 0.7,
                 "VentralAttention": 0.4, "Limbic": 0.6 + 0.04 * t, "Frontoparietal": 0.45}
        else:          # payoff: urgency salience spikes, analytic control drops
            f = {"Visual": 0.6, "Default": 0.4, "Somatomotor": 0.4, "DorsalAttention": 0.6,
                 "VentralAttention": 0.85, "Limbic": 0.9, "Frontoparietal": 0.2}
        frames.append(f)
    return frames


def test_timeline_from_networks():
    tl = analyze_timeline(per_second_networks=_ad_arc(), fps=1.0, source="ad-arc(illustrative)")
    check("12 frames produced", len(tl.frames) == 12)
    check("each frame has metrics", all("manipulation" in f.metrics for f in tl.frames))
    check("timeline serializes", isinstance(tl.to_dict(), dict))
    ts, manip = tl.metric_series("manipulation")
    check("persuasion rises into the CTA (payoff > intro)",
          sum(manip[8:]) / 4 > sum(manip[:4]) / 4)


def test_stats_layer():
    tl = analyze_timeline(per_second_networks=_ad_arc(), fps=1.0)
    st = compile_stats(tl)
    check("stats has per-metric summaries", "manipulation" in st.metrics)
    check("peak persuasion is in the payoff third (t>=8)", st.metrics["manipulation"]["peak_t"] >= 8)
    check("network dynamics computed", "VentralAttention" in st.network_dynamics)
    check("VentralAttention is highly dynamic", st.network_dynamics["VentralAttention"]["range"] > 20)
    check("headline generated", len(st.headline) > 0)
    check("stats serializes", isinstance(st.to_dict(), dict))


def test_personalized_timeline_differs():
    from neurosignal import BrainFile
    base = analyze_timeline(per_second_networks=_ad_arc())
    # Mary path: a per-subject head adjusts the predicted networks -> a per-person live brain
    bf = BrainFile(subject="reward_seeker",
                   network_head={"Limbic": +0.3, "Frontoparietal": -0.25, "VentralAttention": +0.1})
    p = analyze_timeline(per_second_networks=_ad_arc(), brain_file=bf)
    check("personalized flag set", p.personalized is True)
    _, mb = base.metric_series("manipulation")
    _, mp = p.metric_series("manipulation")
    check("personalized timeline differs from baseline", mb != mp)
    check("reward-seeker shows higher mean persuasion",
          sum(mp) / len(mp) > sum(mb) / len(mb))


if __name__ == "__main__":
    for t in (test_timeline_from_networks, test_stats_layer, test_personalized_timeline_differs):
        print(f"\n[{t.__name__}]")
        t()
    print("\n" + ("ALL TIMELINE TESTS PASSED" if not _failures else f"FAILURES: {_failures}"))
    sys.exit(1 if _failures else 0)
