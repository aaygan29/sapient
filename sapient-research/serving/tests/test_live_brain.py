"""The live-brain adapter, end-to-end through the mock engine (no checkpoint).
Run: python3 tests/test_live_brain.py   (or via pytest)
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "..", "neurosignal"))

from sapient_serving.engine.base import Stimulus            # noqa: E402
from sapient_serving.engine.live_brain import engine_to_reports, live_brain_html  # noqa: E402
from sapient_serving.engine.mock import MockEngine          # noqa: E402

_failures = []


def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    if not cond:
        _failures.append(name)


def test_engine_exposes_network_timeline():
    pred = MockEngine().predict(Stimulus(transcript="luxury reveal", filename="a.mp4"))
    nt = pred.signals.timeline.network_timeline
    check("network_timeline populated", bool(nt))
    check("has all 7 Yeo-7 networks", nt is not None and len(nt) == 7)
    check("per-second length matches n_timepoints",
          nt is not None and len(next(iter(nt.values()))) == pred.n_timepoints)


def test_adapter_builds_reports_and_personalizes():
    eng = MockEngine()
    stim = Stimulus(transcript="beautiful cinematic luxury reveal, exciting reward", filename="ad.mp4")
    reports = engine_to_reports(eng, stim, subject_idxs=(0, 1, 2),
                                subject_labels=["S0", "S1", "S2"])
    check("3 subject reports", set(reports) == {"S0", "S1", "S2"})
    f0 = reports["S0"]["timeline"]["frames"]
    check("100 per-second frames", len(f0) == 100)
    check("frames carry metrics + networks",
          "manipulation" in f0[0]["metrics"] and "Limbic" in f0[0]["networks"])
    # subject_idx conditions the (mock) per-subject head -> distinct brains
    s0 = reports["S0"]["timeline"]["frames"][10]["networks"]
    s1 = reports["S1"]["timeline"]["frames"][10]["networks"]
    check("different subjects -> different brains", s0 != s1)


def test_html_renders():
    eng = MockEngine()
    html = live_brain_html(eng, Stimulus(transcript="exciting reveal", filename="a.mp4"),
                           subject_idxs=(0, 1))
    check("html has the live-brain markup", "ns-brain" in html and "DATA = {" in html)
    check("payload substituted", "__DATA__" not in html)


if __name__ == "__main__":
    for t in (test_engine_exposes_network_timeline,
              test_adapter_builds_reports_and_personalizes, test_html_renders):
        print(f"\n[{t.__name__}]")
        t()
    print("\n" + ("ALL LIVE-BRAIN TESTS PASSED" if not _failures else f"FAILURES: {_failures}"))
    sys.exit(1 if _failures else 0)
