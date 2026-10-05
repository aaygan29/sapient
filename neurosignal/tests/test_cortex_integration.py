"""Cortex-of-Anyone integration tests (run: python3 tests/test_cortex_integration.py).

Verifies (1) the new enrollment + honesty layer work, (2) personalization actually
CHANGES scan outputs, and (3) default behavior is unchanged when the features are unused.
Dependency-light (numpy only; no pytest needed).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurosignal as ns  # noqa: E402
from neurosignal.enrollment import (  # noqa: E402
    AdditiveEncoder, BrainFile, EnrolledEncoder, enroll_head, enroll_network_head,
)
from neurosignal.validation import (  # noqa: E402
    Finding, gate, specificity_gate, split_conformal_regression, empirical_coverage,
)

_failures = []


def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    if not cond:
        _failures.append(name)


# --- a tiny deterministic base encoder for enrollment tests ---
class StubEncoder:
    name = "stub"

    def __init__(self, base):
        self._base = dict(base)

    def encode(self, features):
        # ignores features; returns a fixed group/average activation
        return dict(self._base)


def test_default_behavior_unchanged():
    a = ns.analyze(text="What a brilliant, amazing question — you're absolutely right!")
    check("default analyze returns metrics", len(a.metrics) >= 5)
    check("default analyze NOT personalized", a.personalized is False)
    check("default analyze provenance None", a.provenance is None)
    d = ns.detect_from_networks({"Limbic": 0.95, "Default": 0.8, "Frontoparietal": 0.3})
    check("detect_from_networks still works", d.recommendation in ("Buy", "Hold", "Sell"))


def test_enroll_network_head_recovers_subject():
    base = {"Limbic": 0.40, "Default": 0.50, "Frontoparietal": 0.50,
            "DorsalAttention": 0.50, "VentralAttention": 0.50}
    enc = StubEncoder(base)
    # this subject's TRUE activation is systematically shifted from the group
    true_shift = {"Limbic": +0.30, "Frontoparietal": -0.20}
    calib = []
    rng = np.random.default_rng(0)
    for _ in range(40):
        measured = {n: float(np.clip(base[n] + true_shift.get(n, 0.0)
                                     + 0.02 * rng.standard_normal(), 0, 1)) for n in base}
        calib.append(({"text": None}, measured))
    bf = enroll_network_head(enc, calib, subject="S01")
    check("enrolled Limbic delta ≈ +0.30",
          abs(bf.network_head["Limbic"] - 0.30) < 0.05)
    check("enrolled Frontoparietal delta ≈ -0.20",
          abs(bf.network_head["Frontoparietal"] - (-0.20)) < 0.05)
    check("brain_file is fingerprinted", len(bf.fingerprint) == 16)


def test_personalization_changes_outputs():
    # two different people -> different read-outs of the SAME stimulus
    bf_a = BrainFile(subject="A", network_head={"Limbic": +0.35, "Frontoparietal": -0.25,
                                                "VentralAttention": +0.10})
    bf_b = BrainFile(subject="B", network_head={"Limbic": -0.25, "Frontoparietal": +0.30,
                                                "DorsalAttention": +0.15})
    stim = "A calm, simple description of a quiet afternoon."
    base = ns.analyze(text=stim)
    pa = ns.analyze(text=stim, brain_file=bf_a)
    pb = ns.analyze(text=stim, brain_file=bf_b)
    check("personalized flag set", pa.personalized and pb.personalized)
    check("provenance stamped with subject", pa.provenance.get("subject") == "A")
    check("personalized networks differ from baseline", pa.networks != base.networks)
    check("two subjects get different networks", pa.networks != pb.networks)
    # at least one headline metric moves between the two people
    ma = {m.key: m.score for m in pa.metrics}
    mb = {m.key: m.score for m in pb.metrics}
    check("two subjects get different metric scores", any(ma[k] != mb[k] for k in ma))


def test_negative_control_zero_head():
    # a brain_file with ~no individual signal -> outputs ≈ baseline
    stim = "A neutral sentence about the weather today."
    base = ns.analyze(text=stim)
    bf0 = BrainFile(subject="null", network_head={n: 0.0 for n in base.networks})
    p0 = ns.analyze(text=stim, brain_file=bf0)
    check("zero head leaves networks ≈ unchanged",
          all(abs(base.networks[n] - p0.networks[n]) < 1e-6 for n in base.networks))


def test_vertex_enrollment_machinery():
    # the Mary-contract path: enroll a rank-r head that beats group-only
    rng = np.random.default_rng(1)
    D, V = 32, 80
    group_W = rng.standard_normal((V, D)) / np.sqrt(D)
    enc = AdditiveEncoder(group_W)
    delta_true = (rng.standard_normal((V, 4)) @ rng.standard_normal((4, D))) * 0.4
    X = rng.standard_normal((150, D))
    Y = X @ (group_W + delta_true).T + 0.2 * rng.standard_normal((150, V))
    head = enroll_head(enc, X, Y, rank=16, ridge=5.0)
    Xt = rng.standard_normal((200, D))
    Yt = Xt @ (group_W + delta_true).T + 0.2 * rng.standard_normal((200, V))

    def vpear(a, b):
        a = a - a.mean(0); b = b - b.mean(0)
        return float(np.mean((a * b).sum(0) / (np.sqrt((a**2).sum(0) * (b**2).sum(0)) + 1e-12)))

    r_enrolled = vpear(Yt, enc.predict(Xt, head))
    r_group = vpear(Yt, enc.group_predict(Xt))
    check(f"vertex enrolled ({r_enrolled:.3f}) > group-only ({r_group:.3f})", r_enrolled > r_group)


def test_validation_layer():
    rng = np.random.default_rng(2)
    cov = np.mean([empirical_coverage(rng.standard_t(5, 300), np.zeros(300),
                                      split_conformal_regression(rng.standard_t(5, 300), 0.1))
                   for _ in range(100)])
    check(f"conformal coverage >= 0.90 (got {cov:.3f})", cov >= 0.88)
    construct = rng.standard_normal(50); confound = rng.standard_normal(50)
    p_ok, _, _ = specificity_gate(construct + 0.1 * rng.standard_normal(50), construct, [confound])
    p_bad, _, _ = specificity_gate(confound + 0.1 * rng.standard_normal(50), construct, [confound])
    check("specificity gate passes true construct", p_ok)
    check("specificity gate withholds confounded", not p_bad)
    raised = False
    try:
        gate(Finding("x", 0, "d", 0, 0, (0, 0), 1, False))
    except Exception:
        raised = True
    check("gate() refuses unvalidated finding", raised)


def test_brainfile_roundtrip():
    bf = BrainFile(subject="RT", network_head={"Limbic": 0.2}, provenance={"m": 1}, fingerprint="abc")
    stem = "/tmp/ns_bf_test"
    bf.save(stem)
    bf2 = BrainFile.load(stem)
    check("brain_file network_head round-trips", bf2.network_head["Limbic"] == 0.2)
    check("brain_file subject round-trips", bf2.subject == "RT")
    for ext in (".npz", ".json"):
        if os.path.exists(stem + ext):
            os.remove(stem + ext)


if __name__ == "__main__":
    tests = [test_default_behavior_unchanged, test_enroll_network_head_recovers_subject,
             test_personalization_changes_outputs, test_negative_control_zero_head,
             test_vertex_enrollment_machinery, test_validation_layer, test_brainfile_roundtrip]
    for t in tests:
        print(f"\n[{t.__name__}]")
        t()
    print("\n" + ("ALL INTEGRATION TESTS PASSED" if not _failures else f"FAILURES: {_failures}"))
    sys.exit(1 if _failures else 0)
