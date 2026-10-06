"""Dependency-light tests for neuro_ai_core (run: python3 tests/test_core.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_ai_core import (  # noqa: E402
    Finding, gate, UnvalidatedClaimError,
    split_conformal_regression, empirical_coverage,
    specificity_gate, Provenance,
)

_failures = []


def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    if not cond:
        _failures.append(name)


def test_finding():
    good = Finding("r", 0.013, "sim", 0.0035, 1.2, (0.009, 0.017), 22, True)
    bad = Finding("r", 0.001, "sim", 0.0035, 0.1, (-0.01, 0.01), 22, False)
    check("Finding PASS repr", "PASS" in repr(good))
    check("Finding UNVALIDATED repr", "UNVALIDATED" in repr(bad))
    gate(good)  # should not raise
    raised = False
    try:
        gate(good, bad)
    except UnvalidatedClaimError:
        raised = True
    check("gate() raises on unvalidated", raised)


def test_conformal_coverage():
    # Heteroskedastic residuals; conformal should hit >= nominal coverage on fresh data.
    rng = np.random.default_rng(0)
    alpha = 0.10
    covs = []
    for _ in range(200):
        cal = rng.standard_t(df=5, size=400) * 1.0
        test = rng.standard_t(df=5, size=400) * 1.0
        q = split_conformal_regression(cal, alpha=alpha)
        covs.append(empirical_coverage(test, np.zeros_like(test), q))
    mean_cov = float(np.mean(covs))
    check(f"conformal coverage >= nominal (got {mean_cov:.3f}, target {1-alpha})",
          mean_cov >= (1 - alpha) - 0.02)
    check(f"conformal not absurdly over-covering (got {mean_cov:.3f})", mean_cov <= 0.97)


def test_specificity_gate():
    rng = np.random.default_rng(1)
    construct = rng.standard_normal(200)
    confound = rng.standard_normal(200)
    # A contrast that IS the construct -> pass.
    passed_c, ratio_c, _ = specificity_gate(construct + 0.1 * rng.standard_normal(200),
                                            construct, [confound])
    # A contrast that IS the confound -> fail (the Mary failure mode).
    passed_f, ratio_f, _ = specificity_gate(confound + 0.1 * rng.standard_normal(200),
                                            construct, [confound])
    check(f"gate passes a true-construct contrast (ratio={ratio_c:.2f})", passed_c)
    check(f"gate withholds a confounded contrast (ratio={ratio_f:.2f})", not passed_f)


def test_provenance():
    p1 = Provenance("mary", "abc123", 42, "simulated", 5.0, "calib_movie")
    p2 = Provenance("mary", "abc123", 43, "simulated", 5.0, "calib_movie")
    check("fingerprint deterministic", p1.fingerprint() == p1.fingerprint())
    check("fingerprint changes with seed", p1.fingerprint() != p2.fingerprint())
    check("to_dict carries fingerprint", "fingerprint" in p1.to_dict())


if __name__ == "__main__":
    for t in (test_finding, test_conformal_coverage, test_specificity_gate, test_provenance):
        print(f"\n[{t.__name__}]")
        t()
    print("\n" + ("ALL CORE TESTS PASSED" if not _failures else f"FAILURES: {_failures}"))
    sys.exit(1 if _failures else 0)
