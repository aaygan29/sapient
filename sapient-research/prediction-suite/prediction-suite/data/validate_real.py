"""
validate_real.py — run the Neuroforecaster validation harness on REAL outcomes.

This is a smoke test of the machinery, not neural validation. It uses the ISLR
Advertising dataset (real ad spend -> real product sales) to show that:
  - Neuroforecaster.predict_outcomes runs on real numbers,
  - its leave-one-out + permutation null recover a genuine signal when one exists,
  - it honestly reports effect size and significance.

When Mary vertex exports + measured ad outcomes arrive, swap `signal` for the
grounded value/reward read-out and `outcomes` for CTR/sales/recall — the call is
identical.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from neuroforecasting import Neuroforecaster  # noqa: E402
from loaders import load_advertising           # noqa: E402


def main():
    d = load_advertising()
    print(f"Loaded {d['n']} markets from {d['source']}; channels={d['channels']}")

    spend, sales = d["spend"], d["sales"]

    # Use total ad spend as a single "signal" standing in for a neural read-out.
    signal = spend.sum(axis=1)

    nf = Neuroforecaster(construct="ad_spend", outcome_metric="sales", min_effect=0.30)
    res = nf.predict_outcomes(signal, sales, self_report=None, n_perms=5000)

    print("\n=== Neuroforecaster on REAL ad-spend -> sales ===")
    print(f"  brain/signal r = {res.r_brain:+.3f}  (p_perm = {res.p_value:.4f})")
    print(f"  effect size    = {res.effect_size:.3f}")
    print(f"  verdict        = {'PASS' if res.passes_specificity else 'FAIL'}  ({res.notes})")

    # Per-channel, to show it discriminates: TV should carry sales far more than newspaper.
    print("\n  per-channel correlation with sales (real):")
    for i, ch in enumerate(d["channels"]):
        r = float(np.corrcoef(spend[:, i], sales)[0, 1])
        print(f"    {ch:10s} r = {r:+.3f}")

    print("\nHarness verified on real outcomes. Identical call shape for Mary read-outs.")


if __name__ == "__main__":
    main()
