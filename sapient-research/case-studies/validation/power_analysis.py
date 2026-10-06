#!/usr/bin/env python3
"""Power analysis — is the validation study feasible, and at what n?

Uses the Fisher z-transform to compute statistical power to detect a true correlation of size
rho at sample size n (two-sided, alpha=0.05), and the minimum detectable effect size (MDES) at
80% power. Anchored to published neuroforecasting effect sizes so the numbers are realistic:

  * AGGREGATE / market forecasting (the Sapient use case) — Genevsky & Knutson report NAcc
    predicting aggregate outcomes with correlations commonly in the rho ~ 0.5-0.8 range
    (crowdfunding, microlending). We use rho=0.5 as a conservative aggregate anchor.
  * INDIVIDUAL choice — weaker, rho ~ 0.2-0.3 (why the product forecasts audiences, not people).

Output: figures/power_curves.png + a printed table. Pure numpy (normal approx).

Run:  python3 power_analysis.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Z_ALPHA = 1.959963985  # two-sided 0.05
Z_POWER = 0.8416212336  # 80%


def _phi(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Spearman's rank correlation has ~(1 + rho^2/2) larger sampling variance than Pearson
# (Fieller/Bonett-Wright). We inflate the SE so the power/n numbers are for SPEARMAN, which is
# what the harness actually computes — Fisher-z on Pearson alone would overstate power.
def _spearman_se_factor(rho):
    return math.sqrt(1.0 + rho * rho / 2.0)


def power_for(rho, n):
    """Power to reject rho=0 given true rho at sample n (Fisher z, Spearman-adjusted SE).

    Normal approximation; assumes an approximately monotone bivariate relationship. For small n
    (<20) treat as indicative, not exact."""
    if n <= 3 or rho == 0:
        return 0.05
    z = 0.5 * math.log((1 + rho) / (1 - rho))
    se = _spearman_se_factor(rho) / math.sqrt(n - 3)
    # power = P(|Z| > z_alpha) under shifted mean z/se
    lam = abs(z) / se
    return _phi(lam - Z_ALPHA) + _phi(-lam - Z_ALPHA)


def n_for(rho, power=0.8):
    """Sample size for target power to detect true rho (Spearman-adjusted)."""
    if rho == 0:
        return math.inf
    z = 0.5 * math.log((1 + rho) / (1 - rho))
    return math.ceil((_spearman_se_factor(rho) * (Z_ALPHA + Z_POWER) / abs(z)) ** 2 + 3)


def mdes(n, power=0.8):
    """Minimum detectable rho at sample n and target power (Spearman-adjusted, solved by bisection
    since the SE factor depends on rho)."""
    if n <= 3:
        return 1.0
    lo, hi = 1e-4, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        z = 0.5 * math.log((1 + mid) / (1 - mid))
        need = (_spearman_se_factor(mid) * (Z_ALPHA + Z_POWER) / z) ** 2 + 3
        if need > n:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def main():
    anchors = {"aggregate (ρ=0.5, Genevsky-Knutson)": 0.5,
               "moderate (ρ=0.4)": 0.4,
               "individual (ρ=0.25)": 0.25}
    ns = np.arange(10, 205, 5)
    print("Required n for 80% power:")
    for label, rho in anchors.items():
        print(f"  {label:38s} n = {n_for(rho):3d}")
    print("\nMDES (detectable ρ at 80% power):")
    for n in (30, 60, 100, 150):
        print(f"  n={n:3d}  ->  ρ >= {mdes(n):.2f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, rho in anchors.items():
        ax1.plot(ns, [power_for(rho, n) for n in ns], lw=2, label=label)
    ax1.axhline(0.8, ls=":", c="#333"); ax1.set_ylim(0, 1)
    ax1.set_xlabel("n (ads or aggregated segments)"); ax1.set_ylabel("power")
    ax1.set_title("Power vs sample size", fontsize=10); ax1.legend(fontsize=7.5, frameon=False)
    ax2.plot(ns, [mdes(n) for n in ns], lw=2, c="#00798c")
    ax2.axhline(0.5, ls=":", c="#8d99ae"); ax2.set_xlabel("n"); ax2.set_ylabel("min detectable ρ")
    ax2.set_title("Minimum detectable effect (80% power)", fontsize=10)
    fig.suptitle("Validation study is feasible: aggregate-scale effects need modest n", fontsize=11)
    fig.text(0.5, 0.005, "At the published aggregate effect size (ρ≈0.5) the study reaches 80% power "
             f"by n≈{n_for(0.5)}. A ~$500 Prolific study (n=100) clears it comfortably.",
             ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "power_curves.png", dpi=150); plt.close(fig)
    print("\nWrote figures/power_curves.png")


if __name__ == "__main__":
    main()
