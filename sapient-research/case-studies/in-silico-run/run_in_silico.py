#!/usr/bin/env python3
"""End-to-end in-silico case-study run.

Full chain, per ad:
    descriptor (Yeo-7)  ->  neurosignal constructs (detect)  ->  neuroforecast composite
    (aggregate mode, bootstrap 95% CI), scored under BOTH references:
      * coordinate-grounded  (published Knutson/Genevsky peaks -> Schaefer/Yeo)
      * empirical-fMRI       (NeuroVault mixed-gambles gain/loss group T maps -> Schaefer/Yeo)

Outputs (this folder):
    figures/insilico_buy_sell_ci.png       per-ad buy/sell with 95% CI (empirical reference)
    figures/insilico_ref_comparison.png    coordinate vs empirical buy/sell
    results_in_silico.json                 machine-readable
    REPORT.md                              the write-up (regenerated)

Honesty: predicted-activation read-outs on expert descriptors, n small. The empirical
reference is real group fMRI but cortical-only (no subcortical NAcc). NOT validation of Link 4.

Run:  cd case-studies/in-silico-run && python3 run_in_silico.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CS = HERE.parent
NS = CS.parent / "neurosignal"
sys.path.insert(0, str(NS))

from neurosignal.detect import detect_from_networks          # noqa: E402
from neurosignal.neuroforecasting import neuroforecast, load_reference, as_finding  # noqa: E402

COORD_REF = load_reference()  # default coordinate-grounded
EMP_PATH = NS / "neurosignal" / "data" / "neuroforecasting_reference_mid_empirical.json"
EMP_REF = json.loads(EMP_PATH.read_text())

CAVEAT = ("In-silico: predicted read-outs on expert descriptors (n small). Empirical reference "
          "is real group fMRI (NeuroVault mixed-gambles) but cortical-only. NOT validation of Link 4.")


def load_ads():
    return json.loads((CS / "ad_descriptors.json").read_text())["ads"]


def run():
    ads = load_ads()
    rows = []
    for ad in ads:
        det = detect_from_networks(ad["networks"], source=ad["name"], normalize=False)
        emp = neuroforecast(ad["networks"], mode="aggregate", reference=EMP_REF, n_boot=2000)
        crd = neuroforecast(ad["networks"], mode="aggregate", reference=COORD_REF, n_boot=2000)
        rows.append({
            "name": ad["name"], "category": ad["category"],
            "detect_buy_sell": det.buy_sell_score,
            "emp_buy_sell": emp.buy_sell, "emp_ci": list(emp.ci95),
            "emp_rec": emp.recommendation, "emp_finding_passed": as_finding(emp).passed,
            "coord_buy_sell": crd.buy_sell, "coord_ci": list(crd.ci95),
            "generalizability": emp.generalizability,
        })
    return rows


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = HERE / "figures"; fig_dir.mkdir(exist_ok=True)
    order = sorted(rows, key=lambda r: r["emp_buy_sell"])
    names = [r["name"].split(" (")[0] for r in order]
    y = np.arange(len(order))
    vals = np.array([r["emp_buy_sell"] for r in order])
    lo = vals - np.array([r["emp_ci"][0] for r in order])
    hi = np.array([r["emp_ci"][1] for r in order]) - vals

    # Fig 1: buy/sell with CI, empirical reference
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cols = ["#00798c" if r["emp_rec"] == "Buy" else "#d1495b" if r["emp_rec"] == "Sell"
            else "#8d99ae" for r in order]
    ax.barh(y, vals, color=cols, edgecolor="white")
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor="#222", elinewidth=1.2, capsize=3)
    ax.axvline(60, ls=":", c="#00798c", lw=1); ax.axvline(40, ls=":", c="#d1495b", lw=1)
    ax.axvline(50, ls="-", c="#999", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("Buy/Sell composite (aggregate, empirical fMRI reference)")
    ax.set_title("In-silico case-study run — buy/sell with bootstrap 95% CI", fontsize=11)
    fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=6.5, color="#b00020")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(fig_dir / "insilico_buy_sell_ci.png", dpi=150); plt.close(fig)

    # Fig 2: coordinate vs empirical
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        ax.scatter(r["coord_buy_sell"], r["emp_buy_sell"], s=120, c="#00798c",
                   edgecolor="white", zorder=3)
        ax.annotate(r["name"].split(" (")[0], (r["coord_buy_sell"], r["emp_buy_sell"]),
                    fontsize=7, xytext=(5, 3), textcoords="offset points")
    lim = [0, 100]; ax.plot(lim, lim, "--", c="#999", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Buy/Sell — coordinate-grounded reference")
    ax.set_ylabel("Buy/Sell — empirical fMRI reference")
    ax.set_title("Reference sensitivity: coordinate vs empirical", fontsize=11)
    fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=6.5, color="#b00020")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(fig_dir / "insilico_ref_comparison.png", dpi=150); plt.close(fig)


def report(rows):
    passed = sum(r["emp_finding_passed"] for r in rows)
    lines = [
        "# In-silico case-study run — report",
        "",
        "Full chain executed per ad: **descriptor → neurosignal constructs → neuroforecast "
        "aggregate composite (bootstrap 95% CI)**, under both the coordinate-grounded and the "
        "real empirical-fMRI reference.",
        "",
        f"- Ads scored: **{len(rows)}**",
        f"- Findings passing the honesty gate (CI excludes chance): **{passed}/{len(rows)}**",
        "",
        "![buy/sell with CI](figures/insilico_buy_sell_ci.png)",
        "",
        "![coordinate vs empirical](figures/insilico_ref_comparison.png)",
        "",
        "## Per-ad",
        "",
        "| Ad | Rec | Buy/Sell (empirical) | 95% CI | Buy/Sell (coord) | Gate |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["emp_buy_sell"]):
        ci = f"[{r['emp_ci'][0]:.0f}, {r['emp_ci'][1]:.0f}]"
        gate = "PASS" if r["emp_finding_passed"] else "abstain"
        lines.append(f"| {r['name']} | {r['emp_rec']} | {r['emp_buy_sell']:.1f} | {ci} | "
                     f"{r['coord_buy_sell']:.1f} | {gate} |")
    lines += [
        "",
        "## Honest reading",
        "- The **empirical** reference is real group fMRI (NeuroVault mixed-gambles gain/loss T "
        "maps) but **cortical-only** — Schaefer-2018 has no subcortical NAcc, so the strongest "
        "reward node is proxied by cortex. Empirical and coordinate references therefore disagree "
        "for some ads (see comparison figure); that gap is the honest uncertainty in a cortical "
        "proxy, not a bug.",
        "- Every score is a **predicted** read-out on an expert descriptor. This run demonstrates "
        "the machinery end-to-end with calibrated intervals; it does **not** validate the "
        "predicted-brain → behavior link (see `../../neurosignal/CHAIN_AUDIT_2026-07-16.md`).",
        "- Next: swap the empirical reference from cortical group maps to a subcortical-inclusive "
        "MID contrast, and run Case Study B (Prolific) for real outcomes.",
        "",
        f"> {CAVEAT}",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))


def main():
    rows = run()
    figures(rows)
    report(rows)
    (HERE / "results_in_silico.json").write_text(json.dumps({"rows": rows, "caveat": CAVEAT}, indent=2))
    passed = sum(r["emp_finding_passed"] for r in rows)
    print(f"In-silico run complete: {len(rows)} ads, {passed} pass the honesty gate.")
    print("Wrote figures/, REPORT.md, results_in_silico.json")


if __name__ == "__main__":
    main()
