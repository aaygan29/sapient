#!/usr/bin/env python3
"""Retrospective ad case study — HONEST descriptor pilot.

Feeds transparent Yeo-7 network descriptors of famous ads (ad_descriptors.json)
through the REAL local neurosignal construct engine and plots the predicted
approach / buy-sell score against publicly-reported outcome magnitudes.

Honesty contract (mirrors neurosignal/CHAIN_AUDIT_2026-07-16.md):
  * These are PREDICTED construct scores from expert descriptors, not measured brains.
  * n is small and outcomes are heterogeneous, approximate, publicly-reported.
  * The correlation is ILLUSTRATIVE of the product's story shape, NOT validation of
    the predicted-brain -> behavior link ("Link 4"). Every figure says so.

Run:
    cd case-studies && python3 retrospective_ad_pilot.py
Outputs: figures/scatter_approach_vs_outcome.png, figures/construct_breakdown.png,
         results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Use the real engine from the sibling neurosignal package.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "neurosignal"))
from neurosignal.detect import detect_from_networks  # noqa: E402

CAVEAT = ("ILLUSTRATIVE machinery demo — predicted scores on SELF-AUTHORED descriptors "
          "(circular), n=7 SURVIVORS only (no flops), INCOMMENSURABLE outcomes pooled. "
          "The rho is NOT an interpretable effect size. Real test: validation/. NOT Link 4.")


def load_ads() -> list[dict]:
    data = json.loads((HERE / "ad_descriptors.json").read_text())
    return data["ads"]


def score_ads(ads: list[dict]) -> list[dict]:
    """normalize=False so the 0..1 descriptor axis stays comparable ACROSS ads
    (single-stimulus min-max would rescale each ad independently and destroy the
    cross-ad comparison the case study needs)."""
    rows = []
    for ad in ads:
        r = detect_from_networks(ad["networks"], source=ad["name"], normalize=False)
        approach = float(np.mean([c.score for c in r.constructs
                                  if c.covered and c.key in
                                  ("reward_value", "emotional_resonance",
                                   "attention_capture", "memory_encoding")]))
        rows.append({
            "name": ad["name"], "category": ad["category"],
            "buy_sell": round(r.buy_sell_score, 1),
            "approach": round(approach, 1),
            "confidence": round(r.confidence, 2),
            "outcome": ad["outcome_magnitude"],
            "outcome_kind": ad["outcome_kind"],
            "constructs": {c.key: c.score for c in r.constructs if c.covered},
        })
    return rows


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def make_figures(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    real = [r for r in rows if r["category"] != "control"]  # rho on real campaigns only
    x = np.array([r["approach"] for r in real])
    y = np.array([r["outcome"] for r in real])
    rho = spearman(x, y)

    # --- Figure 1: scatter approach vs outcome ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"political": "#d1495b", "brand": "#00798c", "psa": "#edae49",
              "control": "#8d99ae"}
    for r in rows:
        ax.scatter(r["approach"], r["outcome"], s=140,
                   c=colors.get(r["category"], "#555"),
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(r["name"].split(" (")[0], (r["approach"], r["outcome"]),
                    fontsize=7.5, xytext=(6, 4), textcoords="offset points")
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m * xs + b, "--", color="#333", linewidth=1, zorder=2)
    ax.set_xlabel("Predicted approach score (neurosignal constructs, 0–100)")
    ax.set_ylabel("Publicly-reported outcome magnitude (normalized)")
    ax.set_title(f"Predicted approach vs known outcome  ·  Spearman ρ = {rho:.2f}  (n={len(real)})",
                 fontsize=11)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                          markerfacecolor=c, markeredgecolor="white", label=k)
               for k, c in colors.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
    fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=7, color="#b00020")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(figdir / "scatter_approach_vs_outcome.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: per-ad construct breakdown ---
    keys = ["reward_value", "emotional_resonance", "attention_capture",
            "memory_encoding", "conflict_risk", "cognitive_load"]
    labels = [k.replace("_", " ") for k in keys]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [r["name"].split(" (")[0] for r in rows]
    xpos = np.arange(len(names))
    width = 0.13
    palette = ["#00798c", "#d1495b", "#edae49", "#66a182", "#8d5a99", "#8d99ae"]
    for i, (k, lab) in enumerate(zip(keys, labels)):
        vals = [r["constructs"].get(k, 0.0) for r in rows]
        ax.bar(xpos + (i - 2.5) * width, vals, width, label=lab, color=palette[i])
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Construct score (0–100)")
    ax.set_title("Per-ad construct fingerprint (predicted)", fontsize=11)
    ax.legend(fontsize=8, ncol=3, frameon=False)
    fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=7, color="#b00020")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(figdir / "construct_breakdown.png", dpi=150)
    plt.close(fig)

    return rho


def main() -> None:
    ads = load_ads()
    rows = score_ads(ads)
    rho = make_figures(rows)
    (HERE / "results.json").write_text(json.dumps(
        {"spearman_rho_real_campaigns": rho, "n_real": len([r for r in rows if r["category"] != "control"]),
         "rows": rows, "caveat": CAVEAT}, indent=2))
    print(f"Scored {len(rows)} ads. Spearman rho (real campaigns) = {rho:.2f}")
    print("Wrote figures/scatter_approach_vs_outcome.png, figures/construct_breakdown.png, results.json")
    print(CAVEAT)


if __name__ == "__main__":
    main()
