#!/usr/bin/env python3
"""Semantic layer demo — concept graph + idea trajectory + lens toggle + neuro bridge.

Shows the wider applicability: give it ANY text (ad, speech, lesson, PSA), get the linked
concepts, how they load on the neuro constructs, and how the idea moves over time — under a
chosen audience lens. Then the concept-derived network profile feeds the SAME buy/sell engine.

Run:  cd case-studies/semantic-demo && python3 run_semantic_demo.py
Out:  figures/concept_graph.png, figures/idea_trajectory.png, semantic_result.json

Honest: predicted semantic-affective map grounded in Warriner et al. (2013) affect norms — an
aggregate/population read-out, NOT mind-reading and NOT 'neurolinguistic programming'.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NS = HERE.parent.parent / "neurosignal"
sys.path.insert(0, str(NS))

from neurosignal.semantic import analyze_text, concept_trajectory  # noqa: E402
from neurosignal.neuroforecasting import neuroforecast              # noqa: E402

SPEECH = [
    "Imagine a future where your family is safe and free.",
    "Today there is fear. There is danger. There is a threat we cannot ignore.",
    "But together we can choose hope over fear, courage over doubt.",
    "This bold plan brings opportunity, reward, and real change for everyone.",
    "Believe in it. Fight for it. The future is ours to win.",
]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_graph(plt, sa):
    nodes = sa.graph["nodes"][:14]
    idx = {n: i for i, n in enumerate(nodes)}
    ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (np.cos(ang[i]), np.sin(ang[i])) for i, n in enumerate(nodes)}
    # color node by dominant construct polarity (reward=teal, conflict=red)
    cmap = {}
    for c in sa.concepts:
        if c["concept"] in idx and c.get("constructs"):
            k = c["constructs"]
            cmap[c["concept"]] = "#00798c" if k["reward_value"] >= k["conflict_risk"] else "#d1495b"
    fig, ax = plt.subplots(figsize=(7.5, 7.5)); ax.axis("off")
    for e in sa.graph["edges"]:
        if e["a"] in pos and e["b"] in pos:
            x = [pos[e["a"]][0], pos[e["b"]][0]]; y = [pos[e["a"]][1], pos[e["b"]][1]]
            ax.plot(x, y, "-", color="#ccc", lw=0.6 + 0.5 * e["w"], zorder=1)
    for n in nodes:
        ax.scatter(*pos[n], s=420, c=cmap.get(n, "#8d99ae"), edgecolor="white", zorder=2)
        ax.annotate(n, pos[n], fontsize=8, ha="center", va="center", color="white", zorder=3)
    ax.set_title("Concept network — linked concepts a stimulus evokes\n"
                 "(teal = reward-loaded, red = conflict-loaded)", fontsize=11)
    fig.text(0.5, 0.02, "Predicted semantic-affective map (Warriner 2013 norms). Not mind-reading.",
             ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(); fig.savefig(HERE / "figures" / "concept_graph.png", dpi=150); plt.close(fig)


def draw_trajectory(plt, traj):
    keys = ["reward_value_net", "conflict_net"]
    reward = [t["networks"]["Limbic"] for t in traj]
    conflict = [t["networks"]["VentralAttention"] for t in traj]
    attention = [t["networks"]["DorsalAttention"] for t in traj]
    x = np.arange(len(traj))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, reward, "-o", color="#00798c", label="reward (Limbic)")
    ax.plot(x, conflict, "-o", color="#d1495b", label="conflict (VentralAttention)")
    ax.plot(x, attention, "-o", color="#edae49", label="attention (DorsalAttention)")
    for t in traj:
        ax.annotate(", ".join(t["top_concepts"][:2]), (t["segment"], 0.02), fontsize=6.5,
                    rotation=30, ha="left", color="#555")
    ax.set_xticks(x); ax.set_xlabel("segment (time →)"); ax.set_ylabel("network activation")
    ax.set_title("Idea trajectory — how the concept-driven neural profile moves through the speech",
                 fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.5, 0.005, "Tracks the idea over time (encoding, predicted). The reproducible decoding "
             "twin is Tang et al. 2023 (noninvasive fMRI semantic decoder).",
             ha="center", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(HERE / "figures" / "idea_trajectory.png", dpi=150); plt.close(fig)


def main():
    (HERE / "figures").mkdir(exist_ok=True)
    full = " ".join(SPEECH)
    sa = analyze_text(full, lens="political_persuasion", top_k=16)
    traj = concept_trajectory(SPEECH, lens="political_persuasion", top_k=8)
    fc = neuroforecast(sa.networks, mode="aggregate", n_boot=1500)
    plt = _mpl()
    draw_graph(plt, sa); draw_trajectory(plt, traj)
    (HERE / "semantic_result.json").write_text(json.dumps({
        "lens": sa.lens, "coverage": sa.coverage, "networks": sa.to_dict()["networks"],
        "forecast": fc.to_dict(), "trajectory": traj}, indent=2))
    print(f"lens={sa.lens} coverage={sa.coverage:.2f} buy_sell={fc.buy_sell} CI={fc.ci95}")
    print("Wrote figures/concept_graph.png, figures/idea_trajectory.png, semantic_result.json")


if __name__ == "__main__":
    main()
