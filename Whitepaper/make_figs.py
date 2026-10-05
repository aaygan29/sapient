#!/usr/bin/env python3
"""Generate vector figures for the Sapient whitepaper. All numbers quoted from
the companion studies' result artifacts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

TEAL = "#2a9d8f"
SAND = "#e9c46a"
CORAL = "#e76f51"
SLATE = "#264653"
GREY = "#9aa0a6"

# ---------------------------------------------------------------------------
# Figure: 3-layer product stack (no recipe-level internals)
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, label, fc, sub="", tsize=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=1.2, edgecolor=SLATE, facecolor=fc, alpha=0.97)
        ax.add_patch(b)
        if sub:
            # title sits in the upper band, description fills the space below it
            ax.text(x + w/2, y + h - 0.30, label, ha="center", va="center",
                    fontsize=tsize, fontweight="bold", color=SLATE)
            ax.text(x + w/2, y + (h - 0.62)/2, sub, ha="center", va="center",
                    fontsize=8.0, color=SLATE, linespacing=1.35)
        else:
            ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                    fontsize=tsize, fontweight="bold", color=SLATE)

    box(0.5, 8.6, 9.0, 0.95, "Any stimulus:  video, audio, or text",
        "#f4f1de", sub="an ad, a message, a conversation, a film, the output of an AI", tsize=10.5)

    # three generic modality encoders (no named backbones)
    mods = ["Video\nunderstanding", "Audio\nunderstanding", "Language\nunderstanding"]
    bw = 2.7; gap = 0.45; x0 = 0.75
    for i, m in enumerate(mods):
        box(x0 + i*(bw+gap), 7.05, bw, 1.0, m, "#cfe8e3", tsize=9.5)

    box(0.5, 5.05, 9.0, 1.5, "MARY:  the engine", "#a9dcd3",
        sub="learns to predict the whole-brain response to any stimulus\nand to tell the shared, average response apart from each person's own\n=  a predicted brain  (a reading across 20,484 points on the cortex)", tsize=12.5)

    box(0.5, 3.35, 9.0, 1.3, "QUALIA:  the understanding layer", "#f3e2b3",
        sub="reads the predicted brain into plain language (which networks engaged)\npersonalizes it to a specific individual  ·  stays honest about what it can know", tsize=12.5)

    box(0.5, 1.65, 9.0, 1.3, "NEUROSIGNAL:  the read-out layer", "#f6cdbf",
        sub="reward · attention · emotion · Manipulation · Sycophancy · Buy / Sell\na live, second-by-second brain  ·  flags when it cannot measure something", tsize=12.5)

    box(0.5, 0.25, 9.0, 0.95, "Applications:  neuro-marketing · AI-safety auditing · personalized AI · clinical & BCI · governance",
        "#f4f1de", tsize=9.0)

    for y0, y1 in [(8.58, 8.12), (7.03, 6.58), (5.03, 4.68), (3.33, 2.98), (1.63, 1.22)]:
        ax.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.5, color=SLATE))
    ax.text(9.62, 4.6, "Mary improves\nso the whole\nproduct improves",
            ha="left", va="center", fontsize=8.0, color=TEAL, fontweight="bold")
    fig.savefig("figs/fig_architecture.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: per-subject vs average (the headline result)
# ---------------------------------------------------------------------------
def fig_persubject():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    data = {
        "Story (22 people, audio + text)": dict(vals=[0.0132, 0.0035, 0.0035],
                                                err=[[0.0044, 0.0011, 0.0012], [0.0041, 0.0012, 0.0011]],
                                                ceiling=0.0088, ratio="3.7x"),
        "Film (4 people, audio + video)": dict(vals=[0.0157, 0.0052, 0.0053],
                                               err=[[0.0038, 0.0013, 0.0013], [0.0041, 0.0013, 0.0014]],
                                               ceiling=0.0533, ratio="3.1x"),
    }
    labels = ["personal\nbrain", "average\nbrain", "shared\nonly"]
    colors = [TEAL, SAND, CORAL]
    for ax, (title, d) in zip(axes, data.items()):
        x = np.arange(3)
        ax.bar(x, d["vals"], yerr=d["err"], color=colors, capsize=3, width=0.62,
               error_kw=dict(elinewidth=1, capthick=1))
        ax.axhline(d["ceiling"], ls="--", color=GREY, lw=1.1)
        ax.text(2.45, d["ceiling"], "  best a model could do", color=GREY,
                fontsize=7, va="bottom", ha="right")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("accuracy (higher is better)", fontsize=8)
        ax.annotate(d["ratio"], xy=(0, d["vals"][0]), xytext=(0.0, d["vals"][0]*1.05),
                    ha="center", fontsize=10, fontweight="bold", color=TEAL)
    fig.suptitle("Same model, same content:  knowing WHO the person is roughly triples accuracy",
                 fontsize=9.5, y=1.03)
    fig.tight_layout()
    fig.savefig("figs/fig_persubject.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: the four convergence results
# ---------------------------------------------------------------------------
def fig_convergence():
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.8))

    ax = axes[0]
    ax.bar([0, 1], [3.4, 1.0], color=[TEAL, GREY], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["personal", "average"], fontsize=8)
    ax.set_ylabel("relative accuracy", fontsize=8)
    ax.set_title("A.  The person is\nthe signal\n(3x better)", fontsize=8.5)
    ax.set_ylim(0, 4)

    ax = axes[1]
    nets = ["1", "2", "3", "4", "5", "6", "7"]
    vals = [-0.010, -0.016, -0.013, -0.018, -0.011, -0.015, -0.017]
    ax.bar(range(7), vals, color=CORAL, width=0.7)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xticks(range(7)); ax.set_xticklabels(nets, fontsize=7)
    ax.set_xlabel("brain network", fontsize=7.5)
    ax.set_ylabel("agreement with a real person", fontsize=7.3)
    ax.set_title("B.  The average\ngets it backwards\n(all 7 networks)", fontsize=8.5)

    ax = axes[2]
    ax.bar([0, 1, 2], [85, 15, 5], color=[TEAL, CORAL, GREY], width=0.6)
    ax.axhline(5, color=GREY, ls=":", lw=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["real\nbrain", "average", "guess"], fontsize=7.5)
    ax.set_ylabel("correct identifications %", fontsize=8)
    ax.set_title("C.  The pipeline works,\nthe average can't\n(85% vs chance)", fontsize=8.5)

    ax = axes[3]
    ax.bar([0, 1, 2], [0.116, 0.156, 0.22], color=[SAND, TEAL, SLATE], width=0.6)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["model", "model+", "real"], fontsize=7.5)
    ax.set_ylabel("match to ground truth", fontsize=8)
    ax.set_title("D.  A predicted brain\nis readable\n(matches reality)", fontsize=8.5)

    fig.suptitle("Four separate studies, one conclusion:  to understand a person, model the person",
                 fontsize=9.2, y=1.05)
    fig.tight_layout()
    fig.savefig("figs/fig_convergence.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: personalization, same ad, three brains
# ---------------------------------------------------------------------------
def fig_personalization():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    people = ["Average\nbrain", "Reward-seeker", "Deliberator"]
    manip = [6, 87, 0]
    colors = [GREY, CORAL, TEAL]
    bars = ax.bar(people, manip, color=colors, width=0.55)
    for b, v in zip(bars, manip):
        ax.text(b.get_x() + b.get_width()/2, v + 2, str(v), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Manipulation score", fontsize=9.5)
    ax.set_title('The same ad,  "Act now, this exclusive deal disappears in minutes!"\n'
                 'lands very differently on each person\'s brain', fontsize=9.5)
    fig.tight_layout()
    fig.savefig("figs/fig_personalization.pdf")
    plt.close(fig)

if __name__ == "__main__":
    fig_architecture()
    fig_persubject()
    fig_convergence()
    fig_personalization()
    print("figures written")
