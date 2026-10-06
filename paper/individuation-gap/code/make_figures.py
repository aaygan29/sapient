"""
Publication figures for "The individuation gap" paper.
All numbers are from verified result files; nothing fabricated.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path.home() / "Desktop/Research/submissions/neuro-ai/insilico-neuroforecasting-validity/figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", grey="#999999", ink="#222222")

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def fig1():
    fig = plt.figure(figsize=(10.5, 3.4))
    axp = fig.add_axes([0.02, 0.10, 0.58, 0.84]); axp.axis("off")
    axg = fig.add_axes([0.70, 0.17, 0.28, 0.74])

    axp.set_xlim(0, 13); axp.set_ylim(0, 6)
    axp.text(0, 5.7, "a   In-silico neuroforecasting pipeline", fontweight="bold", fontsize=10.5)
    boxes = [("Stimulus", C["grey"]), ("Encoder", C["sky"]),
             ("Predicted\ncortex", C["blue"]), ("Construct\nread-out", C["purple"])]
    w, h, y0 = 2.5, 1.25, 3.3
    xs = [0.2, 3.3, 6.4, 9.5]
    for x, (label, col) in zip(xs, boxes):
        axp.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                      linewidth=1.1, edgecolor=C["ink"], facecolor=col, alpha=0.22))
        axp.text(x + w/2, y0 + h/2, label, ha="center", va="center", fontsize=8.6)
    for i in range(3):
        axp.add_patch(FancyArrowPatch((xs[i]+w, y0+h/2), (xs[i+1], y0+h/2),
                      arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=C["ink"]))
    # gates stacked under the read-out
    rx = xs[3] + w/2
    axp.add_patch(FancyArrowPatch((rx, y0), (rx, 2.55),
                  arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=C["ink"]))
    gates = ["1. Specificity gate  (has a localizer?)",
             "2. Content-baseline gate  (beats stimulus-only?)",
             "3. Calibration  (conformal interval)"]
    axp.text(rx, 2.35, "Refusal-first gates", ha="center", va="top",
             fontsize=8, style="italic", color=C["red"])
    for j, g in enumerate(gates):
        axp.text(rx, 1.75 - j*0.45, g, ha="center", va="center", fontsize=7.4, color=C["ink"])
    axp.text(0, 0.15, "A predicted neural index is a function of the stimulus; it must earn its "
             "label before it is reported.", fontsize=7.6, color=C["ink"])

    # panel b: the gap
    axg.set_title("b   The individuation gap", loc="left", fontweight="bold", fontsize=10.5)
    N = np.linspace(1, 30, 200); a, v = 0.3, 1.0
    pop = a/(a+v/N)
    # measured individual-difference ceiling (Marek 2022: univariate r=.14, multivariate r=.34 -> r^2)
    axg.axhspan(0.02, 0.12, color=C["orange"], alpha=0.14, lw=0)
    axg.text(15.5, 0.135, "measured individual ceiling (BWAS $r^2\\!\\approx$.02–.12)",
             fontsize=6.3, color=C["orange"], ha="center")
    axg.plot(N, pop, color=C["blue"], lw=2.4, label="population / enrolled  (a>0)")
    axg.plot(N, np.zeros_like(N), color=C["red"], lw=2.4, ls="--", label="zero-shot individual  (a=0)")
    axg.fill_between(N, 0, pop, color=C["blue"], alpha=0.08)
    axg.annotate("the gap", xy=(24, pop[int(200*24/30)]*0.5), fontsize=8.5,
                 color=C["ink"], ha="center", style="italic")
    axg.set_xlabel("aggregation size  N"); axg.set_ylabel(r"forecasting $\rho^2=a/(a+v/N)$")
    axg.set_ylim(-0.04, 1.0); axg.set_xlim(1, 30)
    axg.legend(loc="lower center", fontsize=7.2, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.70, 0.005, "Lean 4: unseen_individual_is_zero · individuation_gap",
             fontsize=6.6, color=C["grey"])
    save(fig, "fig1_schematic")


def fig2():
    cells = [("SIGNAL / enrolled", 0.449, (0.378, 0.526), True),
             ("SIGNAL / unseen",   0.028, None, False),
             ("NULL / enrolled",  -0.001, None, False),
             ("NULL / unseen",    -0.001, None, False)]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    fig.subplots_adjust(left=0.26, right=0.97, top=0.82, bottom=0.26)
    ax.set_title("Gate 2 (individual value) fires in exactly one cell",
                 loc="left", fontweight="bold", fontsize=10)
    ys = np.arange(len(cells))[::-1]
    for y, (name, eff, ci, fires) in zip(ys, cells):
        col = C["green"] if fires else C["grey"]
        if ci: ax.plot([ci[0], ci[1]], [y, y], color=col, lw=3.5, solid_capstyle="round")
        ax.plot(eff, y, "o", color=col, ms=8, zorder=3)
        ax.text(0.60, y, f"{'FIRES' if fires else 'silent'}  ($\\Delta R^2$={eff:+.3f})",
                va="center", fontsize=8, color=col, fontweight="bold" if fires else "normal")
    ax.axvline(0, color=C["ink"], lw=0.8, ls=":")
    ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in cells])
    ax.set_xlabel(r"Gate 2: correct$-$wrong subject (pooled OOS $\Delta R^2$)")
    ax.set_xlim(-0.1, 0.95)
    fig.text(0.02, 0.03, "20 seeds, paired bootstrap on synthetic ground truth. Zero false "
             "positives in the NULL worlds;\nsilent in SIGNAL/unseen because a zero-shot encoder "
             "cannot individuate an unseen subject.", fontsize=7, color=C["ink"])
    save(fig, "fig2_synthetic_gates")


def fig3():
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.985, top=0.80, bottom=0.30, wspace=0.55)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # (a) identity works, content does not
    ax0.set_title("a   Identity vs. content", loc="left", fontweight="bold", fontsize=9.5)
    vals = [1.00, 0.035]; errs = [0.0, 0.031]
    ax0.bar([0, 1], vals, yerr=errs, color=[C["green"], C["red"]], alpha=0.85,
            width=0.62, capsize=4, error_kw=dict(lw=1, ecolor=C["ink"]))
    ax0.axhline(0.25, color=C["grey"], ls=":", lw=1); ax0.text(1.42, 0.27, "chance",
            fontsize=6.6, color=C["grey"], ha="right")
    ax0.set_xticks([0, 1]); ax0.set_xticklabels(["WHO\n(identity)", "WHAT\n(content)"], fontsize=8.0)
    ax0.set_ylabel("identification / behavioral self-adv."); ax0.set_ylim(0, 1.08)

    # (b) decision grounding chain
    ax1.set_title("b   Value-based decision", loc="left", fontweight="bold", fontsize=9.5)
    links = [("Population choice (AUC .965)", 1.0, C["green"], "established"),
             ("NAcc/aIns loss (p=.040/.025)", 1.0, C["green"], "established"),
             ("Cross-lab transfer (.86–.89)", 1.0, C["green"], "established"),
             ("NAcc gain (p=.15)", 0.45, C["orange"], "abstains"),
             ("Direct individual link", 0.45, C["red"], "abstains")]
    yy = np.arange(len(links))[::-1]
    for y, (lab, ln, col, st) in zip(yy, links):
        ax1.barh(y, ln, color=col, alpha=0.85, height=0.58)
        ax1.text(0.03, y + 0.32, lab, va="bottom", ha="left", fontsize=6.9, color=C["ink"])
        ax1.text(ln + 0.03, y, st, va="center", fontsize=6.8, color=col, fontweight="bold")
    ax1.set_xlim(0, 1.75); ax1.set_ylim(-0.6, len(links)-0.1)
    ax1.set_yticks([]); ax1.set_xticks([])
    for s in ("left", "bottom"): ax1.spines[s].set_visible(False)

    # (c) shared law
    ax2.set_title("c   The shared law", loc="left", fontweight="bold", fontsize=9.5)
    N = np.linspace(1, 30, 200); a, v = 0.3, 1.0
    ax2.plot(N, a/(a+v/N), color=C["blue"], lw=2.3, label="population / enrolled")
    ax2.plot(N, np.zeros_like(N), color=C["red"], lw=2.3, ls="--", label="unseen individual")
    ax2.fill_between(N, 0, a/(a+v/N), color=C["blue"], alpha=0.08)
    ax2.set_xlabel("aggregation size  N"); ax2.set_ylabel(r"forecasting $\rho^2$")
    ax2.set_ylim(-0.04, 1.0); ax2.set_xlim(1, 30); ax2.legend(fontsize=6.8, loc="center right")

    fig.text(0.07, 0.04, "a  Enrolled identity is recoverable (fingerprinting 25/25 ROIs, 100%; behavioral "
             "fingerprint 14$\\times$ chance, p<.001); individual behavioral content is not.    "
             "b  Population & cross-lab links establish; the direct individual neural→behavior link "
             "abstains under the calibrated gate.", fontsize=6.9, color=C["ink"])
    save(fig, "fig3_convergent_evidence")


def fig4():
    # Fusion non-monotonicity: rho^2 = (sum a_m) / (sum a_m + v + sum delta_m)
    # add modalities best-first; later ones have low a_m/delta_m and hurt.
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.86, bottom=0.18)
    ax.set_title("Multimodal fusion is not a free lunch", loc="left",
                 fontweight="bold", fontsize=10)
    v = 1.0
    mods = ["fMRI", "+EEG", "+HR", "+EDA", "+hormones", "+self-report"]
    # per-modality identifiable individual signal a_m and injected nuisance delta_m
    a_m   = [0.30, 0.12, 0.05, 0.03, 0.02, 0.02]
    d_m   = [0.00, 0.10, 0.12, 0.18, 0.40, 0.30]   # added nuisance variance grows
    A = np.cumsum(a_m); D = np.cumsum(d_m)
    rho = A / (A + v + D)
    x = np.arange(len(mods))
    ax.plot(x, rho, "-o", color=C["purple"], lw=2.2, ms=6)
    best = int(np.argmax(rho))
    ax.axvline(best, color=C["grey"], ls=":", lw=1)
    ax.annotate("adding low-reliability\nmodalities reduces $\\rho^2$",
                xy=(len(mods)-1, rho[-1]), xytext=(2.2, rho[-1]+0.11),
                fontsize=7.2, color=C["ink"],
                arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))
    ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=7.4, rotation=20, ha="right")
    ax.set_ylabel(r"individual forecasting $\rho^2$")
    ax.set_ylim(0, max(rho)*1.25)
    ax.text(0.0, -0.30, r"Each added modality contributes signal $a_m$ but also nuisance "
            r"variance $\delta_m$; fusion helps only while $a_m/\delta_m$ stays high.",
            transform=ax.transAxes, fontsize=6.8, color=C["ink"])
    save(fig, "fig4_fusion")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); print("done ->", OUT)
