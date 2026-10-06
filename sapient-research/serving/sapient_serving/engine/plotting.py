"""Render the brain-response image returned to clients (as a base64 PNG data URI).

Mock mode: a clean ROI-activation bar chart (no neuro deps).
Real mode: a cortical-surface plot via nilearn (wired with RealEngine).
"""
from __future__ import annotations

import base64
import io


def roi_bar_image(roi_scores: dict[str, float], title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(roi_scores.keys())
    vals = [roi_scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=110)
    ax.barh(names, vals, color="#4C9AA8")
    ax.set_xlim(0, max(0.6, max(vals) if vals else 0.6))
    ax.invert_yaxis()
    ax.set_xlabel("predicted activation (a.u.)")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def surface_image(vertex_pred) -> str:  # pragma: no cover - real path
    """fsaverage5 surface plot of the predicted response. Requires the [real] extra."""
    raise NotImplementedError(
        "surface_image needs nilearn (the [real] extra). Mirror eval.py's plot_brain_surface here."
    )
