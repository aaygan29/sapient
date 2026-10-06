"""Adapter: Sapient-1 engine output -> the live digital brain.

Runs the encoding engine (mock or real) over a stimulus, pulls the per-second Yeo-7
network timeline, and feeds it through the `neurosignal` pipeline
(analyze_timeline -> compile_stats -> render_report) to produce the interactive,
second-by-second live brain. Personalization is real: each subject_idx conditions the
model's per-subject head (Cortex of One), so the subject switcher shows distinct brains.

    from sapient_serving.engine import build_engine
    from sapient_serving.engine.base import Stimulus
    from sapient_serving.engine.live_brain import save_live_brain

    eng = build_engine()                       # mock today; real with a checkpoint
    stim = Stimulus(transcript="...", filename="ad.mp4")
    save_live_brain("live_brain.html", eng, stim, subject_idxs=(0, 1, 2))

`neurosignal` must be importable (sibling package: pip install -e ../neurosignal, or
add it to PYTHONPATH). The import is lazy with a clear error.
"""
from __future__ import annotations

import dataclasses

from .base import Engine, Stimulus


def _require_neurosignal():
    try:
        import neurosignal  # noqa: F401
        return neurosignal
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "live_brain needs the `neurosignal` package importable. Install the sibling "
            "repo editable (pip install -e ../neurosignal) or add it to PYTHONPATH."
        ) from exc


def _per_second_networks(prediction) -> list[dict]:
    """Pull the engine's per-second Yeo-7 network timeline into the neurosignal frame format."""
    tl = prediction.signals.timeline
    nt = tl.network_timeline
    if not nt:
        raise RuntimeError(
            "engine prediction has no signals.timeline.network_timeline; update the engine "
            "(build_signals now populates it) or re-run on a current build."
        )
    n = len(tl.seconds)
    return [{net: series[t] for net, series in nt.items()} for t in range(n)]


def engine_to_reports(engine: Engine, stim: Stimulus, *, subject_idxs=(0,),
                      subject_labels=None, fps: float = 1.0) -> dict:
    """Run the engine per subject_idx and build the {label: {timeline, stats}} report map."""
    ns = _require_neurosignal()
    labels = list(subject_labels) if subject_labels else [f"Subject {i}" for i in subject_idxs]
    reports = {}
    for idx, label in zip(subject_idxs, labels):
        pred = engine.predict(dataclasses.replace(stim, subject_idx=int(idx)))
        per_sec = _per_second_networks(pred)
        tl = ns.analyze_timeline(per_second_networks=per_sec, fps=fps, source=label)
        st = ns.compile_stats(tl)
        reports[label] = {"timeline": tl.to_dict(), "stats": st.to_dict()}
    return reports


def live_brain_html(engine: Engine, stim: Stimulus, *, subject_idxs=(0,),
                    subject_labels=None, fps: float = 1.0,
                    title: str = "Sapient-1 — Live Digital Brain",
                    subtitle: str = "") -> str:
    ns = _require_neurosignal()
    reports = engine_to_reports(engine, stim, subject_idxs=subject_idxs,
                                subject_labels=subject_labels, fps=fps)
    sub = subtitle or f"{getattr(engine, 'model_version', 'engine')} · {stim.filename or 'upload'}"
    return ns.render_report(reports, title=title, subtitle=sub)


def save_live_brain(path: str, engine: Engine, stim: Stimulus, **kw) -> str:
    ns = _require_neurosignal()
    reports = engine_to_reports(engine, stim,
                                subject_idxs=kw.get("subject_idxs", (0,)),
                                subject_labels=kw.get("subject_labels"),
                                fps=kw.get("fps", 1.0))
    return ns.save_report(path, reports, title=kw.get("title", "Sapient-1 — Live Digital Brain"),
                          subtitle=kw.get("subtitle", getattr(engine, "model_version", "")))
