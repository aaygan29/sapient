"""Mock engine: correctly-shaped, deterministic-per-input outputs including the
consumer-signal layer. Lets the full demo run before a checkpoint exists.

Numbers are illustrative (model_version says MOCK; /v1/info states provenance).
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..schemas import Prediction
from ..settings import settings
from . import plotting
from . import postprocess as pp
from . import signals as sig
from .base import Stimulus


def _seed_from(stim: Stimulus) -> int:
    h = hashlib.sha256()
    if stim.filename:
        h.update(stim.filename.encode("utf-8"))
    if stim.transcript:
        h.update(stim.transcript.encode("utf-8"))
    for b in (stim.video_bytes, stim.audio_bytes):
        if b:
            h.update(str(len(b)).encode("ascii"))
    # subject_idx conditions the per-subject head in the real model; vary the mock too so
    # the personalized live-brain switcher shows distinct subjects in the sandbox.
    h.update(b"subj" + str(stim.subject_idx).encode("ascii"))
    return int.from_bytes(h.digest()[:8], "big")


def _smooth01(rng: np.random.Generator, n: int, k: int = 9) -> np.ndarray:
    x = rng.normal(0.0, 1.0, n + k)
    xs = np.convolve(x, np.ones(k) / k, mode="same")[:n]
    lo, hi = xs.min(), xs.max()
    return (xs - lo) / (hi - lo + 1e-9)


class MockEngine:
    model_version = settings.model_version + " (MOCK — no weights loaded)"

    def predict(self, stim: Stimulus, *, include_full_parcels: bool = False) -> Prediction:
        rng = np.random.default_rng(_seed_from(stim))
        n = 100  # 1 Hz / 100-TR window
        seconds = np.arange(n, dtype=float)

        # per-clip baseline engagement -> clips land across Buy / Hold / Sell.
        # Lightly nudged by transcript wording so the DEMO feels sensible (illustrative only;
        # the real model reads the actual video — see /v1/info).
        pos_words = ("luxury", "emotional", "reveal", "crowd", "beautiful", "love", "puppy",
                     "exciting", "aspirational", "cinematic", "music", "surprise", "warm", "bass",
                     "neon", "energy", "stunning", "desire", "reward", "win", "gorgeous", "premium")
        neg_words = ("boring", "confusing", "disclaimer", "dense", "monotone", "spreadsheet",
                     "terms", "legal", "tax", "tutorial", "gray", "grey", "slow", "complicated",
                     "warning", "instructions", "jargon", "tedious", "refund")
        level = float(rng.uniform(0.42, 0.72))
        if stim.transcript:
            t = stim.transcript.lower()
            level += 0.12 * sum(w in t for w in pos_words) - 0.13 * sum(w in t for w in neg_words)
        level = float(np.clip(level, 0.12, 0.92))

        def _curve(center: float) -> np.ndarray:
            return np.clip(center + 0.16 * (2.0 * _smooth01(rng, n) - 1.0), 0.0, 1.0)

        construct_ts = {}
        for c in sig.CONSTRUCTS:
            # approach drivers track engagement; friction drivers (conflict, load) run inversely
            center = level if c.weight > 0 else (0.85 - 0.55 * level)
            construct_ts[c.key] = _curve(center)
        # reward integrates emotion + attention (value follows affect + orienting)
        construct_ts["reward"] = np.clip(
            0.6 * construct_ts["reward"] + 0.25 * construct_ts["emotion"] + 0.15 * construct_ts["attention"],
            0.0, 1.0,
        )
        signals = sig.build_signals(construct_ts, seconds)

        roi_scores = {name: float(np.clip(rng.normal(0.25, 0.12), 0.0, 1.0)) for name in pp.YEO7_NETWORKS}
        parcel_mean = np.clip(rng.uniform(0.0, 0.6, size=pp.N_PARCELS), 0.0, 1.0)
        summary = pp.summarize_parcels(parcel_mean, include_full=include_full_parcels)
        image = plotting.roi_bar_image(roi_scores, title="Predicted ROI activation (mock)")

        return Prediction(
            roi_scores=roi_scores,
            parcels=summary,
            signals=signals,
            image_data_uri=image,
            n_timepoints=n,
        )
