"""Transparent reference encoder.

Maps interpretable per-modality features -> Yeo-7 network activation with documented,
auditable weights. This is NOT a learned fMRI model — it is a theory-driven reference so
the full multimodal pipeline runs out of the box and every score is inspectable. Swap in
the LearnedEncoder for fMRI-grounded activation.

Mappings (rationale):
  Limbic (reward/affect)        <- arousal, valence intensity, social praise/agreement; audio energy; faces
  Default (self/social/memory)  <- 2nd-person + agreement + narrative valence; faces
  VentralAttention (salience)   <- negativity, hedging/uncertainty, surprise; audio pitch/loudness variance
  DorsalAttention (orienting)   <- info density, directives; audio tempo/onsets; video motion/scene-change
  Frontoparietal (control)      <- information density + certainty (analytic load)
  Visual                        <- imagery words; video brightness/color/object salience
  Somatomotor                  <- action words; audio voicing/brightness
"""
from __future__ import annotations

import numpy as np

from ..constructs import YEO7_NETWORKS


def _c(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _text_to_networks(tf: dict) -> dict[str, float]:
    g = lambda k: float(tf.get(k, 0.0))
    negativity = max(0.0, -g("valence_signed"))
    return {
        "Limbic": _c(0.40 * g("arousal") + 0.25 * g("valence_intensity") + 0.20 * g("praise") + 0.15 * g("agreement")),
        "Default": _c(0.45 * g("second_person") + 0.30 * g("agreement") + 0.25 * g("valence_intensity")),
        "VentralAttention": _c(0.45 * negativity + 0.30 * g("hedging") + 0.25 * g("arousal")),
        "DorsalAttention": _c(0.55 * g("info_density") + 0.25 * g("certainty") + 0.20 * g("action")),
        "Frontoparietal": _c(0.55 * g("info_density") + 0.30 * g("certainty") + 0.15 * (1.0 - g("arousal"))),
        "Visual": _c(g("imagery")),
        "Somatomotor": _c(g("action")),
    }


def _audio_to_networks(af: dict) -> dict[str, float]:
    g = lambda k: float(af.get(k, 0.0))
    return {
        "Limbic": _c(0.5 * g("energy") + 0.5 * g("loudness_var")),
        "Somatomotor": _c(0.5 * g("voicing") + 0.5 * g("brightness")),
        "DorsalAttention": _c(0.5 * g("tempo") + 0.5 * g("onset_rate")),
        "VentralAttention": _c(0.5 * g("pitch_var") + 0.5 * g("loudness_var")),
    }


def _video_to_networks(vf: dict) -> dict[str, float]:
    g = lambda k: float(vf.get(k, 0.0))
    return {
        "Visual": _c(0.4 * g("brightness") + 0.3 * g("color_variety") + 0.3 * g("object_salience")),
        "DorsalAttention": _c(0.5 * g("motion") + 0.5 * g("scene_change")),
        "Limbic": _c(g("face_presence")),
        "Default": _c(0.5 * g("face_presence")),
    }


class ReferenceEncoder:
    name = "reference (transparent heuristic)"

    def encode(self, features) -> dict[str, float]:
        contribs: dict[str, list[float]] = {n: [] for n in YEO7_NETWORKS}
        for modality, fn in (("text", _text_to_networks), ("audio", _audio_to_networks), ("video", _video_to_networks)):
            feats = features.get(modality)
            if feats:
                feats = feats.as_dict() if hasattr(feats, "as_dict") else dict(feats)
                for net, val in fn(feats).items():
                    contribs[net].append(val)
        return {net: float(np.mean(vals)) for net, vals in contribs.items() if vals}
