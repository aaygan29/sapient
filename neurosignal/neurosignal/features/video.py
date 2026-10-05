"""Video feature extraction (interface + schema).

The reference encoder consumes these [0,1] features; raw-video extraction needs the
[video] extra (decord/opencv, optionally the Digital Brain CLIP encoder for category cues).
Implement extract_video_features to populate this schema, or pass a precomputed dict.

Schema: {motion, scene_change, face_presence, brightness, color_variety, object_salience}
  motion/scene_change -> attention (dorsal); face_presence -> social (limbic/default);
  brightness/color_variety/object_salience -> visual cortex.
For learned, fMRI-grounded video activation use the Digital Brain / TRIBE encoder backend.
"""
from __future__ import annotations

VIDEO_FEATURE_SCHEMA = ("motion", "scene_change", "face_presence", "brightness", "color_variety", "object_salience")


def extract_video_features(path: str) -> dict:  # pragma: no cover - needs [video]
    raise NotImplementedError(
        "Raw-video extraction needs the [video] extra (decord/opencv). Implement here to fill "
        f"VIDEO_FEATURE_SCHEMA={VIDEO_FEATURE_SCHEMA}, or pass a precomputed feature dict to analyze(video=...)."
    )
