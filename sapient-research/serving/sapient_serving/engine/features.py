"""Feature extraction for the REAL engine: raw media -> the 3 frozen-encoder features
that sapient-1 consumes.

Per the model's forward signature and the repo's data/extract_*:
  video : V-JEPA 2 Gigantic   -> (200, 1280) @ 2 Hz
  audio : Wav2Vec-BERT 2.0    -> (200, 1024) @ 2 Hz
  text  : Llama-3.2-3B        -> (200, 3072) @ 2 Hz   (sole text encoder)

This must mirror data/extract_{video,audio,text}_features.py EXACTLY (same model
revisions, sampling, pooling, alignment) or the features won't match training and
predictions degrade silently. Left as a typed stub to wire when the real engine
is stood up on a GPU box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Features:
    video: np.ndarray  # (T2, 1280)
    audio: np.ndarray  # (T2, 1024)
    text: np.ndarray   # (T2, 3072)


class RealFeatureExtractor:  # pragma: no cover - real path
    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def extract(self, *, video_bytes, audio_bytes, transcript) -> Features:
        raise NotImplementedError(
            "Wire to the repo's data/extract_{video,audio,text}_features.py "
            "(V-JEPA2 / W2V-BERT / Llama-3.2-3B @ 2 Hz). Must match training-time extraction exactly."
        )
