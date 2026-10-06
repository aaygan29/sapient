"""Engine interface + the in-memory stimulus object (never persisted)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..schemas import Prediction


@dataclass
class Stimulus:
    """A request's raw inputs, held in memory only for the duration of processing.

    The model consumes *features* (V-JEPA2 / W2V-BERT / Llama), so the engine runs
    feature extraction internally. These raw bytes are discarded immediately after.
    """
    video_bytes: bytes | None = None
    audio_bytes: bytes | None = None
    transcript: str | None = None
    filename: str | None = None
    subject_idx: int = 0  # sapient-1 was trained with n_subjects=4; 0 is the default conditioning


@runtime_checkable
class Engine(Protocol):
    model_version: str

    def predict(self, stim: Stimulus, *, include_full_parcels: bool = False) -> Prediction:
        ...
