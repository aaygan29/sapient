"""Encoders: map modality features -> Yeo-7 network activation.

- ReferenceEncoder: transparent, documented heuristic — runs anywhere, fully auditable.
- LearnedEncoder: slot for a trained fMRI encoder (TRIBE / Sapient / Digital Brain).
"""
from __future__ import annotations

from .reference import ReferenceEncoder


def get_encoder(name="reference"):
    # passthrough: an already-constructed Encoder (e.g. an EnrolledEncoder) is returned as-is
    if not isinstance(name, str):
        if hasattr(name, "encode"):
            return name
        raise ValueError(f"encoder must be a name or an object with .encode(); got {type(name)}")
    if name == "reference":
        return ReferenceEncoder()
    if name == "learned":
        from .learned import LearnedEncoder
        return LearnedEncoder()
    raise ValueError(f"unknown encoder {name!r}; use 'reference' or 'learned'")
