"""Slot for a LEARNED fMRI encoder (TRIBE / Sapient / Digital Brain).

A learned encoder maps multimodal stimulus features -> predicted whole-cortex activation,
which is then aggregated to Yeo-7 networks (see atlases.parcels_to_networks) and fed to the
same metric layer. This makes the pipeline fMRI-grounded rather than heuristic.

Wire it by implementing `encode(features) -> {Yeo-7 network: activation}` around your trained
model (e.g. the Digital Brain geometry-aware encoder, or a TRIBE-style video/audio/text encoder).
"""
from __future__ import annotations


class LearnedEncoder:  # pragma: no cover - requires a trained model
    name = "learned (fMRI-grounded)"

    def __init__(self, model=None):
        self._model = model

    def encode(self, features) -> dict[str, float]:
        raise NotImplementedError(
            "Plug a trained encoder here: stimulus features -> predicted cortical activation -> "
            "Yeo-7 networks. Until a checkpoint is wired, use encoder='reference'."
        )
