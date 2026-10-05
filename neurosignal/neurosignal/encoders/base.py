"""Encoder interface."""
from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable


@runtime_checkable
class Encoder(Protocol):
    name: str

    def encode(self, features: Mapping[str, dict]) -> dict[str, float]:
        """features: {modality: feature_dict} -> {Yeo-7 network: activation in [0,1]}.

        Networks no modality speaks to should be OMITTED (so coverage stays honest).
        """
        ...
