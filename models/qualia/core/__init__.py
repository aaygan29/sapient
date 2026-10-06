"""Qualia core — the Mary engine boundary + model registry."""
from .registry import resolve, CHANNELS
from .mary_engine import MaryEngine

__all__ = ["resolve", "CHANNELS", "MaryEngine"]
