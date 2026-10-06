"""The serving engine: turns a stimulus into coarse, paper-faithful outputs.

Two implementations behind one interface:
  - MockEngine  (mock.py): shaped, plausible outputs, zero heavy deps. Runs today.
  - RealEngine  (real.py): imports the private sapient1 package + frozen encoders,
                            loads the checkpoint, runs the real forward pass.

Select via SAPIENT_ENGINE=mock|real (see build_engine()).
"""
from __future__ import annotations

from ..settings import settings
from .base import Engine, Stimulus


def build_engine() -> Engine:
    if settings.engine_mode == "real":
        from .real import RealEngine
        return RealEngine()
    if settings.engine_mode == "digital_brain":
        from .digital_brain import DigitalBrainEngine
        return DigitalBrainEngine()
    from .mock import MockEngine
    return MockEngine()


__all__ = ["Engine", "Stimulus", "build_engine"]
