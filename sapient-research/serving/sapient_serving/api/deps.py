"""Shared singletons (the engine). Built once at startup."""
from __future__ import annotations

from ..engine import Engine, build_engine

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine
