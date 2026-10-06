"""Config loading, seed setting, device helpers — mirrors sapient1/sapient1/utils.py.

Config inheritance: a config may declare `inherits: <other.yaml>` at the top
level. Child keys override parent; nested dicts merge recursively; lists are
replaced (not merged).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    cfg_path = Path(path).resolve()
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    parent_ref = cfg.pop("inherits", None)
    if parent_ref is None:
        return cfg
    parent_path = (cfg_path.parent / parent_ref).resolve()
    parent = load_config(parent_path)
    return _deep_merge(parent, cfg)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
