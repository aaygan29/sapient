"""Config loading, seed setting, device helpers.

Used by train.py, eval.py, release.py, and the data scripts.

Config inheritance: a config file may declare `inherits: <other_file.yaml>`
at the top level. We resolve inheritance with a shallow recursive merge —
child keys override parent keys; nested dicts are merged recursively; lists
are *replaced* (not merged) so a child's `metrics:` list overrides the
parent's entirely.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML config and resolve a single `inherits:` chain.

    The `inherits:` value is resolved relative to the child config's directory.
    """
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
    """Seed Python, NumPy, and PyTorch (CPU + CUDA). Deterministic where feasible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
