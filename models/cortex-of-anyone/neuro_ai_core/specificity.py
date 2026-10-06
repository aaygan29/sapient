"""The specificity gate — operationalizes the Mary read-out lesson: never report a
contrast as construct-X unless it concords with construct-X more than with the
low-level audiovisual confounds. Generalized from mary_readouts.py. Default-on:
a construct score is WITHHELD when the signal is better explained by a confound.
"""
from __future__ import annotations

import numpy as np


def concordance(a, b) -> float:
    """Mean-centered cosine concordance between two spatial maps/vectors."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def specificity_gate(contrast, construct_map, confound_maps, threshold: float = 1.0):
    """Pass iff |concordance(contrast, construct)| / max_k |concordance(contrast, confound_k)| >= threshold.

    Returns (passed: bool, ratio: float, details: dict).
    A passed=False means: WITHHOLD this construct score; the signal is confounded.
    """
    c_con = abs(concordance(contrast, construct_map))
    c_conf = [abs(concordance(contrast, m)) for m in confound_maps]
    max_conf = max(c_conf) if c_conf else 0.0
    ratio = (c_con / max_conf) if max_conf > 0 else float("inf")
    return ratio >= threshold, ratio, {
        "construct_concordance": c_con,
        "max_confound_concordance": max_conf,
        "confound_concordances": c_conf,
    }
