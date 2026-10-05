"""The honesty / validation layer (a.k.a. neuro-ai-core), vendored into neurosignal.

Four default-on primitives so every emitted number is defensible:
  * Finding          — a number must carry its evidence or it prints UNVALIDATED
  * conformal        — distribution-free prediction intervals + honest abstention
  * specificity_gate — withhold a construct score when the signal is confounded
  * Provenance       — encoder/data/seed manifest with a short fingerprint

Pure NumPy, deterministic, no network calls — consistent with the neurosignal core.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------- Finding
@dataclass
class Finding:
    name: str
    value: float
    dataset: str
    baseline: float
    effect_size: float
    ci95: Tuple[float, float]
    n: int
    passed: bool
    note: str = ""

    def __repr__(self) -> str:
        tag = "PASS" if self.passed else "UNVALIDATED"
        lo, hi = self.ci95
        out = (f"Finding[{tag}] {self.name}={self.value:.4f} "
               f"(baseline={self.baseline:.4f}, eff={self.effect_size:.3f}, "
               f"95%CI=[{lo:.4f},{hi:.4f}], n={self.n})")
        return out + (f" :: {self.note}" if self.note else "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci95"] = [float(self.ci95[0]), float(self.ci95[1])]
        return d


class UnvalidatedClaimError(RuntimeError):
    pass


def gate(*findings: "Finding") -> None:
    """Raise before serving if any Finding is unvalidated."""
    bad = [f for f in findings if not f.passed]
    if bad:
        raise UnvalidatedClaimError(
            "Refusing to emit unvalidated claims:\n"
            + "\n".join(f"  - {f!r}" for f in bad)
        )


# -------------------------------------------------------------------------- conformal
def split_conformal_regression(cal_residuals, alpha: float = 0.10) -> float:
    """Conformal radius q with P(|y - yhat| <= q) >= 1-alpha (Angelopoulos & Bates, 2023)."""
    cal = np.abs(np.asarray(cal_residuals, dtype=float)).ravel()
    n = cal.size
    if n == 0:
        raise ValueError("empty calibration set")
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(cal, level, method="higher"))


def empirical_coverage(y_true, y_pred, q: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) <= q))


def conformal_interval(point: float, q: float) -> Tuple[float, float]:
    return (float(point - q), float(point + q))


# ----------------------------------------------------------------------- specificity
def concordance(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def specificity_gate(contrast, construct_map, confound_maps: Sequence,
                     threshold: float = 1.0):
    """Pass iff |corr(contrast, construct)| / max_k |corr(contrast, confound_k)| >= threshold.

    passed=False => WITHHOLD the construct score; the signal is confounded.
    Returns (passed, ratio, details).
    """
    c_con = abs(concordance(contrast, construct_map))
    c_conf = [abs(concordance(contrast, m)) for m in confound_maps]
    max_conf = max(c_conf) if c_conf else 0.0
    ratio = (c_con / max_conf) if max_conf > 0 else float("inf")
    return ratio >= threshold, ratio, {
        "construct_concordance": c_con, "max_confound_concordance": max_conf,
    }


# ----------------------------------------------------------------------- provenance
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_array(arr) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(arr)).tobytes())


@dataclass
class Provenance:
    encoder_id: str
    encoder_commit: str = "unset"
    seed: int = 0
    enrollment_modality: str = "none"     # 'fmri' | 'eeg' | 'eeg+fmri' | 'simulated' | 'none'
    minutes_of_data: float = 0.0
    calibration_stimulus: str = ""
    data_checksums: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    extra: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return sha256_bytes(payload)[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d
