"""Split-conformal calibration — distribution-free prediction intervals with a
finite-sample marginal coverage guarantee (Angelopoulos & Bates, 2023). Lifted
from neurobridge/core.py and generalized to regression read-outs. The point: the
read-out abstains/widens honestly rather than emitting false-precise point estimates.
"""
from __future__ import annotations

import numpy as np


def split_conformal_regression(cal_residuals, alpha: float = 0.10) -> float:
    """Return conformal radius q such that P(|y - yhat| <= q) >= 1 - alpha.

    cal_residuals: signed or absolute residuals on a held-out calibration split.
    Uses the (n+1)(1-alpha)/n finite-sample corrected quantile.
    """
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


def conformal_intervals(y_pred, q: float):
    y_pred = np.asarray(y_pred, dtype=float)
    return np.stack([y_pred - q, y_pred + q], axis=-1)


# --- classification variant (1 - p(true)) for label read-outs -------------------
def split_conformal_classification(cal_true_probs, alpha: float = 0.10) -> float:
    """qhat on nonconformity s = 1 - p(true class), computed on calibration."""
    s = 1.0 - np.asarray(cal_true_probs, dtype=float).ravel()
    n = s.size
    if n == 0:
        raise ValueError("empty calibration set")
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(s, level, method="higher"))


def prediction_sets(probs, qhat: float):
    """Boolean mask of labels y with 1 - p(y) <= qhat (set size 0..K).
    A set of size > 1 is an abstention; size 1 is a committed prediction."""
    probs = np.asarray(probs, dtype=float)
    return (1.0 - probs) <= qhat
