"""The additive-head encoder interface + a faithful simulator of Mary's mechanism.

Mary (Cortex of One): out = group_head(x) + per_subject_head[idx](x), per-subject
low-rank (rank 16). This module exposes that exact contract so the enrollment code
is identical for the simulator and for real Mary; only `group_W` / heads change.

>>> REAL MARY WIRING <<<
Replace SimulatedCortex with a MaryAdapter whose `.group_predict(features)` calls
Mary's frozen group head on cached backbone features, and whose enrolled `head`
is the rank-16 per_subject delta in vertex×feature space. Everything downstream is
unchanged. The simulator exists ONLY to verify the machinery + statistics, exactly
like the ISM `--mock` run; no scientific claim is read off simulated data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class AdditiveEncoder:
    """out = X @ (group_W + head).T ; head is the per-subject delta (V×D), or None."""

    def __init__(self, group_W: np.ndarray):
        self.group_W = np.asarray(group_W, dtype=float)  # (V, D)

    @property
    def V(self) -> int:
        return self.group_W.shape[0]

    @property
    def D(self) -> int:
        return self.group_W.shape[1]

    def group_predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.group_W.T  # (N, V)

    def predict(self, X: np.ndarray, head: np.ndarray | None = None) -> np.ndarray:
        W = self.group_W if head is None else self.group_W + head
        return np.asarray(X, dtype=float) @ W.T


@dataclass
class Population:
    encoder: AdditiveEncoder
    heads: list          # true per-subject deltas (V×D) — UNKNOWN to enrollment
    meta: dict

    @property
    def average_head(self) -> np.ndarray:
        return np.mean(np.stack(self.heads), axis=0)


def make_population(n_subjects: int, D: int = 64, V: int = 200, rank: int = 8,
                    var_group: float = 0.30, var_indiv: float = 0.30,
                    var_noise: float = 0.40, seed: int = 0) -> Population:
    """A scientifically honest simulator with EXPLICIT variance shares.

    Subjects SHARE a group map and a common individuation SUBSPACE (A), differing
    only by subject-specific LOADINGS (c_s) — correlated, not independent RNG draws
    (the exact failure the digital-brain `DO_NOT_CITE` pilot taught us to avoid).

    The three signal sources are rescaled to hit target per-vertex variance shares:
      var_group  — what the shared 'average brain' can explain on its own
      var_indiv  — what only a per-subject head can recover  (the thing being tested)
      var_noise  — irreducible measurement noise (sets the test-retest ceiling)
    With (0.30, 0.30, 0.40), group-only r ≈ √0.30 ≈ 0.55 and a well-fit enrolled head
    should reach ≈ √0.60 ≈ 0.77 — so individuation is a real, but not dominant, fraction.
    """
    rng = np.random.default_rng(seed)
    group_W0 = rng.standard_normal((V, D))
    A = rng.standard_normal((V, rank))
    Bdir = rng.standard_normal((rank, D))
    loadings = rng.standard_normal((n_subjects, rank))
    deltas0 = [(A * loadings[s]) @ Bdir for s in range(n_subjects)]  # (V, D), shared subspace

    # calibrate per-vertex variances on a fixed probe set, then rescale to target shares
    X0 = rng.standard_normal((2000, D))

    def perv_var(W):
        Yt = X0 @ W.T
        Yt = Yt - Yt.mean(axis=0, keepdims=True)
        return float((Yt ** 2).mean())

    group_W = group_W0 * np.sqrt(var_group / perv_var(group_W0))
    vi = float(np.mean([perv_var(d) for d in deltas0]))
    s_i = np.sqrt(var_indiv / vi)
    heads = [d * s_i for d in deltas0]
    noise = float(np.sqrt(var_noise))

    meta = dict(D=D, V=V, rank=rank, var_group=var_group, var_indiv=var_indiv,
                var_noise=var_noise, noise=noise, seed=seed, A=A, Bdir=Bdir,
                loadings=loadings)
    return Population(AdditiveEncoder(group_W), heads, meta)


def sample_stimuli(D: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, D))


def observe(pop: Population, subject: int, X: np.ndarray, seed: int) -> np.ndarray:
    """Measured response of a subject to stimuli X = true encoder + measurement noise."""
    rng = np.random.default_rng(seed)
    clean = pop.encoder.predict(X, pop.heads[subject])
    return clean + pop.meta["noise"] * rng.standard_normal(clean.shape)


def vertex_pearson(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Mean over vertices of Pearson r across stimuli (Cortex-of-One's metric)."""
    yt = Y_true - Y_true.mean(axis=0, keepdims=True)
    yp = Y_pred - Y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0)) + 1e-12
    return float(np.mean(num / den))


def noise_ceiling(pop: Population, subject: int, X: np.ndarray,
                  seed_a: int, seed_b: int) -> float:
    """Test-retest reliability: correlate two noisy observations of the SAME subject.
    The honest ceiling any predictor is measured against (Cortex-of-One §3.2)."""
    a = observe(pop, subject, X, seed_a)
    b = observe(pop, subject, X, seed_b)
    return vertex_pearson(a, b)
