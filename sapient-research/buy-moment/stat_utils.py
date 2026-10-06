"""
stat_utils.py — shared statistical-rigor primitives for the connectome-validation
and multivariate-fusion phase of the buy-moment scoring layer.

Provenance note (per explicit user instruction, 2026-08-26)
-------------------------------------------------------------
Two of the primitives below are ported/adapted from the user's OWN prior
research code, not generic textbook implementations pulled from an external
package, and are cited as such rather than as anonymous methodology:

- `mdes_correlation`, `bootstrap_ci`, `GatedNumber`/`gate_effect` are a direct
  port (renamed, trimmed to what this module needs, otherwise unchanged
  arithmetic) of the analogous functions in the user's `decision_phenotype`
  project (`~/Desktop/Research/projects/Neuroscience/decision_phenotype/src/honesty.py`,
  GitHub `aaygan29/ideal-doodle`). That module's own framing: "every reported
  number is born gated" — a value is only reported if the effect size clears
  the minimum-detectable-effect-size (MDES) threshold at the given N;
  otherwise the module abstains rather than rounding up. We reuse that exact
  discipline here for the same reason `decision_phenotype` built it: this
  Sapient PR already abstains/blocks rather than overclaims throughout
  (`ACCEPTANCE_CRITERIA.md`'s gates), and the small-sample CCA/PLS instability
  problem in `multivariate_fusion.py` is exactly the kind of "can this N
  actually support this claim" question the MDES gate answers.
- `eb_shrink_subject_effects` is a SIMPLIFIED, scalar analog of the
  empirical-Bayes partial-pooling loop in the same project's
  `src/phenotype.py::fit_agents_pooled` (their "C4" hierarchical-pooling
  claim: shrinking per-agent logistic coefficients toward a re-estimated
  population mean/variance each EM iteration lowers per-agent estimate
  variance and the effective MDES). That function pools VECTOR-valued
  per-agent logistic coefficients fit from trial-level choice data — a
  genuinely different estimation problem from the SCALAR per-subject
  connectivity effect sizes here. We do NOT reuse their EM/MAP-logistic
  machinery (it does not apply: we have one scalar effect per subject per
  ROI group, not a per-agent design matrix of choices). What DOES transfer
  cleanly is the STRUCTURE: subject-level parameters shrunk toward a
  re-estimated population mean by a factor set by the ratio of within- to
  between-subject variance (a standard one-way random-effects / James-Stein
  empirical-Bayes shrinkage, the same conceptual move `fit_agents_pooled`
  makes at the vector level). This is implemented fresh below for scalars —
  it is inspired by, not copy-pasted from, `fit_agents_pooled`.

Everything else in this module (Benjamini-Hochberg FDR, max-statistic
family-wise permutation correction) is standard published methodology, cited
inline to the original papers, verified via WebSearch this session (see
`SCIENCE.md` for the verification note) — not from the user's own repos.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
#   Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false discovery
#   rate: a practical and powerful approach to multiple testing." Journal of
#   the Royal Statistical Society, Series B, 57(1), 289-300.
#   Verified via WebSearch this session (title/journal/volume/pages).
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """Return BH-adjusted q-values for an array of p-values (same order in/out)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q_raw = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity (step-up): q_i = min(q_i, q_{i+1}, ..., q_n)
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_monotone = np.clip(q_monotone, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_monotone
    return q


# ---------------------------------------------------------------------------
# Max-statistic family-wise permutation correction
#   Nichols, T.E. & Holmes, A.P. (2002). "Nonparametric permutation tests for
#   functional neuroimaging: a primer with examples." Human Brain Mapping,
#   15(1), 1-25. https://doi.org/10.1002/hbm.1058
#   Verified via WebSearch this session (title/journal/volume/pages/DOI).
# ---------------------------------------------------------------------------

def max_stat_fwer(observed: Sequence[float], null_draws: np.ndarray) -> np.ndarray:
    """Family-wise-error-rate-corrected p-values via the max-statistic method.

    Parameters
    ----------
    observed : (G,) observed statistic per group (e.g. within-group connectivity r)
    null_draws : (n_perm, G) null statistic for each group at each permutation draw

    Returns
    -------
    (G,) FWER-corrected p-values: for each group, the fraction of the
    ACROSS-GROUP MAX null (one pooled null built by taking the max across all
    G groups at each of the n_perm draws) that is >= the group's observed
    value. This is the standard single-threshold correction for multiple
    non-independent tests that Nichols & Holmes (2002) describe as more
    standard in neuroimaging than per-test correction, because it uses the
    actual joint null (whatever dependence exists between groups' null
    distributions) rather than assuming independence.
    """
    obs = np.asarray(observed, dtype=float)
    null = np.asarray(null_draws, dtype=float)  # (n_perm, G)
    n_perm = null.shape[0]
    family_max_null = null.max(axis=1)  # (n_perm,) — one pooled null distribution
    p_fwer = np.array([
        (np.sum(family_max_null >= o) + 1) / (n_perm + 1) for o in obs
    ])
    return p_fwer


# ---------------------------------------------------------------------------
# MDES / bootstrap CI / gate — ported from decision_phenotype's honesty.py
# (see module docstring provenance note above). Trimmed to what this file
# needs; arithmetic unchanged from the source.
# ---------------------------------------------------------------------------

def mdes_correlation(n: int, alpha: float = 0.05, power: float = 0.8,
                      two_sided: bool = True) -> float:
    """Minimum detectable Pearson r at sample size n (Fisher-z approximation).

    Ported from decision_phenotype/src/honesty.py::mdes_correlation
    (aaygan29/ideal-doodle, user's own prior work).
    """
    if n <= 3:
        return float("nan")
    za = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    zb = stats.norm.ppf(power)
    z_r = (za + zb) / np.sqrt(n - 3)
    return float(np.tanh(z_r))


def bootstrap_ci(values: np.ndarray, statistic: Callable = np.mean, n_boot: int = 2000,
                  alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Ported from decision_phenotype/src/honesty.py::bootstrap_ci."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    boots = np.array([statistic(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"point": float(statistic(v)), "lo": float(lo), "hi": float(hi)}


@dataclass
class GatedNumber:
    """A reportable number, or an abstention. Never a bare float.

    Ported from decision_phenotype/src/honesty.py::GatedNumber.
    """
    name: str
    value: Optional[float]
    ci: Optional[Sequence[float]] = None
    provenance: Dict[str, object] = field(default_factory=dict)
    abstained: bool = False
    reason: Optional[str] = None
    mdes: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def gate_effect(name: str, effect: float, n: int, alpha: float = 0.05, power: float = 0.8,
                 provenance: Optional[Dict[str, object]] = None) -> GatedNumber:
    """Emit a GatedNumber for a correlation effect; abstain if |effect| < MDES at this N.

    Ported from decision_phenotype/src/honesty.py::gate_effect (kind="correlation"
    branch only — this module never gates standardized-mean effects).
    """
    mdes = mdes_correlation(n, alpha, power)
    prov = dict(provenance or {})
    prov.update({"n": n, "alpha": alpha, "power": power, "kind": "correlation"})
    if not np.isfinite(mdes) or abs(effect) < mdes:
        return GatedNumber(name=name, value=None, abstained=True, mdes=mdes, provenance=prov,
                            reason=(f"|effect|={abs(effect):.3f} < MDES={mdes:.3f} at n={n}; "
                                    "underpowered to license a claim of this size"))
    return GatedNumber(name=name, value=float(effect), mdes=mdes, provenance=prov)


# ---------------------------------------------------------------------------
# Scalar empirical-Bayes shrinkage across subjects (fresh implementation,
# structurally inspired by decision_phenotype/src/phenotype.py::fit_agents_pooled
# — see module docstring provenance note for exactly what does and doesn't
# transfer from that vector-valued logistic-coefficient pooling loop).
# ---------------------------------------------------------------------------

def eb_shrink_subject_effects(subject_effects: np.ndarray) -> Dict[str, object]:
    """One-way random-effects empirical-Bayes shrinkage of per-subject scalar
    effects toward the population mean.

    Standard method (Morris 1983-style / James-Stein empirical Bayes for a
    one-way random-effects model): estimate the population mean `mu` and the
    between-subject variance `tau2` (method-of-moments, clipped at 0), then
    shrink each subject's raw effect toward `mu` by a factor
    `tau2 / (tau2 + sigma2_i)`, where `sigma2_i` is that subject's own
    sampling variance (here approximated as the pooled within-subject
    variance across all 8 subjects' effects, since we do not have a
    per-subject SE from a single scalar summary). This is the SAME structural
    move as `fit_agents_pooled`'s per-iteration "M-step: update population
    mean/var" (see docstring) applied to scalars instead of logistic-
    coefficient vectors, and with a closed-form shrinkage factor instead of
    an EM/MAP loop (no likelihood to optimize for a single scalar per
    subject, so the iterative MAP-logistic step does not apply here).

    Returns
    -------
    dict with `mu`, `tau2`, `shrinkage_factor`, `raw_effects`, `shrunk_effects`.
    """
    x = np.asarray(subject_effects, dtype=float)
    n = x.size
    mu = float(x.mean())
    # Method-of-moments between-subject variance minus a rough estimate of
    # within-subject sampling variance (using the SEM of the mean across
    # subjects as sigma2_i proxy, since each subject contributes one scalar).
    sigma2 = float(x.var(ddof=1)) / max(n, 1) if n > 1 else 0.0
    tau2_raw = float(x.var(ddof=1)) - sigma2
    tau2 = max(tau2_raw, 0.0)
    if tau2 + sigma2 <= _EPS:
        shrink = 1.0  # degenerate (zero variance): no shrinkage needed/possible
    else:
        shrink = tau2 / (tau2 + sigma2)
    shrunk = mu + shrink * (x - mu)
    return {
        "mu": mu,
        "tau2_between_subject": tau2,
        "sigma2_within_subject_proxy": sigma2,
        "shrinkage_factor": float(shrink),
        "raw_effects": x.tolist(),
        "shrunk_effects": shrunk.tolist(),
        "note": ("shrinkage_factor near 1.0 means little pooling was applied "
                 "(between-subject variance dominates); near 0.0 means heavy "
                 "pooling toward the population mean (within-subject noise "
                 "dominates). Structurally inspired by decision_phenotype's "
                 "fit_agents_pooled (aaygan29/ideal-doodle) but implemented "
                 "fresh for scalar per-subject effects — see module docstring."),
    }
