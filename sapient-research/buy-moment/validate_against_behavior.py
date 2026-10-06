"""
validate_against_behavior.py — real-behavior validation gate for the
buy-moment composite score.

Motivation
----------
arXiv:2607.01400, "A global predicted-fMRI drive signal from TRIBE does not
predict YouTube replay heatmaps," shows that collapsing a TRIBE-style
whole-cortex encoder's output into a single aggregate engagement/drive signal
does NOT correlate with real behavioral engagement (YouTube "most replayed"
heatmaps) — pooled position-controlled partial r = +0.058, 95% CI
[-0.04, 0.15], p = 0.23, below basic loudness/motion baselines — even though
the same encoder achieves good held-out fMRI Pearson r on its training
objective. Good encoder fit does not imply the aggregated read-out predicts
behavior.

`buy_moment_detector.detect_buy_moments` does structurally the same thing:
a hand-weighted sum over ROI-group means, collapsed into one composite
`score_z`. Until now, `sapienteval/ACCEPTANCE_CRITERIA.md` only gated ship
decisions on fMRI Pearson R (encoder fit), never on a real behavioral
outcome for the buy-moment composite specifically — exactly the gap the
paper above identifies as failure-prone.

What ground truth exists in the pipeline today
-----------------------------------------------
We checked `sapienteval/03_parcellate.py` and `sapienteval/04_align_stimuli.py`
(the NeuroEngage ds004996 pipeline). Their outputs are (a) per-run (T, 400)
parcellated BOLD and (b) per-TR aligned transcript/event-type text. Neither
stage currently emits a behavioral ENGAGEMENT rating, purchase decision, or
other outcome label distinct from the BOLD signal itself — `04_align_stimuli`
aligns event_type tags (fixation_cross / comprehension / production / silence
/ turn_initiation) and transcript text, not a scalar behavioral outcome.
`ACCEPTANCE_CRITERIA.md` §3's "ground-truth NeuroEngage engagement label" is
a v1 mapping-head construct (see SCIENCE.md §7, "not yet implemented"), not a
channel this detector could validate against today.

Honest gap noted: there is currently no real behavioral ground-truth channel
wired into this pipeline for the buy-moment composite to be checked against.
Real-outcome validation (e.g. once a real purchase/engagement behavioral
channel exists) is future work and should REPLACE this harness, not sit
alongside it indefinitely — see the ship-gate text this motivates in
`sapienteval/ACCEPTANCE_CRITERIA.md` §4.

What this script does instead
-------------------------------
A synthetic validation harness, disclosed as synthetic throughout:

1. Generate synthetic (T, 400) parcel-activation data with a KNOWN, injected
   "engagement" latent process (a slow smoothed random walk plus discrete
   bump events), wired into the ROI groups so that a genuine buy-moment
   detector SHOULD track it.
2. Run `detect_buy_moments` on the synthetic data.
3. Correlate the detector's `score_z` against the injected ground-truth
   engagement process.
4. Run a permutation test (circular time-shifts of the ground-truth signal,
   which preserves its autocorrelation structure — a stronger null than iid
   shuffling for time series) to ask whether the observed correlation is
   above what a chance alignment would produce.

This only demonstrates that the detector's arithmetic is *capable* of
tracking an injected signal shaped like engagement, under favorable
synthetic conditions. It is NOT evidence the composite tracks real human
behavior — that is exactly the gap arXiv:2607.01400 warns against papering
over. Treat a PASS here as "the arithmetic is not obviously broken," not as
"the buy-moment score is behaviorally validated."

Run as a script
----------------
    python validate_against_behavior.py [--seed N] [--n-perm N] [--t N]
"""

from __future__ import annotations
import argparse
import sys
from typing import Any

import numpy as np

from buy_moment_detector import detect_buy_moments, ROI_GROUPS


def _synthetic_engagement_process(T: int, rng: np.random.Generator) -> np.ndarray:
    """
    A slow smoothed random walk (baseline drift in engagement) plus a
    handful of discrete "engagement bump" events (analogous to a genuine
    buy-moment spike). Normalized to zero mean / unit variance.
    """
    walk = np.cumsum(rng.normal(0, 1, size=T))
    # Smooth with a simple moving average to give it slower dynamics than
    # single-TR noise (engagement doesn't spike/decay in 1s in reality).
    kernel = np.ones(5) / 5.0
    walk = np.convolve(walk, kernel, mode="same")

    n_bumps = max(3, T // 60)
    bump_centers = rng.choice(np.arange(10, T - 10), size=n_bumps, replace=False)
    bumps = np.zeros(T)
    for c in bump_centers:
        width = rng.integers(3, 7)
        amp = rng.uniform(2.0, 4.0)
        idx = np.arange(max(0, c - width), min(T, c + width))
        bumps[idx] += amp * np.exp(-0.5 * ((idx - c) / (width / 2)) ** 2)

    engagement = walk + bumps
    engagement = (engagement - engagement.mean()) / (engagement.std() + 1e-9)
    return engagement.astype(np.float32)


def _synthetic_parcel_data_from_engagement(
    engagement: np.ndarray,
    *,
    signal_gain: float = 1.2,
    noise_sd: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a synthetic (T, 400) parcel matrix where the POSITIVE-weighted
    "generalizable" and "idiosyncratic" ROI groups are driven by the
    injected engagement process (with independent per-parcel noise), the
    negative-weighted groups are driven by the INVERSE of engagement (a
    genuine buy signal should show reward up + resistance/insula down
    together), and all other parcels are pure noise. This is disclosed
    synthetic data — see module docstring's Honest-gap-noted section.
    """
    rng = np.random.default_rng(seed)
    T = engagement.shape[0]
    parcel_act = rng.normal(0, noise_sd, size=(T, 400)).astype(np.float32)

    for name, spec in ROI_GROUPS.items():
        cols = np.asarray(sorted(set(spec["parcels"])), dtype=int)
        if cols.size == 0:
            continue
        drive = engagement if spec["weight"] > 0 else -engagement
        noise = rng.normal(0, 1, size=(T, cols.size)).astype(np.float32)
        parcel_act[:, cols] = signal_gain * drive[:, None] + noise

    return parcel_act


def _circular_shift_null(
    score_z: np.ndarray,
    ground_truth: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """
    Permutation test via circular time-shift of the ground-truth signal
    (preserves its autocorrelation, unlike iid shuffling — the appropriate
    null for correlating two autocorrelated time series). Returns the
    observed Pearson r and the null distribution of r under random shifts.
    """
    T = score_z.shape[0]
    observed_r = float(np.corrcoef(score_z, ground_truth)[0, 1])

    null_rs = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        shift = int(rng.integers(1, T))  # avoid the zero shift (= observed)
        shifted = np.roll(ground_truth, shift)
        null_rs[i] = float(np.corrcoef(score_z, shifted)[0, 1])

    return observed_r, null_rs


def run_validation(
    *,
    T: int = 600,
    n_perm: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    engagement_gt = _synthetic_engagement_process(T, rng)
    parcel_act = _synthetic_parcel_data_from_engagement(engagement_gt, seed=seed)

    result = detect_buy_moments(parcel_act)
    score_z = result["score_z"]

    observed_r, null_rs = _circular_shift_null(
        score_z, engagement_gt, n_perm=n_perm, rng=rng)

    # One-sided: does the buy-moment score track engagement MORE than chance
    # alignment would predict?
    p_value = float((np.sum(null_rs >= observed_r) + 1) / (n_perm + 1))
    passed = bool(observed_r > 0 and p_value < alpha)

    return {
        "observed_r": observed_r,
        "p_value": p_value,
        "alpha": alpha,
        "n_perm": n_perm,
        "n_timepoints": T,
        "null_r_mean": float(null_rs.mean()),
        "null_r_std": float(null_rs.std()),
        "passed": passed,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--t", type=int, default=600, help="number of synthetic timepoints")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv[1:])

    out = run_validation(T=args.t, n_perm=args.n_perm, seed=args.seed, alpha=args.alpha)

    print("Buy-moment vs synthetic-behavior validation (arXiv:2607.01400 motivated)")
    print("=" * 78)
    print("SYNTHETIC DATA — no real behavioral ground truth exists in the pipeline")
    print("today (checked 03_parcellate.py / 04_align_stimuli.py; see module docstring).")
    print("-" * 78)
    print(f"n_timepoints        : {out['n_timepoints']}")
    print(f"observed Pearson r  : {out['observed_r']:+.4f}  "
          f"(score_z vs injected ground-truth engagement)")
    print(f"permutation null    : mean {out['null_r_mean']:+.4f}, "
          f"std {out['null_r_std']:.4f}  (n_perm={out['n_perm']}, circular time-shift)")
    print(f"one-sided p-value   : {out['p_value']:.4f}  (alpha={out['alpha']})")
    print(f"RESULT              : {'PASS' if out['passed'] else 'FAIL'} — "
          f"detector {'recovers' if out['passed'] else 'does NOT recover'} the "
          f"injected signal above chance")
    print()
    print("Reminder: PASS here means the detector's arithmetic can track a")
    print("synthetic signal explicitly wired to look like engagement. It is")
    print("NOT evidence of real behavioral validity — see ACCEPTANCE_CRITERIA.md")
    print("§4 for the ship gate this harness stands in for until a real")
    print("behavioral channel exists in the pipeline.")

    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
