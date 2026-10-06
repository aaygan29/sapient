"""
multivariate_fusion.py — PLS/CCA fusion of ROI_GROUPS-restricted parcel
activity against the synthetic behavioral proxy, with an explicit
small-sample instability disclosure and a lower-variance alternative.

Why this exists, and why it is heavily caveated
--------------------------------------------------
`ACCEPTANCE_CRITERIA.md` gate 4 and `SCIENCE.md` §11/§13 already establish
that Sapient's pipeline trains on N=4 (pilot) to N=8 (scale-up target)
SUBJECTS. Before implementing any multivariate brain-behavior fusion method
here, this session searched (WebSearch, verified, not from memory) for
recent literature on CCA/PLS reliability at small sample sizes:

- Helmer et al. (2024), "On the stability of canonical correlation analysis
  and partial least squares with application to brain-behavior
  associations," Communications Biology 7:1223. Confirmed via WebSearch
  (title, journal, article number). Central finding: CCA and PLS
  brain-behavior associations require a LARGE sample (the paper's own
  power-calculator work models this in the N > ~1000 range) to be stable;
  at typical-study sample sizes, both the MAGNITUDE and the FEATURE PATTERN
  (which brain regions load onto the association) recovered by CCA/PLS are
  highly unstable and can differ substantially run-to-run or resample-to-
  resample. This directly follows the Marek et al. (2022, Nature) line on
  reproducibility requiring thousands of participants for brain-wide
  association studies, applied specifically to CCA/PLS.
- A companion/related paper (bioRxiv 2020.08.25.265546, "On stability of
  Canonical Correlation Analysis and Partial Least Squares with application
  to brain-behavior associations," later published in the Communications
  Biology work above) makes the same point with an explicit power/sample-
  size calculator for CCA/PLS stability.

Sapient's real training N (4-8 subjects) is far below the stability regime
either paper describes. Per this session's own instruction for exactly this
situation: rather than silently implementing PLS/CCA as if it were a safe
drop-in replacement for the hand-picked ROI_GROUPS weights, we implement it
ANYWAY (per the task's option (a)) but wrap every fitted number in:
  1. an explicit, named instability disclosure citing Helmer et al. (2024)
     inline in every output dict, not just this docstring;
  2. a block-bootstrap stability check (weight variance / sign-agreement
     rate across resamples), so the output itself is honest about how
     unstable the fit is;
  3. an MDES (minimum-detectable-effect-size) abstention gate on the
     achieved latent correlation, ported from the user's own
     `decision_phenotype` project's honesty layer (see `stat_utils.py`
     docstring) — if the fitted correlation doesn't clear the MDES at the
     effective (block-bootstrap) sample size, we report an abstention
     instead of a bare number, exactly the discipline this whole PR uses
     elsewhere (`ACCEPTANCE_CRITERIA.md`'s gates abstain/block rather than
     round up);
  4. a lower-variance univariate alternative (per-group correlation with
     the proxy, FDR-corrected across groups) AND an accuracy-weighted
     channel-combination baseline structurally adapted from the user's
     `behavioral_decoding` project (`MultimodalEnsemble`, aaygan29/
     behavioral_decoding): weight each ROI group by its OUT-OF-FOLD
     correlation with the proxy, zero out any group that does not beat
     chance out of fold, normalize the rest. `behavioral_decoding` does this
     across MODALITIES (fMRI/face/behavior) using out-of-fold balanced
     accuracy; here we do the structurally identical thing across ROI
     GROUPS using out-of-fold correlation, because both problems share the
     same shape (several noisy signal sources of very different reliability,
     combined without letting the most overfit one dominate). We do NOT
     reuse `behavioral_decoding`'s classifier/bagging machinery itself (that
     is built for a labeled classification task over fMRI/face/behavior
     feature blocks across SUBJECTS; this is a scalar regression-style
     signal over TIME within one synthetic run) — only the reconciliation
     PRINCIPLE (out-of-fold, chance-floored, weighted combination) transfers.

Sample-size axis disclosure (important, do not skip)
--------------------------------------------------------
The Helmer et al. (2024) instability warning is about N=SUBJECTS in a
brain-behavior individual-differences study (one brain map + one behavior
score per subject). This module, like `validate_against_behavior.py`, has
no real per-subject behavioral outcome to fit against yet (gate 4 remains
BLOCKED per `SCIENCE.md` §13) — it fits PLS/CCA on (T timepoints x 8 ROI
groups) SYNTHETIC data against a SYNTHETIC proxy, i.e. the "N" here is
timepoints/blocks within one run, not subjects. This is a DIFFERENT sample
axis than the one the cited literature addresses. We disclose this
explicitly rather than implying this module has already run the
literature's exact concern: the moment a real per-subject behavioral
channel exists (StudyForrest, Emo-FilM, or NeuroEngage's still-unreleased
questionnaire data — see SCIENCE.md §13), refitting THIS SAME multivariate
fusion at the SUBJECT level with N=4-8 subjects is exactly the regime
Helmer et al. (2024) says is unstable, and that future subject-level fit
must carry the same abstention/caveat discipline this module already
applies at the timepoint level.

Run as a script (synthetic self-test)
--------------------------------------
    python multivariate_fusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression, CCA

sys.path.insert(0, str(Path(__file__).parent))
from buy_moment_detector import ROI_GROUPS, _group_means  # noqa: E402
from validate_against_behavior import (  # noqa: E402
    _synthetic_engagement_process,
    _synthetic_parcel_data_from_engagement,
)
from stat_utils import benjamini_hochberg, gate_effect, bootstrap_ci  # noqa: E402

GROUP_NAMES = list(ROI_GROUPS.keys())
HAND_PICKED_WEIGHTS = np.array([ROI_GROUPS[g]["weight"] for g in GROUP_NAMES])

HELMER_2024_CITATION = (
    "Helmer et al. (2024), 'On the stability of canonical correlation "
    "analysis and partial least squares with application to brain-behavior "
    "associations,' Communications Biology 7:1223 — verified via WebSearch "
    "this session. CCA/PLS brain-behavior weight patterns are highly "
    "unstable below approximately N~1000 subjects; Sapient trains on N=4-8. "
    "Treat every fitted weight/correlation below as EXPLORATORY, not a "
    "validated replacement for the hand-picked ROI_GROUPS weights."
)


def _feature_matrix(parcel_act: np.ndarray) -> np.ndarray:
    """(T, 400) parcel activity -> (T, 8) ROI_GROUPS-restricted group-mean matrix,
    in the fixed GROUP_NAMES order."""
    gm = _group_means(parcel_act)
    return np.column_stack([gm[g] for g in GROUP_NAMES])


def _fit_one(X: np.ndarray, y: np.ndarray, method: str) -> dict[str, Any]:
    """Fit PLS or CCA with 1 component; return weights + latent correlation."""
    y2 = y.reshape(-1, 1)
    if method == "pls":
        model = PLSRegression(n_components=1, scale=True)
        model.fit(X, y2)
        weights = model.x_weights_[:, 0]
        x_scores = model.x_scores_[:, 0]
    elif method == "cca":
        model = CCA(n_components=1, scale=True)
        model.fit(X, y2)
        weights = model.x_weights_[:, 0]
        x_scores, _ = model.transform(X, y2)
        x_scores = x_scores[:, 0]
    else:
        raise ValueError(method)
    latent_r = float(np.corrcoef(x_scores, y)[0, 1])
    return {"weights": weights, "latent_r": latent_r}


def _block_bootstrap_stability(
    X: np.ndarray, y: np.ndarray, method: str, *, block_size: int = 30,
    n_boot: int = 200, seed: int = 0,
) -> dict[str, Any]:
    """Block-bootstrap resampling (blocks, not iid timepoints, to respect the
    autocorrelation in both X and y) to characterize how unstable the fitted
    weights/latent correlation are — the honest disclosure the Helmer et al.
    (2024) instability finding demands.
    """
    rng = np.random.default_rng(seed)
    T = X.shape[0]
    n_blocks = T // block_size
    if n_blocks < 2:
        raise ValueError(f"T={T} too short for block_size={block_size}")

    full = _fit_one(X, y, method)
    full_weights = full["weights"]
    full_sign = np.sign(full_weights)

    boot_weights = np.empty((n_boot, X.shape[1]))
    boot_latent_r = np.empty(n_boot)
    for b in range(n_boot):
        block_starts = rng.integers(0, T - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])
        Xb, yb = X[idx], y[idx]
        try:
            fit = _fit_one(Xb, yb, method)
        except Exception:
            boot_weights[b] = np.nan
            boot_latent_r[b] = np.nan
            continue
        boot_weights[b] = fit["weights"]
        boot_latent_r[b] = fit["latent_r"]

    valid = np.isfinite(boot_latent_r)
    boot_weights_v = boot_weights[valid]
    boot_latent_r_v = boot_latent_r[valid]

    # Sign-agreement rate: fraction of bootstrap fits whose weight sign
    # matches the full-sample fit, per group — the feature-pattern-instability
    # metric Helmer et al. (2024) specifically flag as the more concerning of
    # the two (magnitude instability being the other).
    boot_sign = np.sign(boot_weights_v)
    sign_agreement = (boot_sign == full_sign[None, :]).mean(axis=0)

    return {
        "method": method,
        "n_boot": n_boot,
        "n_valid_boot": int(valid.sum()),
        "block_size": block_size,
        "n_blocks_per_resample": n_blocks,
        "full_sample_weights": {g: round(float(w), 4) for g, w in zip(GROUP_NAMES, full_weights)},
        "full_sample_latent_r": round(full["latent_r"], 4),
        "bootstrap_weight_mean": {g: round(float(m), 4) for g, m in
                                   zip(GROUP_NAMES, boot_weights_v.mean(axis=0))},
        "bootstrap_weight_sd": {g: round(float(s), 4) for g, s in
                                 zip(GROUP_NAMES, boot_weights_v.std(axis=0))},
        "bootstrap_sign_agreement_rate": {g: round(float(s), 4) for g, s in
                                           zip(GROUP_NAMES, sign_agreement)},
        "bootstrap_latent_r_mean": round(float(boot_latent_r_v.mean()), 4),
        "bootstrap_latent_r_sd": round(float(boot_latent_r_v.std()), 4),
        "instability_disclosure": HELMER_2024_CITATION,
        "interpretation": (
            "A group with sign_agreement_rate near 1.0 is a relatively stable "
            "direction across resamples; near 0.5 means the bootstrap can't even "
            "agree on the SIGN of that group's contribution — treat such a group's "
            "weight as noise, not signal, regardless of what the full-sample fit "
            "says. bootstrap_weight_sd large relative to |full_sample_weights| "
            "similarly flags an unstable magnitude."
        ),
    }


def _univariate_alternative(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Lower-variance alternative: simple per-group Pearson correlation against
    the proxy, FDR-corrected across the 8 groups. This is the option-(b)
    fallback the task describes, reported alongside PLS/CCA (not instead of
    it) so the reader can compare a low-variance univariate view against the
    higher-variance multivariate fit.
    """
    from scipy import stats as sstats
    rs = np.empty(X.shape[1])
    ps = np.empty(X.shape[1])
    for i in range(X.shape[1]):
        r, p = sstats.pearsonr(X[:, i], y)
        rs[i] = r
        ps[i] = p
    q = benjamini_hochberg(ps)
    return {
        "per_group_pearson_r": {g: round(float(r), 4) for g, r in zip(GROUP_NAMES, rs)},
        "per_group_p_uncorrected": {g: round(float(p), 6) for g, p in zip(GROUP_NAMES, ps)},
        "per_group_fdr_q": {g: round(float(qq), 6) for g, qq in zip(GROUP_NAMES, q)},
        "note": "Simple univariate correlation per ROI group against the proxy, "
                "BH-FDR corrected across the 8 groups. Much lower-variance than "
                "PLS/CCA (one parameter per group, no joint optimization), so "
                "less susceptible to the Helmer et al. (2024) small-N instability "
                "— appropriate as a sanity check on the multivariate fit above.",
    }


def _accuracy_weighted_combination(X: np.ndarray, y: np.ndarray, *, n_splits: int = 5,
                                     seed: int = 0) -> dict[str, Any]:
    """Out-of-fold, chance-floored, weighted combination of ROI groups —
    structurally adapted from behavioral_decoding's MultimodalEnsemble
    'accuracy_weighted' reconciliation rule (aaygan29/behavioral_decoding,
    src/behavioral_decoding/models/ensemble.py): weight each modality (there:
    fMRI/face/behavior; here: ROI group) by how far its OUT-OF-FOLD score
    exceeds chance, drop any group that doesn't beat chance out of fold, then
    take a weighted combination of the rest. We reuse the PRINCIPLE
    (out-of-fold weighting prevents the noisiest/most overfit source from
    dominating) with time-block folds instead of subject-grouped folds and
    Pearson r instead of balanced accuracy, since this is a scalar
    regression-shaped problem over time rather than a subject-level binary
    classification problem — the ensemble machinery itself (bagged
    classifiers, sklearn BaseEstimator) does not transfer, only the
    weighting rule.
    """
    from scipy import stats as sstats
    T = X.shape[0]
    fold_bounds = np.linspace(0, T, n_splits + 1).astype(int)
    oof_pred = np.zeros(T)
    per_group_oof_r = np.empty(X.shape[1])

    for gi in range(X.shape[1]):
        oof_col = np.full(T, np.nan)
        for k in range(n_splits):
            lo, hi = fold_bounds[k], fold_bounds[k + 1]
            test_idx = np.arange(lo, hi)
            train_idx = np.setdiff1d(np.arange(T), test_idx)
            # simple univariate linear fit (slope, intercept) on the training fold
            slope, intercept = np.polyfit(X[train_idx, gi], y[train_idx], 1)
            oof_col[test_idx] = slope * X[test_idx, gi] + intercept
        r, _ = sstats.pearsonr(oof_col, y)
        per_group_oof_r[gi] = r

    chance = 0.0  # chance correlation is 0
    excess = np.clip(per_group_oof_r - chance, 0, None)  # drop_below_chance behavior
    total = excess.sum()
    if total <= 0:
        weights = np.full(X.shape[1], 1.0 / X.shape[1])
        degenerate = True
    else:
        weights = excess / total
        degenerate = False

    combined = X @ weights
    combined_r = float(np.corrcoef(combined, y)[0, 1])

    return {
        "adapted_from": "behavioral_decoding MultimodalEnsemble 'accuracy_weighted' "
                        "reconciliation rule (aaygan29/behavioral_decoding, "
                        "src/behavioral_decoding/models/ensemble.py) — see docstring "
                        "for exactly what transfers and what doesn't.",
        "n_splits": n_splits,
        "per_group_oof_pearson_r": {g: round(float(r), 4) for g, r in
                                     zip(GROUP_NAMES, per_group_oof_r)},
        "weights": {g: round(float(w), 4) for g, w in zip(GROUP_NAMES, weights)},
        "weights_degenerate": degenerate,
        "combined_score_correlation_with_proxy": round(combined_r, 4),
        "note": ("weights are proportional to each group's OUT-OF-FOLD correlation "
                 "with the proxy (zero-floored); a group that cannot beat chance "
                 "out of fold contributes nothing, same principle as "
                 "behavioral_decoding's drop_below_chance modality weighting."),
    }


def run_fusion(*, T: int = 600, seed: int = 0, method: str = "pls",
               block_size: int = 30, n_boot: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    engagement_gt = _synthetic_engagement_process(T, rng)
    parcel_act = _synthetic_parcel_data_from_engagement(engagement_gt, seed=seed)
    X = _feature_matrix(parcel_act)
    y = engagement_gt

    fit = _fit_one(X, y, method)
    stability = _block_bootstrap_stability(X, y, method, block_size=block_size,
                                            n_boot=n_boot, seed=seed)
    univariate = _univariate_alternative(X, y)
    ensemble = _accuracy_weighted_combination(X, y, seed=seed)

    n_blocks = T // block_size
    gated = gate_effect(
        f"{method}_latent_correlation", effect=fit["latent_r"], n=n_blocks,
        provenance={"method": method, "block_size": block_size,
                    "sample_axis": "time_blocks_within_one_synthetic_run "
                                    "(NOT subjects — see module docstring "
                                    "sample-size-axis disclosure)"})

    # Fitted vs hand-picked comparison
    fitted_weights = fit["weights"]
    fitted_sign = np.sign(fitted_weights)
    hand_sign = np.sign(HAND_PICKED_WEIGHTS)
    sign_agreement = float((fitted_sign == hand_sign).mean())
    fitted_rank = np.argsort(np.argsort(-np.abs(fitted_weights)))
    hand_rank = np.argsort(np.argsort(-np.abs(HAND_PICKED_WEIGHTS)))
    rank_corr = float(np.corrcoef(fitted_rank, hand_rank)[0, 1])

    comparison = {
        "hand_picked_weights": {g: float(w) for g, w in zip(GROUP_NAMES, HAND_PICKED_WEIGHTS)},
        "fitted_weights": {g: round(float(w), 4) for g, w in zip(GROUP_NAMES, fitted_weights)},
        "sign_agreement_fraction": round(sign_agreement, 4),
        "rank_order_correlation": round(rank_corr, 4),
        "note": ("Compares the PLS/CCA-fitted per-group weight vector (on synthetic "
                 "data, one run) against the hand-picked ROI_GROUPS weights, both by "
                 "sign agreement and by rank-order correlation of |weight|. Given the "
                 "Helmer et al. (2024) instability finding and the block-bootstrap "
                 "sign-agreement numbers above, do NOT read a high or low agreement "
                 "here as validating or invalidating the hand-picked weights on its "
                 "own — cross-check against bootstrap_sign_agreement_rate first: a "
                 "fitted weight whose OWN bootstrap sign-agreement is near chance "
                 "(~0.5) tells you nothing reliable about the hand-picked weight it's "
                 "being compared to."),
    }

    return {
        "method": method,
        "n_timepoints": T,
        "fit": {"weights": {g: round(float(w), 4) for g, w in zip(GROUP_NAMES, fitted_weights)},
                "latent_r": round(fit["latent_r"], 4)},
        "gated_latent_correlation": gated.to_dict(),
        "bootstrap_stability": stability,
        "univariate_alternative": univariate,
        "accuracy_weighted_ensemble_combination": ensemble,
        "hand_picked_weight_comparison": comparison,
        "instability_disclosure": HELMER_2024_CITATION,
        "sample_axis_disclosure": (
            "This fit is over T timepoints within ONE synthetic run, not over "
            "subjects. Sapient's real small-N problem (N=4-8 SUBJECTS) is a "
            "different axis; see module docstring 'Sample-size axis disclosure'."
        ),
    }


def main(argv: list[str]) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", choices=["pls", "cca"], default="pls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "multivariate_fusion_out")
    args = ap.parse_args(argv[1:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run_fusion(T=args.t, seed=args.seed, method=args.method, n_boot=args.n_boot)

    out_path = args.out_dir / f"multivariate_fusion_{args.method}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"Multivariate fusion ({args.method.upper()}) — synthetic self-test")
    print("=" * 78)
    print("SMALL-N INSTABILITY DISCLOSURE (Helmer et al. 2024, verified via WebSearch):")
    print(f"  {HELMER_2024_CITATION}")
    print("-" * 78)
    print(f"Fitted latent r: {result['fit']['latent_r']:+.4f}")
    gated = result["gated_latent_correlation"]
    if gated["abstained"]:
        print(f"MDES gate: ABSTAINED — {gated['reason']}")
    else:
        print(f"MDES gate: reported (value={gated['value']:+.4f}, mdes={gated['mdes']:.4f})")
    print("\nBootstrap sign-agreement rate per group (near 0.5 = unstable direction):")
    for g, rate in result["bootstrap_stability"]["bootstrap_sign_agreement_rate"].items():
        print(f"  {g:28s} {rate:.3f}")
    print("\nHand-picked vs fitted weight comparison:")
    print(f"  sign agreement fraction: {result['hand_picked_weight_comparison']['sign_agreement_fraction']:.3f}")
    print(f"  rank-order correlation:  {result['hand_picked_weight_comparison']['rank_order_correlation']:+.3f}")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
