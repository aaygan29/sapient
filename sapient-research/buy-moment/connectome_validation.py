"""
connectome_validation.py — real-connectome grounding for buy_moment_detector.py's
`ROI_GROUPS` parcel bundles.

Motivation
----------
`buy_moment_detector.py`'s docstring flags a known limitation: `ROI_GROUPS` bundles
Schaefer-400 parcels together by label-string matching against the nilearn atlas
("Limbic_OFC", "Default_PFC_vmPFC", etc.), not by anatomically/functionally curated
grouping. This module adds an empirical check, using real public data where
feasible, of whether those bundles actually cohere as connectivity units.

Two checks, run separately and reported with an explicit real-vs-literature label
on every number (never silently mixed):

1. REAL-DATA CHECK (`real_data_internal_connectivity`)
   Uses `nilearn.datasets.fetch_development_fmri` — the "development dataset"
   (Richardson et al.; Pixar "Partly Cloudy" naturalistic movie-watching fMRI,
   preprocessed derivatives hosted on OSF, CC0-style public release, no data-use
   agreement, fetchable via `pip install nilearn`; original raw data on OpenNeuro
   ds000228). This is genuine human fMRI connectivity data — not restricted HCP,
   no DUA. It is naturalistic AUDIOVISUAL NARRATIVE viewing, which is a closer
   task analog to Sapient's ad/video-viewing use case than a typical resting-state
   scan, though it is a children's animated short, not an ad.
   We fetch adult subjects only (`Child_Adult == 'adult'`), extract Schaefer-400
   parcel time series with `NiftiLabelsMasker` (standardized, confound-regressed,
   band-pass filtered), average each subject's pairwise Pearson connectivity
   matrix across subjects, and for each `ROI_GROUPS` bundle compare:
     - mean within-group pairwise connectivity (all parcel pairs inside the group)
   against
     - mean connectivity of 1000 random parcel sets of the same size (drawn from
       all 400 parcels), giving a null distribution and an empirical percentile.
   This tells us whether the bundle is MORE internally coherent than chance-sized
   random parcel sets in real human connectivity data — not whether it is the
   "correct" grouping, and not validated against any buy-related behavior.

2. LITERATURE-ONLY CHECK (`yeo7_network_consistency`)
   Cross-tabulates each `ROI_GROUPS` bundle against the Schaefer-2018 atlas's own
   published Yeo-7 canonical network labels (shipped by nilearn's
   `fetch_atlas_schaefer_2018`, i.e. Schaefer et al. 2018, Cerebral Cortex,
   "Local-Global Parcellation of the Human Cerebral Cortex..."). This is real
   published data (not synthetic), but it is a structural/label check, not an
   independent functional-connectivity measurement — reported separately from
   check 1 and explicitly labeled as such.

Honesty rules (matching phase-1 style in this directory)
---------------------------------------------------------
- Every number this module prints or returns is tagged with its provenance:
  "real_data" (fetched fMRI, computed connectivity) or "literature_label"
  (Schaefer atlas's own published network assignment, no new computation beyond
  cross-tabulation).
- No claim here says the ROI_GROUPS bundles predict buying behavior. This module
  only checks internal connectivity coherence — a necessary-but-not-sufficient
  precondition for a parcel bundle to be a meaningful single "channel".
- If the real-data fetch fails (no network, OSF outage), the module falls back
  to reporting the Yeo-7 structural check alone and says so explicitly; it does
  NOT fabricate connectivity numbers.

Run as a script
---------------
    python connectome_validation.py [--n-subjects N] [--out-dir DIR]

Or import:
    from connectome_validation import run_full_validation
    result = run_full_validation()
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from buy_moment_detector import ROI_GROUPS  # noqa: E402
from stat_utils import (  # noqa: E402
    benjamini_hochberg,
    max_stat_fwer,
    eb_shrink_subject_effects,
)

N_PARCELS = 400
N_NULL_DRAWS = 1000
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Check 2: Yeo-7 structural consistency (real published labels, no new data)
# ---------------------------------------------------------------------------

def yeo7_network_consistency() -> dict[str, Any]:
    """For each ROI_GROUPS bundle, report how its parcels distribute across the
    Schaefer atlas's own published Yeo-7 network labels (Schaefer et al. 2018).

    provenance: "literature_label" — real published atlas data, structural
    cross-tab only, no independent connectivity computed here.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(n_rois=N_PARCELS, yeo_networks=7, resolution_mm=1)
    labels = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in atlas.labels]
    # labels[0] == "Background"; labels[1..400] correspond to 0-based parcel
    # columns 0..399 in the (T, 400) matrix convention this codebase uses.
    parcel_labels = labels[1:]
    assert len(parcel_labels) == N_PARCELS, f"expected {N_PARCELS} labels, got {len(parcel_labels)}"

    def yeo_network_of(label: str) -> str:
        # e.g. "7Networks_LH_Limbic_OFC_1" -> "Limbic"
        parts = label.split("_")
        return parts[2] if len(parts) > 2 else "Unknown"

    out: dict[str, Any] = {}
    for group_name, spec in ROI_GROUPS.items():
        cols = spec["parcels"]
        nets = [yeo_network_of(parcel_labels[c]) for c in cols]
        counts: dict[str, int] = {}
        for n in nets:
            counts[n] = counts.get(n, 0) + 1
        dominant_net, dominant_n = max(counts.items(), key=lambda kv: kv[1])
        purity = dominant_n / len(nets) if nets else 0.0
        out[group_name] = {
            "provenance": "literature_label",
            "n_parcels": len(cols),
            "yeo7_network_counts": counts,
            "dominant_network": dominant_net,
            "purity": round(purity, 3),
            "single_network": purity == 1.0,
            "note": ("bundle sits entirely within one Yeo-7 network"
                     if purity == 1.0 else
                     "bundle spans multiple Yeo-7 networks — a deliberate "
                     "cross-network combination (see ROI_GROUPS rationale), "
                     "or an unintended anatomical mixing; this check cannot "
                     "distinguish the two, only flag it for manual review."),
        }
    return out


# ---------------------------------------------------------------------------
# Check 1: real-data internal connectivity (fetched fMRI, computed correlations)
# ---------------------------------------------------------------------------

def _fetch_adult_subjects(n_subjects_to_scan: int) -> tuple[list[str], Any]:
    """Fetch up to n_subjects_to_scan development-dataset subjects and return
    the func file paths + confounds for the ones flagged 'adult' in phenotypics.
    """
    from nilearn.datasets import fetch_development_fmri

    data = fetch_development_fmri(n_subjects=n_subjects_to_scan)
    pheno = data.phenotypic
    adult_mask = pheno["Child_Adult"].to_numpy() == "adult"
    idx = np.nonzero(adult_mask)[0]
    funcs = [data.func[i] for i in idx]
    confounds = [data.confounds[i] for i in idx]
    ids = pheno["participant_id"].to_numpy()[idx].tolist()
    return ids, list(zip(funcs, confounds))


def _subject_connectivity(func_path: str, confounds_path: str, atlas_maps: str) -> np.ndarray | None:
    """Extract Schaefer-400 parcel time series for one subject and return the
    (400, 400) Pearson connectivity matrix, or None if extraction fails."""
    from nilearn.maskers import NiftiLabelsMasker

    try:
        masker = NiftiLabelsMasker(
            labels_img=atlas_maps,
            standardize="zscore_sample",
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=2.0,  # development_fmri dataset TR
            resampling_target="data",
        )
        ts = masker.fit_transform(func_path, confounds=confounds_path)
    except Exception as e:  # noqa: BLE001 — report and skip, don't crash the batch
        print(f"  [skip] {func_path}: {type(e).__name__}: {e}", file=sys.stderr)
        return None

    if ts.shape[1] != N_PARCELS:
        print(f"  [skip] {func_path}: got {ts.shape[1]} parcels, expected {N_PARCELS} "
              f"(background parcel likely absent from this subject's coverage)",
              file=sys.stderr)
        return None

    conn = np.corrcoef(ts.T)
    return conn


def real_data_internal_connectivity(n_subjects_to_scan: int = 40,
                                     rng_seed: int = RNG_SEED) -> dict[str, Any]:
    """For each ROI_GROUPS bundle, compare its mean within-group pairwise
    connectivity (averaged across real adult subjects' connectomes) against a
    null distribution of same-size random parcel sets.

    provenance: "real_data" — fetched fMRI (nilearn `fetch_development_fmri`,
    Richardson et al. movie-watching dataset, OSF/OpenNeuro ds000228, public,
    no DUA), Schaefer-400 parcellation via NiftiLabelsMasker, Pearson
    connectivity, averaged across subjects who were successfully extracted.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(n_rois=N_PARCELS, yeo_networks=7, resolution_mm=1)

    print(f"Fetching development_fmri (scanning up to {n_subjects_to_scan} subjects "
          f"for adults)...", file=sys.stderr)
    try:
        ids, pairs = _fetch_adult_subjects(n_subjects_to_scan)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "fetch_failed",
            "error": f"{type(e).__name__}: {e}",
            "note": "Real-data fetch failed (no network access, OSF outage, or "
                    "similar). No connectivity numbers are reported here — see "
                    "yeo7_network_consistency() for the literature-only fallback.",
        }

    if not pairs:
        return {"status": "no_adult_subjects_found",
                "note": "fetch_development_fmri returned no subjects flagged "
                        "'adult' within the scanned range; increase n_subjects_to_scan."}

    print(f"  found {len(ids)} adult subjects: {ids}", file=sys.stderr)

    conns = []
    used_ids = []
    for sid, (func, conf) in zip(ids, pairs):
        print(f"  extracting {sid} ...", file=sys.stderr)
        c = _subject_connectivity(func, conf, atlas.maps)
        if c is not None and np.isfinite(c).all():
            conns.append(c)
            used_ids.append(sid)

    if not conns:
        return {"status": "no_usable_subjects",
                "note": "All fetched adult subjects failed parcel extraction "
                        "(see stderr for per-subject errors)."}

    group_conn = np.mean(np.stack(conns, axis=0), axis=0)  # (400, 400), Fisher-naive average
    n_subjects_used = len(conns)

    rng = np.random.default_rng(rng_seed)

    def mean_offdiag(conn: np.ndarray, cols: np.ndarray) -> float:
        if len(cols) < 2:
            return float("nan")
        sub = conn[np.ix_(cols, cols)]
        iu = np.triu_indices(len(cols), k=1)
        return float(sub[iu].mean())

    all_parcels = np.arange(N_PARCELS)
    group_names = list(ROI_GROUPS.keys())

    # -----------------------------------------------------------------------
    # ORIGINAL (pseudo-replicated) pooled-average check — kept, unchanged
    # arithmetic, for continuity with SCIENCE.md §12a / connectome_validation.json
    # consumers. Labeled explicitly as the weaker of the two checks below.
    # -----------------------------------------------------------------------
    results: dict[str, Any] = {}
    # Also collect the full null-draw matrix per group so we can build the
    # max-statistic family-wise null across groups (Nichols & Holmes 2002).
    null_matrix = np.empty((N_NULL_DRAWS, len(group_names)))
    observed_vec = np.empty(len(group_names))
    for gi, group_name in enumerate(group_names):
        spec = ROI_GROUPS[group_name]
        cols = np.asarray(spec["parcels"], dtype=int)
        observed = mean_offdiag(group_conn, cols)
        observed_vec[gi] = observed

        null_vals = np.empty(N_NULL_DRAWS)
        for i in range(N_NULL_DRAWS):
            draw = rng.choice(all_parcels, size=len(cols), replace=False)
            null_vals[i] = mean_offdiag(group_conn, draw)
        null_matrix[:, gi] = null_vals

        percentile = float((null_vals < observed).mean())  # fraction of null draws beaten
        # Uncorrected one-sided p-value from the same null draws (for FDR input).
        p_uncorrected = float((np.sum(null_vals >= observed) + 1) / (N_NULL_DRAWS + 1))
        results[group_name] = {
            "provenance": "real_data",
            "approach": "pooled_average (pseudo-replicated: 8 subjects averaged into one "
                        "(400,400) matrix BEFORE nulling — treats 8 subjects as one data "
                        "point; kept for continuity, see subject_level_connectivity for the "
                        "corrected per-subject check)",
            "n_parcels": len(cols),
            "observed_mean_within_group_r": round(observed, 4),
            "null_mean_random_r": round(float(null_vals.mean()), 4),
            "null_sd_random_r": round(float(null_vals.std()), 4),
            "empirical_percentile_vs_random": round(percentile, 4),
            "beats_random_null_p95": bool(percentile >= 0.95),
            "p_uncorrected": round(p_uncorrected, 6),
        }

    # ---- Multiple-comparisons correction across the 8 group-level tests ----
    # (2026-08-26 addition; additive fields only, does not alter the numbers
    # above.) Two corrections, both reported, per task instructions:
    p_uncorrected_vec = np.array([results[g]["p_uncorrected"] for g in group_names])

    # (i) Benjamini-Hochberg FDR (Benjamini & Hochberg 1995, JRSS-B 57:289-300).
    q_values = benjamini_hochberg(p_uncorrected_vec)

    # (ii) Max-statistic family-wise permutation correction (Nichols & Holmes
    # 2002, Hum Brain Mapp 15:1-25) — the max, across all 8 groups' null
    # draws, at each of the 1000 permutations builds one family-wise null;
    # each group's observed value is tested against that single corrected null.
    p_fwer_vec = max_stat_fwer(observed_vec, null_matrix)

    for gi, group_name in enumerate(group_names):
        results[group_name]["fdr_q_value"] = round(float(q_values[gi]), 6)
        results[group_name]["fdr_significant_q05"] = bool(q_values[gi] < 0.05)
        results[group_name]["fwer_p_value"] = round(float(p_fwer_vec[gi]), 6)
        results[group_name]["fwer_significant_p05"] = bool(p_fwer_vec[gi] < 0.05)

    multiple_comparisons_correction = {
        "method_fdr": "Benjamini & Hochberg (1995), JRSS-B 57(1):289-300 — "
                      "step-up FDR control across the 8 group-level p-values.",
        "method_fwer": "Nichols & Holmes (2002), Hum Brain Mapp 15(1):1-25 — "
                      "max-statistic permutation correction: the max within-group "
                      "connectivity value across all 8 groups' null draws at each "
                      "of the 1000 permutations builds one family-wise null "
                      "distribution; each group tested against that single "
                      "corrected null.",
        "n_tests": len(group_names),
        "n_permutations": N_NULL_DRAWS,
        "note": ("Both corrections are reported ADDITIVELY alongside the original "
                 "uncorrected p_uncorrected/empirical_percentile_vs_random fields "
                 "per group above — neither replaces the original numbers. Prior "
                 "to this pass, 8 independent 'beats random null p95' calls carried "
                 "no correction for testing 8 hypotheses at once."),
    }

    # -----------------------------------------------------------------------
    # CORRECTED (subject-respecting) check: per-subject within-group
    # connectivity + per-subject null, then a group-level test across the 8
    # subjects that does NOT average subjects into one data point first.
    # (2026-08-26 addition, fixes the pseudo-replication flagged in the task.)
    # -----------------------------------------------------------------------
    subject_level: dict[str, Any] = {}
    # Per-subject null draws use fewer permutations (200) than the pooled
    # check's 1000 for compute reasons (8x the null-generation cost) — this
    # is disclosed, not hidden, and 200 draws is still enough for a stable
    # z-score estimate at the tolerances used here.
    N_NULL_DRAWS_PER_SUBJECT = 200
    rng_subj = np.random.default_rng(rng_seed + 1)

    for group_name in group_names:
        spec = ROI_GROUPS[group_name]
        cols = np.asarray(spec["parcels"], dtype=int)
        subj_observed = np.empty(n_subjects_used)
        subj_z = np.empty(n_subjects_used)
        for si, conn in enumerate(conns):
            obs = mean_offdiag(conn, cols)
            subj_observed[si] = obs
            null_vals = np.empty(N_NULL_DRAWS_PER_SUBJECT)
            for i in range(N_NULL_DRAWS_PER_SUBJECT):
                draw = rng_subj.choice(all_parcels, size=len(cols), replace=False)
                null_vals[i] = mean_offdiag(conn, draw)
            null_mu, null_sd = float(null_vals.mean()), float(null_vals.std())
            subj_z[si] = (obs - null_mu) / null_sd if null_sd > 0 else 0.0

        # Wilcoxon signed-rank test: are the 8 subject-level null-normalized
        # z-scores systematically different from 0 (i.e. from each subject's
        # OWN random-null baseline, not a pooled one)? Non-parametric, no
        # normality assumption, appropriate for n=8.
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(subj_z, alternative="greater")
        except ValueError as e:
            # wilcoxon() raises if all differences are zero or n too small
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

        eb = eb_shrink_subject_effects(subj_z)

        subject_level[group_name] = {
            "provenance": "real_data",
            "approach": "subject_level (corrected: each of the 8 subjects' within-group "
                        "connectivity is nulled against ITS OWN random-parcel draws before "
                        "any pooling across subjects — no pseudo-replication)",
            "n_subjects": n_subjects_used,
            "subject_ids": used_ids,
            "per_subject_observed_r": [round(float(x), 4) for x in subj_observed],
            "per_subject_null_z": [round(float(x), 4) for x in subj_z],
            "mean_null_z": round(float(subj_z.mean()), 4),
            "sd_null_z": round(float(subj_z.std()), 4),
            "wilcoxon_signed_rank": {
                "statistic": (round(float(wilcoxon_stat), 4)
                              if np.isfinite(wilcoxon_stat) else None),
                "p_value_one_sided_greater_than_0": (
                    round(float(wilcoxon_p), 6) if np.isfinite(wilcoxon_p) else None),
                "note": "tests whether the 8 per-subject null-normalized z-scores are "
                        "systematically > 0, i.e. whether the group is more internally "
                        "connected than its own random-parcel null IN EACH SUBJECT, not "
                        "whether one pooled group-average matrix beats one pooled null.",
            },
            "empirical_bayes_shrinkage": eb,
        }

    # FDR/FWER correction on the 8 subject-level Wilcoxon p-values too.
    subj_p_vec = np.array([
        (subject_level[g]["wilcoxon_signed_rank"]["p_value_one_sided_greater_than_0"]
         if subject_level[g]["wilcoxon_signed_rank"]["p_value_one_sided_greater_than_0"] is not None
         else 1.0)
        for g in group_names
    ])
    subj_q_values = benjamini_hochberg(subj_p_vec)
    for gi, group_name in enumerate(group_names):
        subject_level[group_name]["wilcoxon_signed_rank"]["fdr_q_value"] = round(
            float(subj_q_values[gi]), 6)
        subject_level[group_name]["wilcoxon_signed_rank"]["fdr_significant_q05"] = bool(
            subj_q_values[gi] < 0.05)

    return {
        "status": "ok",
        "n_subjects_scanned": n_subjects_to_scan,
        "n_adult_subjects_found": len(ids),
        "n_subjects_used": n_subjects_used,
        "subject_ids_used": used_ids,
        "dataset": "nilearn.datasets.fetch_development_fmri "
                   "(Richardson et al., naturalistic movie-watching fMRI; "
                   "OpenNeuro ds000228; public, no data-use agreement)",
        "n_null_draws": N_NULL_DRAWS,
        "groups": results,
        "multiple_comparisons_correction": multiple_comparisons_correction,
        "subject_level_connectivity": subject_level,
        "pooled_vs_subject_level_disclosure": (
            "'groups' above is the ORIGINAL (2026-08-26 phase-1) pooled-average "
            "check: 8 subjects' connectivity matrices averaged into one (400,400) "
            "matrix BEFORE comparing to a null. This treats 8 subjects as a single "
            "data point, which inflates apparent significance and ignores "
            "between-subject variance (pseudo-replication). "
            "'subject_level_connectivity' is the CORRECTED check added this pass: "
            "each subject's within-group connectivity is nulled against its own "
            "random-parcel draws first, THEN a Wilcoxon signed-rank test asks "
            "whether the resulting 8 per-subject effects are systematically above "
            "zero. Both are reported; the subject-level check is the one that "
            "should be trusted for any claim resting on 'is this true across "
            "subjects', not the pooled-average one."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------

def run_full_validation(n_subjects_to_scan: int = 40) -> dict[str, Any]:
    print("=== Check 1: real-data internal connectivity ===", file=sys.stderr)
    real = real_data_internal_connectivity(n_subjects_to_scan=n_subjects_to_scan)
    print("=== Check 2: Yeo-7 structural consistency (literature label) ===", file=sys.stderr)
    yeo = yeo7_network_consistency()
    return {
        "real_data_internal_connectivity": real,
        "yeo7_network_consistency": yeo,
        "disclosure": (
            "real_data_internal_connectivity numbers come from fetched, real "
            "human fMRI connectivity (see its 'dataset' field) — genuine "
            "empirical validation of within-group connectivity coherence, NOT "
            "of any link to buying behavior. yeo7_network_consistency numbers "
            "come from the Schaefer atlas's own published Yeo-7 network labels "
            "— real published data, but a structural cross-tab, not an "
            "independently measured connectivity check. The two are reported "
            "separately and must not be conflated."
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subjects", type=int, default=40,
                         help="how many development_fmri subjects to scan for adults")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "connectome_validation_out")
    args = parser.parse_args(argv[1:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run_full_validation(n_subjects_to_scan=args.n_subjects)

    out_path = args.out_dir / "connectome_validation.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote: {out_path}")

    real = result["real_data_internal_connectivity"]
    print("\n--- Summary (real_data_internal_connectivity) ---")
    if real.get("status") == "ok":
        print(f"n_subjects_used={real['n_subjects_used']}  ids={real['subject_ids_used']}")
        for g, v in real["groups"].items():
            flag = "PASS (>95th pct)" if v["beats_random_null_p95"] else "no"
            print(f"  {g:28s} r={v['observed_mean_within_group_r']:+.3f}  "
                  f"null_mean={v['null_mean_random_r']:+.3f}  "
                  f"pctile={v['empirical_percentile_vs_random']:.3f}  "
                  f"p_unc={v['p_uncorrected']:.4f}  fdr_q={v['fdr_q_value']:.4f}  "
                  f"fwer_p={v['fwer_p_value']:.4f}  {flag}")

        print("\n--- Summary (subject_level_connectivity, corrected, n=8) ---")
        for g, v in real["subject_level_connectivity"].items():
            wx = v["wilcoxon_signed_rank"]
            print(f"  {g:28s} mean_z={v['mean_null_z']:+.3f}  sd_z={v['sd_null_z']:.3f}  "
                  f"wilcoxon_p={wx['p_value_one_sided_greater_than_0']}  "
                  f"fdr_q={wx['fdr_q_value']}")
    else:
        print(f"status={real.get('status')}: {real.get('note') or real.get('error')}")

    print("\n--- Summary (yeo7_network_consistency, literature label) ---")
    for g, v in result["yeo7_network_consistency"].items():
        print(f"  {g:28s} dominant={v['dominant_network']:12s} purity={v['purity']:.2f}  "
              f"counts={v['yeo7_network_counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
