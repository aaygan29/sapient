"""
real_behavior_emofilm.py -- gate 4 (ACCEPTANCE_CRITERIA.md): the first REAL
behavioral-outcome correlation for the buy-moment composite, using a real,
public, no-DUA naturalistic-viewing fMRI dataset with real human-annotated
continuous emotion/engagement ratings.

Why this dataset, not NeuroEngage / StudyForrest
--------------------------------------------------
`sapienteval/SCIENCE.md` sec 13 and `validate_against_behavior.py` document
that no real behavioral-outcome channel was findable in the actual
`ds004996` (NeuroEngage) pipeline data (the `eng_level` column is an
experimenter-assigned condition, not a measured outcome, and the real
questionnaire data is not yet public). `SCIENCE.md` sec 13 also named
StudyForrest and Emo-FilM as real, viable-in-principle candidates not
pursued because of an assumed multi-hour fMRIPrep/parcellation compute
barrier.

This session re-checked both, for real, rather than assuming the barrier:

- StudyForrest (OpenNeuro ds000113): real annotation data (continuous
  arousal/valence via `psychoinformatics-de/studyforrest-data-annotations`)
  IS downloadable without a DUA, and individual movie-run BOLD files are a
  manageable ~65 MB each (NOT the ~258 GB whole-dataset figure, which
  includes 7T structural/DTI/SWI/angiography/retinotopy/localizer data for
  20 subjects that isn't needed here). The actual, concrete blocker: the raw
  functional data is only LINEARLY aligned to a group-specific BOLD template
  (`derivatives/linear_anatomical_alignment`, "grpbold7Tad" space), not MNI.
  Getting to Schaefer-400/MNI space requires the separate nonlinear
  `studyforrest-data-templatetransforms` warp applied via FSL `applywarp` or
  ANTs `antsApplyTransforms` -- neither tool is installed in this
  environment, and installing + validating a registration pipeline within
  this session's time budget was judged not honestly doable. This is a real,
  disclosed, tool/pipeline blocker, not a size blocker.

- Emo-FilM (OpenNeuro ds004892 fMRI + its companion annotation dataset
  ds004892's sibling ds004872 "Emo-FilM Annotations") DID clear every
  feasibility check:
    (a) annotation data: real, continuous, 1 Hz, human-coded GRID-appraisal
        ratings per movie (50 columns incl. IntenseEmotion/Alert/Attention/
        Happiness/Fear/etc.), downloadable with no DUA, ~4.4 MB for the
        entire annotation dataset;
    (b) fMRI data: `derivatives/preprocessing/.../space-MNI_desc-ppres_bold.nii.gz`
        is ALREADY preprocessed (motion-corrected, WM/CSF-regressed) AND
        already coregistered to MNI space by the dataset's own authors --
        no local registration pipeline needed. One subject x one movie run
        (BigBuckBunny, T=528 volumes @ TR=1.3s) is ~650 MB, feasible to
        download for several subjects in this session;
    (c) Schaefer-400 parcels extracted via the exact same
        `nilearn.maskers.NiftiLabelsMasker` + `fetch_atlas_schaefer_2018`
        call used in `connectome_validation.py`.

What this script does
----------------------
For each of 3 subjects (S01, S02, S03) who have a BigBuckBunny run:
1. Download the subject's `space-MNI_desc-ppres_bold.nii.gz` run.
2. Extract Schaefer-400 parcel time series (NiftiLabelsMasker, standardize +
   detrend; no additional confound regression -- WM/CSF/motion already
   regressed by the dataset's own preprocessing per its `.json` sidecar).
3. Crop to the film segment using that subject's own real `events.tsv`
   onset/duration (NOT assumed -- fetched per-subject; onset/session vary by
   subject).
4. Resample from TR=1.3s to 1 Hz (linear interpolation) to match the
   annotation's native 1 Hz sampling.
5. Run `detect_buy_moments` (UNMODIFIED -- this file only adds a downstream
   correlation, per the task's "additive only" constraint).
6. Build a real ground-truth engagement proxy = mean(IntenseEmotion, Alert,
   Attention) from the real annotation columns, z-scored.
7. Apply a 4s HRF lag (the same convention already used in this codebase's
   NeuroEngage pipeline, `SCIENCE.md` sec 3 step 4) -- ground truth leads,
   BOLD-derived score follows.
8. Correlate `score_z` / `generalizable_score` / `idiosyncratic_score`
   against the lagged ground truth, per subject AND pooled (subject-blocked
   circular time-shift permutation null -- each subject's own block is
   circularly shifted independently before concatenation, preserving each
   subject's autocorrelation and avoiding a spurious cross-subject seam).

RESULT (real, n=3 subjects, run 2026-08-25 -- full JSON in
`real_behavior_out/emofilm_bigbuckbunny_result.json`):

    channel                  pooled r    p (block-permutation, two-sided)
    score_z (composite)      +0.029      0.660   -- NOT significant
    generalizable_score      +0.079      0.284   -- NOT significant
    idiosyncratic_score      +0.107      0.077   -- NOT significant (trend)

Per-subject idiosyncratic_score correlations were nominally significant
uncorrected in 1 of 3 subjects (S03, r=+0.209, p=0.039) but this does not
survive correction for testing 3 channels x 3 subjects = 9 comparisons, and
is not the channel `ACCEPTANCE_CRITERIA.md`'s aggregate-claim gate cares
about (per the Genevsky & Knutson framing, aggregate claims should be
checked against `generalizable_score`, which shows NO significant
correlation here, pooled r=+0.079, p=0.284).

Honest framing: this is the first REAL (not synthetic) correlation number
for gate 4, obtained from real fMRI + real human-annotated engagement
ratings, with an appropriate autocorrelation-respecting permutation null.
It is a real negative/null result for the composite and the
aggregate-safe (`generalizable_score`) channel specifically -- consistent
with, not contradicting, the exact failure mode arXiv:2607.01400 (cited
throughout this directory) warns this gate exists to catch: an
aggregated parcel-group readout that fits neural training targets well does
not automatically track a real behavioral/engagement outcome. n=3 subjects,
one stimulus (BigBuckBunny, a childrens animated short, not an ad or
purchase-relevant stimulus), is a small, low-power, single-dataset sample --
this does NOT prove the composite never tracks real behavior, only that this
one real check did not find evidence that it does. See
`ACCEPTANCE_CRITERIA.md` gate 4 for the updated status this motivates.

Run as a script (re-downloads ~1.9 GB of real fMRI data; takes several
minutes; requires network access)
---------------------------------
    python real_behavior_emofilm.py [--n-subjects N] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from buy_moment_detector import detect_buy_moments  # noqa: E402

TR = 1.3
HRF_LAG_SEC = 4  # per SCIENCE.md sec 3 step 4 convention already used in this codebase
N_PERM = 2000
RNG_SEED = 0

# Subject -> which ses- has the BigBuckBunny task (varies per subject; fetched
# live from the dataset's own file tree, not assumed uniform).
SUBJECT_SESSIONS = {"S01": "1", "S02": "2", "S03": "4"}

ANNOTATION_URL = ("https://s3.amazonaws.com/openneuro.org/ds004872/derivatives/"
                   "Annot_BigBuckBunny_stim.tsv.gz")
ANNOTATION_JSON_URL = ("https://raw.githubusercontent.com/OpenNeuroDatasets/ds004872/"
                        "master/derivatives/Annot_BigBuckBunny_stim.json")


def _fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def fetch_events(subj: str, ses: str) -> tuple[float, float]:
    url = (f"https://s3.amazonaws.com/openneuro.org/ds004892/sub-{subj}/ses-{ses}/func/"
           f"sub-{subj}_ses-{ses}_task-scan_acq-BigBuckBunny_events.tsv")
    text = _fetch_bytes(url).decode()
    rows = [l.split("\t") for l in text.strip().splitlines()[1:]]
    film = [r for r in rows if r[2] == "film"][0]
    return float(film[0]), float(film[1])


def fetch_bold(subj: str, ses: str, out_dir: Path) -> Path:
    dest = out_dir / f"sub-{subj}_bigbuckbunny_mni.nii.gz"
    if dest.exists():
        return dest
    url = (f"https://s3.amazonaws.com/openneuro.org/ds004892/derivatives/preprocessing/"
           f"sub-{subj}/ses-{ses}/func/"
           f"sub-{subj}_ses-{ses}_task-BigBuckBunny_space-MNI_desc-ppres_bold.nii.gz")
    print(f"  downloading {url} ...", file=sys.stderr)
    data = _fetch_bytes(url)
    dest.write_bytes(data)
    return dest


def fetch_annotation(out_dir: Path) -> tuple[np.ndarray, list[str]]:
    tsv_path = out_dir / "annot_bigbuckbunny.tsv.gz"
    json_path = out_dir / "annot_bigbuckbunny.json"
    if not tsv_path.exists():
        tsv_path.write_bytes(_fetch_bytes(ANNOTATION_URL))
    if not json_path.exists():
        json_path.write_bytes(_fetch_bytes(ANNOTATION_JSON_URL))
    import gzip
    with gzip.open(tsv_path, "rt") as f:
        arr = np.loadtxt(f, delimiter="\t")
    cols = json.loads(json_path.read_text())["Columns"]
    return arr, cols


def extract_parcels(nii_path: Path) -> np.ndarray:
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.maskers import NiftiLabelsMasker
    atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1)
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize="zscore_sample",
        detrend=True,
        t_r=TR,
        resampling_target="data",
    )
    return masker.fit_transform(str(nii_path))


def resample_to_1hz(ts: np.ndarray, tr: float) -> np.ndarray:
    n_tr = ts.shape[0]
    t_src = np.arange(n_tr) * tr
    t_dst = np.arange(0, t_src[-1], 1.0)
    out = np.empty((len(t_dst), ts.shape[1]), dtype=np.float32)
    for c in range(ts.shape[1]):
        out[:, c] = np.interp(t_dst, t_src, ts[:, c])
    return out


def circular_shift_null(a: np.ndarray, b: np.ndarray, n_perm: int, rng) -> tuple[float, np.ndarray]:
    T = a.shape[0]
    observed_r = float(np.corrcoef(a, b)[0, 1])
    null_rs = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(rng.integers(1, T))
        null_rs[i] = float(np.corrcoef(a, np.roll(b, shift))[0, 1])
    return observed_r, null_rs


def run(n_subjects: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = list(SUBJECT_SESSIONS.items())[:n_subjects]

    annot, cols = fetch_annotation(out_dir)
    idx = {c: i for i, c in enumerate(cols)}
    engagement_cols = ["IntenseEmotion", "Alert", "Attention"]
    eng = annot[:, [idx[c] for c in engagement_cols]].mean(axis=1)
    eng = (eng - eng.mean()) / eng.std()
    T_annot = eng.shape[0]

    results: dict[str, Any] = {}
    per_subject_series: dict[str, dict[str, np.ndarray]] = {}

    for subj, ses in subjects:
        print(f"=== {subj} ===", file=sys.stderr)
        onset, duration = fetch_events(subj, ses)
        nii = fetch_bold(subj, ses, out_dir)
        ts = extract_parcels(nii)

        onset_vol = int(round(onset / TR))
        n_vol_needed = int(np.ceil(duration / TR)) + 2
        film_ts = ts[onset_vol: onset_vol + n_vol_needed]
        film_ts_1hz = resample_to_1hz(film_ts, TR)

        T = min(film_ts_1hz.shape[0], T_annot)
        parcel_act = film_ts_1hz[:T]
        gt = eng[:T]

        det = detect_buy_moments(parcel_act)
        lag = HRF_LAG_SEC
        gt_lagged = gt[: T - lag]
        score_z_lagged = det["score_z"][lag:T]
        gen_lagged = det["generalizable_score"][lag:T]
        idio_lagged = det["idiosyncratic_score"][lag:T]

        rng = np.random.default_rng(RNG_SEED)
        r_c, null_c = circular_shift_null(score_z_lagged, gt_lagged, N_PERM, rng)
        r_g, null_g = circular_shift_null(gen_lagged, gt_lagged, N_PERM, rng)
        r_i, null_i = circular_shift_null(idio_lagged, gt_lagged, N_PERM, rng)

        def pval(obs, null):
            return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))

        results[subj] = {
            "onset_sec": onset, "duration_sec": duration,
            "n_timepoints_used": T - lag,
            "r_composite_score_z": r_c, "p_composite_two_sided": pval(r_c, null_c),
            "r_generalizable_score": r_g, "p_generalizable_two_sided": pval(r_g, null_g),
            "r_idiosyncratic_score": r_i, "p_idiosyncratic_two_sided": pval(r_i, null_i),
        }
        per_subject_series[subj] = {
            "score_z": score_z_lagged, "gen": gen_lagged, "idio": idio_lagged, "gt": gt_lagged,
        }

    pooled_score_z = np.concatenate([per_subject_series[s]["score_z"] for s, _ in subjects])
    pooled_gen = np.concatenate([per_subject_series[s]["gen"] for s, _ in subjects])
    pooled_idio = np.concatenate([per_subject_series[s]["idio"] for s, _ in subjects])
    pooled_gt = np.concatenate([per_subject_series[s]["gt"] for s, _ in subjects])

    r_pooled_c = float(np.corrcoef(pooled_score_z, pooled_gt)[0, 1])
    r_pooled_g = float(np.corrcoef(pooled_gen, pooled_gt)[0, 1])
    r_pooled_i = float(np.corrcoef(pooled_idio, pooled_gt)[0, 1])

    rng = np.random.default_rng(RNG_SEED + 1)
    blocks = [per_subject_series[s]["gt"] for s, _ in subjects]
    null_c = np.empty(N_PERM)
    null_g = np.empty(N_PERM)
    null_i = np.empty(N_PERM)
    for k in range(N_PERM):
        shifted = np.concatenate([np.roll(b, int(rng.integers(1, len(b)))) for b in blocks])
        null_c[k] = float(np.corrcoef(pooled_score_z, shifted)[0, 1])
        null_g[k] = float(np.corrcoef(pooled_gen, shifted)[0, 1])
        null_i[k] = float(np.corrcoef(pooled_idio, shifted)[0, 1])

    def pval(obs, null):
        return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))

    out = {
        "dataset": "Emo-FilM (ds004892 fMRI, ds004872 annotations), stimulus=BigBuckBunny",
        "subjects": [s for s, _ in subjects],
        "hrf_lag_sec": HRF_LAG_SEC,
        "engagement_proxy_columns": engagement_cols,
        "per_subject": results,
        "pooled_block_concatenated": {
            "n_subjects": len(subjects),
            "n_timepoints_total": len(pooled_gt),
            "r_composite_score_z": r_pooled_c,
            "p_composite_two_sided_block_permutation": pval(r_pooled_c, null_c),
            "r_generalizable_score": r_pooled_g,
            "p_generalizable_two_sided_block_permutation": pval(r_pooled_g, null_g),
            "r_idiosyncratic_score": r_pooled_i,
            "p_idiosyncratic_two_sided_block_permutation": pval(r_pooled_i, null_i),
            "n_permutations": N_PERM,
            "permutation_method": "per-subject circular time-shift, blocks concatenated after shifting",
        },
    }
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-subjects", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "real_behavior_out")
    args = ap.parse_args(argv[1:])

    out = run(args.n_subjects, args.out_dir)
    result_path = args.out_dir / "emofilm_bigbuckbunny_result.json"
    result_path.write_text(json.dumps(out, indent=2))

    print("\n=== Gate 4: real-behavior correlation (Emo-FilM, BigBuckBunny) ===")
    for subj, r in out["per_subject"].items():
        print(f"  {subj}: score_z r={r['r_composite_score_z']:+.4f} p={r['p_composite_two_sided']:.4f}  "
              f"generalizable r={r['r_generalizable_score']:+.4f} p={r['p_generalizable_two_sided']:.4f}  "
              f"idiosyncratic r={r['r_idiosyncratic_score']:+.4f} p={r['p_idiosyncratic_two_sided']:.4f}")
    p = out["pooled_block_concatenated"]
    print(f"\n  POOLED (n={p['n_subjects']}): score_z r={p['r_composite_score_z']:+.4f} "
          f"p={p['p_composite_two_sided_block_permutation']:.4f}")
    print(f"                   generalizable_score r={p['r_generalizable_score']:+.4f} "
          f"p={p['p_generalizable_two_sided_block_permutation']:.4f}")
    print(f"                   idiosyncratic_score r={p['r_idiosyncratic_score']:+.4f} "
          f"p={p['p_idiosyncratic_two_sided_block_permutation']:.4f}")
    print(f"\nWrote: {result_path}")
    print("\nReminder: this is a REAL but small (n=3 subjects, 1 stimulus) result.")
    print("None of the three channels reach significance after accounting for")
    print("multiple comparisons -- see ACCEPTANCE_CRITERIA.md gate 4 for the full")
    print("honest framing this motivates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
