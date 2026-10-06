"""
03_parcellate.py — Schaefer-400 7-Networks parcellation of preprocessed BOLD.

Reads:  sapienteval/data/derivatives/fmriprep/sub-XX/func/sub-XX_task-*_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
Writes: sapienteval/data/parcellated/sub-XX_task-{task}_run-{run}.npy  shape (T, 400) float32

Uses nilearn.maskers.NiftiLabelsMasker with the Schaefer-400 7-Networks atlas
at 2mm resolution (matching the fMRIPrep output --output-spaces ...:res-2).

Standardization: zscore_sample. Low-pass 0.1 Hz, high-pass 0.01 Hz, TR=2.0s.

Idempotent: skips runs whose .npy output already exists.

Usage:
  python sapienteval/03_parcellate.py
  python sapienteval/03_parcellate.py --subjects sub-01,sub-02
  python sapienteval/03_parcellate.py --deriv-dir /path/to/derivatives
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DERIV_DIR = REPO_ROOT / "data" / "derivatives"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "parcellated"

# fMRIPrep output filename pattern we expect (volumetric, MNI152NLin2009cAsym, 2mm).
# Example: sub-01_task-engage_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
BOLD_GLOB = "*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
BOLD_RE = re.compile(
    r"^(?P<subject>sub-[A-Za-z0-9]+)"
    r"_task-(?P<task>[A-Za-z0-9]+)"
    r"(?:_run-(?P<run>[0-9]+))?"
    r"_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold\.nii\.gz$"
)

# Bandpass + TR settings (NeuroEngage / ds004996).
TR_SECONDS = 2.0
LOW_PASS_HZ = 0.1
HIGH_PASS_HZ = 0.01


def log(msg: str) -> None:
    print(msg, flush=True)


def err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def discover_subjects(fmriprep_root: Path) -> list[str]:
    """Return all sub-XX directories under derivatives/fmriprep/."""
    if not fmriprep_root.exists():
        return []
    return sorted(
        p.name
        for p in fmriprep_root.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )


def parse_subjects(arg: str | None, fmriprep_root: Path) -> list[str]:
    if not arg:
        subs = discover_subjects(fmriprep_root)
        if not subs:
            err(f"ERROR: no sub-XX directories found under {fmriprep_root}")
            sys.exit(2)
        return subs
    subs = [s.strip() for s in arg.split(",") if s.strip()]
    for s in subs:
        if not s.startswith("sub-"):
            err(f"ERROR: subject ID must start with 'sub-': got {s!r}")
            sys.exit(2)
    return subs


def parse_run_metadata(filename: str) -> tuple[str, str, str] | None:
    """Return (subject, task, run) from a fMRIPrep BOLD filename, or None."""
    m = BOLD_RE.match(filename)
    if not m:
        return None
    return (
        m.group("subject"),
        m.group("task"),
        m.group("run") or "1",
    )


def output_path(out_dir: Path, subject: str, task: str, run: str) -> Path:
    return out_dir / f"{subject}_task-{task}_run-{run}.npy"


def fetch_atlas():
    """Lazy-load the Schaefer-400 7-Networks 2mm volumetric atlas.

    Cached by nilearn under ~/nilearn_data/ on first call.
    Returns the atlas object with `.maps` (a labels image) and `.labels`.
    """
    from nilearn import datasets  # local import keeps --dry-run fast

    return datasets.fetch_atlas_schaefer_2018(
        n_rois=400,
        yeo_networks=7,
        resolution_mm=2,
    )


def build_masker(atlas_maps):
    """Build the NiftiLabelsMasker once per process and reuse across runs."""
    from nilearn.maskers import NiftiLabelsMasker

    return NiftiLabelsMasker(
        labels_img=atlas_maps,
        standardize="zscore_sample",
        detrend=True,
        low_pass=LOW_PASS_HZ,
        high_pass=HIGH_PASS_HZ,
        t_r=TR_SECONDS,
        memory=None,
        verbose=0,
    )


def parcellate_run(masker, bold_path: Path):
    """Run nilearn on one BOLD file. Returns a (T, 400) numpy array."""
    import numpy as np  # local imports keep --dry-run cheap

    arr = masker.fit_transform(str(bold_path))
    return np.asarray(arr, dtype="float32")


def process_subject(
    subject: str,
    fmriprep_root: Path,
    out_dir: Path,
    masker,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Returns (processed, skipped, errored) counts for this subject."""
    import numpy as np

    processed = 0
    skipped = 0
    errored = 0

    func_dir = fmriprep_root / subject / "func"
    if not func_dir.exists():
        err(f"[warn] {subject}: no func/ dir at {func_dir}")
        return (0, 0, 0)

    bold_files = sorted(func_dir.glob(BOLD_GLOB))
    if not bold_files:
        err(f"[warn] {subject}: no preprocessed BOLD files matched {BOLD_GLOB}")
        return (0, 0, 0)

    for bold_path in bold_files:
        meta = parse_run_metadata(bold_path.name)
        if meta is None:
            err(f"[warn] could not parse run metadata from {bold_path.name}")
            errored += 1
            continue
        file_subject, task, run = meta
        if file_subject != subject:
            err(f"[warn] subject mismatch: {bold_path.name} vs {subject}")
            errored += 1
            continue

        out_path = output_path(out_dir, subject, task, run)
        if out_path.exists():
            log(f"[skip] {subject} task-{task} run-{run} → already at {out_path.name}")
            skipped += 1
            continue

        if dry_run:
            log(f"[dry-run] would parcellate {bold_path.name} → {out_path.name}")
            processed += 1
            continue

        try:
            ts = parcellate_run(masker, bold_path)
        except Exception as exc:  # noqa: BLE001 — log + continue
            err(f"[fail] {subject} task-{task} run-{run}: {exc}")
            errored += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, ts)
        log(f"[parcellate] {subject} task-{task} run-{run} → {tuple(ts.shape)} saved")
        processed += 1

    return (processed, skipped, errored)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Schaefer-400 7-Networks parcellation of fMRIPrep BOLD output.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs (e.g. sub-01,sub-02). Defaults to all sub-XX dirs in derivatives.",
    )
    parser.add_argument(
        "--deriv-dir",
        type=str,
        default=str(DEFAULT_DERIV_DIR),
        help=f"Path to fMRIPrep derivatives root (default: {DEFAULT_DERIV_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Where to write parcellated .npy files (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be parcellated; do not fetch atlas or write files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    deriv_dir = Path(args.deriv_dir).resolve()
    fmriprep_root = deriv_dir / "fmriprep"
    out_dir = Path(args.out_dir).resolve()

    subjects = parse_subjects(args.subjects, fmriprep_root)

    log(f"[plan] derivatives: {deriv_dir}")
    log(f"[plan] fmriprep root: {fmriprep_root}")
    log(f"[plan] out dir: {out_dir}")
    log(f"[plan] subjects ({len(subjects)}): {', '.join(subjects)}")
    log(f"[plan] atlas: Schaefer-400 7-Networks (volumetric, 2mm)")
    log(f"[plan] TR={TR_SECONDS}s  high-pass={HIGH_PASS_HZ}Hz  low-pass={LOW_PASS_HZ}Hz")
    log(f"[plan] dry run: {args.dry_run}")

    masker = None
    if not args.dry_run:
        log("[atlas] fetching Schaefer-400 (cached under ~/nilearn_data/) …")
        atlas = fetch_atlas()
        log(f"[atlas] labels_img: {atlas.maps}")
        masker = build_masker(atlas.maps)

    total_processed = 0
    total_skipped = 0
    total_errored = 0

    for subject in subjects:
        p, s, e = process_subject(
            subject=subject,
            fmriprep_root=fmriprep_root,
            out_dir=out_dir,
            masker=masker,
            dry_run=args.dry_run,
        )
        total_processed += p
        total_skipped += s
        total_errored += e

    log("")
    log("[summary]")
    log(f"  processed={total_processed}  skipped={total_skipped}  errored={total_errored}")
    log(f"  out dir: {out_dir}")

    return 0 if total_errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
