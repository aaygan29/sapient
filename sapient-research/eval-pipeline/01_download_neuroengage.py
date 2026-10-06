"""Download NeuroEngage (OpenNeuro ds004996) raw BIDS data via DataLad.

Purpose
-------
Pull the raw BIDS dataset for the Sapient Cognitive Eval fine-tune (Track B).
Subjects come from `sapienteval/splits.json` (default: pilot subjects sub-01..sub-05).
For each subject, fetches anatomical (T1w), functional (BOLD), and fieldmap files
plus their JSON sidecars, into `sapienteval/data/ds004996/` (gitignored).

Dataset
-------
OpenNeuro ds004996 (NeuroEngage), license CC0.
Citation: Torubarova et al., "NeuroEngage", HRI 2025.
~30 GB total for all 50 subjects (anat + 3 BOLD runs + fmaps + sidecars).
Pilot (5 subjects) ≈ 3 GB.

Usage
-----
Install deps:
    pip install -e sapienteval   # or pip install datalad

Default (pilot subjects from splits.json):
    python sapienteval/01_download_neuroengage.py

Specific subjects:
    python sapienteval/01_download_neuroengage.py --subjects sub-01,sub-02,sub-03

Without fieldmaps (smaller, slightly less robust preprocessing):
    python sapienteval/01_download_neuroengage.py --no-include-fieldmaps

Dry run (lists what would be downloaded, no network):
    python sapienteval/01_download_neuroengage.py --dry-run

Idempotency
-----------
For each subject, if the subject's `anat/` and `func/` directories already contain
non-placeholder files matching the expected patterns, the script skips the
`datalad get` for that subject. Running the script a second time on a fully
populated subject is a no-op (just prints "skip: already present").

DataLad handles partial state gracefully: re-running `datalad get` on already-present
files is a no-op there too, so it's safe to interrupt and resume.

Notes
-----
- Requires `datalad` on the PATH. If missing, the script prints an install hint
  and exits non-zero.
- The dataset is cloned (not downloaded in full) on first invocation. Only the
  files matched by `--subjects` are then materialized by `datalad get`.
- File counts and approximate MB downloaded are reported per subject.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
SPLITS_PATH = REPO_ROOT / "splits.json"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "ds004996"
DATASET_URL = "https://github.com/OpenNeuroDatasets/ds004996.git"

# Per-subject patterns to fetch with `datalad get` (relative to subject dir).
# We pull T1w, BOLD, fieldmaps, and matching JSON sidecars.
ANAT_PATTERNS = ["anat/*_T1w.nii.gz", "anat/*_T1w.json"]
FUNC_PATTERNS = ["func/*_bold.nii.gz", "func/*_bold.json", "func/*_events.tsv"]
FMAP_PATTERNS = ["fmap/*.nii.gz", "fmap/*.json"]


def log(msg: str) -> None:
    print(msg, flush=True)


def err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def check_datalad() -> None:
    if shutil.which("datalad") is None:
        err(
            "ERROR: `datalad` not found on PATH.\n"
            "Install with: pip install datalad   (or: pip install -e sapienteval)\n"
            "datalad also requires `git-annex` — see https://handbook.datalad.org/en/latest/intro/installation.html"
        )
        sys.exit(2)


def load_default_subjects() -> list[str]:
    """Return the pilot subject list from splits.json."""
    if not SPLITS_PATH.exists():
        err(f"ERROR: splits.json not found at {SPLITS_PATH}")
        sys.exit(2)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    pilot = splits.get("pilot", {}).get("subjects")
    if not pilot:
        err("ERROR: splits.json has no `pilot.subjects` entry")
        sys.exit(2)
    return list(pilot)


def parse_subjects(arg: str | None) -> list[str]:
    if not arg:
        return load_default_subjects()
    subs = [s.strip() for s in arg.split(",") if s.strip()]
    for s in subs:
        if not s.startswith("sub-"):
            err(f"ERROR: subject ID must start with 'sub-': got {s!r}")
            sys.exit(2)
    return subs


def ensure_clone(data_dir: Path, dry_run: bool) -> None:
    """Clone the OpenNeuro dataset repo if not already present."""
    if (data_dir / ".datalad").exists() or (data_dir / ".git").exists():
        log(f"[clone] already present at {data_dir}")
        return
    if dry_run:
        log(f"[dry-run] would clone {DATASET_URL} -> {data_dir}")
        return
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"[clone] {DATASET_URL} -> {data_dir}")
    subprocess.run(
        ["datalad", "clone", DATASET_URL, str(data_dir)],
        check=True,
    )


def subject_already_present(subject_dir: Path) -> bool:
    """Idempotency check: do anat/ and func/ have non-placeholder files?

    DataLad placeholder files are typically symlinks to broken paths or tiny
    annex pointer files. A "present" file is one that resolves to a real,
    non-empty (>1 KB) blob.
    """
    anat = subject_dir / "anat"
    func = subject_dir / "func"
    if not (anat.exists() and func.exists()):
        return False
    have_anat = any(
        p.is_file() and p.stat().st_size > 1024
        for p in anat.glob("*_T1w.nii.gz")
    )
    have_func = any(
        p.is_file() and p.stat().st_size > 1024
        for p in func.glob("*_bold.nii.gz")
    )
    return have_anat and have_func


def list_target_patterns(subject: str, include_fieldmaps: bool) -> list[str]:
    """Return the glob patterns (relative to the dataset root) to fetch."""
    patterns: list[str] = []
    for p in ANAT_PATTERNS:
        patterns.append(f"{subject}/{p}")
    for p in FUNC_PATTERNS:
        patterns.append(f"{subject}/{p}")
    if include_fieldmaps:
        for p in FMAP_PATTERNS:
            patterns.append(f"{subject}/{p}")
    return patterns


def fetch_subject(
    data_dir: Path,
    subject: str,
    include_fieldmaps: bool,
    dry_run: bool,
) -> dict:
    """Run `datalad get` for one subject. Returns a small stats dict."""
    subject_dir = data_dir / subject
    if subject_already_present(subject_dir):
        log(f"[skip] {subject}: already present")
        return {"subject": subject, "status": "skip", "files": 0, "mb": 0.0}

    patterns = list_target_patterns(subject, include_fieldmaps)
    if dry_run:
        log(f"[dry-run] would `datalad get` for {subject}:")
        for p in patterns:
            log(f"  {p}")
        return {"subject": subject, "status": "dry-run", "files": len(patterns), "mb": 0.0}

    log(f"[get] {subject}: fetching {len(patterns)} pattern(s)")
    cmd = ["datalad", "get"] + patterns
    # Run from the dataset root so relative patterns resolve.
    proc = subprocess.run(cmd, cwd=str(data_dir))
    if proc.returncode != 0:
        err(f"[fail] {subject}: datalad get exited {proc.returncode}")
        return {"subject": subject, "status": "fail", "files": 0, "mb": 0.0}

    # Tally what actually landed.
    files = 0
    bytes_total = 0
    for sub_glob in ("anat", "func", "fmap"):
        sub_path = subject_dir / sub_glob
        if not sub_path.exists():
            continue
        for p in sub_path.rglob("*"):
            if p.is_file() and p.stat().st_size > 1024:
                files += 1
                bytes_total += p.stat().st_size
    mb = bytes_total / (1024 * 1024)
    log(f"[done] {subject}: {files} files, {mb:.1f} MB")
    return {"subject": subject, "status": "ok", "files": files, "mb": round(mb, 1)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download NeuroEngage (OpenNeuro ds004996) via DataLad.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs (e.g. sub-01,sub-02). Defaults to pilot from splits.json.",
    )
    parser.add_argument(
        "--include-fieldmaps",
        dest="include_fieldmaps",
        action="store_true",
        default=True,
        help="Include fieldmap files (default: true).",
    )
    parser.add_argument(
        "--no-include-fieldmaps",
        dest="include_fieldmaps",
        action="store_false",
        help="Skip fieldmap files.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Where to clone the dataset (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded; do nothing.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.dry_run:
        check_datalad()

    subjects = parse_subjects(args.subjects)
    data_dir = Path(args.data_dir).resolve()

    log(f"[plan] dataset: ds004996 (NeuroEngage, CC0)")
    log(f"[plan] data dir: {data_dir}")
    log(f"[plan] subjects ({len(subjects)}): {', '.join(subjects)}")
    log(f"[plan] include fieldmaps: {args.include_fieldmaps}")
    log(f"[plan] dry run: {args.dry_run}")

    ensure_clone(data_dir, dry_run=args.dry_run)

    stats: list[dict] = []
    for subject in subjects:
        stats.append(
            fetch_subject(
                data_dir=data_dir,
                subject=subject,
                include_fieldmaps=args.include_fieldmaps,
                dry_run=args.dry_run,
            )
        )

    log("")
    log("[summary]")
    ok = sum(1 for s in stats if s["status"] == "ok")
    skip = sum(1 for s in stats if s["status"] == "skip")
    fail = sum(1 for s in stats if s["status"] == "fail")
    dry = sum(1 for s in stats if s["status"] == "dry-run")
    total_files = sum(s["files"] for s in stats)
    total_mb = sum(s["mb"] for s in stats)
    log(f"  ok={ok}  skip={skip}  fail={fail}  dry-run={dry}")
    log(f"  total files materialized: {total_files}")
    log(f"  total downloaded: {total_mb:.1f} MB")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
