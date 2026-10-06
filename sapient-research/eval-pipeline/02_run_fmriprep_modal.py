"""Thin client wrapper around the `sapienteval-fmriprep` Modal app.

Purpose
-------
Submit fMRIPrep jobs to Modal for a chosen set of NeuroEngage (ds004996)
subjects. Defaults to the pilot subjects defined in `sapienteval/splits.json`
(`pilot.subjects`). All real preprocessing happens inside the Modal app
defined in `sapienteval/modal/fmriprep_runner.py`.

Usage
-----
Pilot (5 subjects from splits.json):
    python sapienteval/02_run_fmriprep_modal.py

Specific subjects (overrides splits.json default):
    python sapienteval/02_run_fmriprep_modal.py --subjects sub-01,sub-02

Dry run (prints the `modal run ...` command without executing it):
    python sapienteval/02_run_fmriprep_modal.py --dry-run

What this does
--------------
Shells out to:
    modal run sapienteval/modal/fmriprep_runner.py --subjects sub-01,sub-02,...

We use the Modal CLI instead of embedding the Modal SDK directly here because
Modal's CLI handles the local-entrypoint orchestration, log streaming, and
auth (workspace `robert-16572`) without extra wiring.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = REPO_ROOT / "sapienteval" / "splits.json"
MODAL_APP_PATH = REPO_ROOT / "sapienteval" / "modal" / "fmriprep_runner.py"


def load_pilot_subjects() -> list[str]:
    """Read `pilot.subjects` from sapienteval/splits.json."""
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"splits.json not found at {SPLITS_PATH}; cannot resolve default subject list"
        )
    with SPLITS_PATH.open() as f:
        splits = json.load(f)
    pilot = splits.get("pilot", {})
    subjects = pilot.get("subjects")
    if not subjects:
        raise ValueError("splits.json has no `pilot.subjects` array")
    return list(subjects)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit fMRIPrep jobs to the sapienteval-fmriprep Modal app.",
    )
    p.add_argument(
        "--subjects",
        default=None,
        help=(
            "Comma-separated subject IDs (e.g. 'sub-01,sub-02'). "
            "Defaults to pilot subjects from sapienteval/splits.json."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the `modal run` command without executing it.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.subjects:
        subject_list = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subject_list = load_pilot_subjects()
        print(f"Using pilot subjects from splits.json: {subject_list}")

    if not subject_list:
        print("No subjects to submit.", file=sys.stderr)
        return 2

    subjects_csv = ",".join(subject_list)
    cmd = [
        "modal",
        "run",
        str(MODAL_APP_PATH),
        "--subjects",
        subjects_csv,
    ]

    print("Command:")
    print("  " + " ".join(cmd))
    print(f"Subjects ({len(subject_list)}): {subject_list}")

    if args.dry_run:
        print("[dry-run] Not invoking Modal.")
        return 0

    if shutil.which("modal") is None:
        print(
            "ERROR: `modal` CLI not found on PATH. Install with `pip install modal` "
            "and run `modal token new` (workspace: robert-16572).",
            file=sys.stderr,
        )
        return 127

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"Modal run failed with returncode={proc.returncode}", file=sys.stderr)
    else:
        print("Modal run finished. Inspect outputs with:")
        print("  modal volume ls sapienteval-data")
        print(
            "  modal volume get sapienteval-data /derivatives/fmriprep/sub-01/func/ ./local-inspection/"
        )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
