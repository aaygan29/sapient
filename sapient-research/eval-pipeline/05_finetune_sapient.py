"""Thin client wrapper around the `sapienteval-finetune` Modal app.

Purpose
-------
Local sanity loop for the fine-tune entrypoint. By default this script runs
in **dry-run mode** — it resolves the subject list (from `--subjects` or
`sapienteval/splits.json` `pilot.subjects`), prints the exact `modal run`
command, and exits without spending any GPU time.

Pass `--dry-run false` (or `--no-dry-run`) to actually submit the job to
Modal. The real training loop lives in `sapienteval/modal/finetune_sapient.py`.

Usage
-----
Dry-run with pilot subjects from splits.json (default):
    python sapienteval/05_finetune_sapient.py

Dry-run a specific subject subset:
    python sapienteval/05_finetune_sapient.py --subjects sub-01,sub-02

Actually submit the job to Modal:
    python sapienteval/05_finetune_sapient.py --no-dry-run

Override the run name:
    python sapienteval/05_finetune_sapient.py --run-name sapient_v0.5_pilot_smoke

What this does
--------------
Shells out to:
    modal run sapienteval/modal/finetune_sapient.py \
        --subjects sub-01,sub-02,... \
        --run-name sapient_v0.5_pilot

We use the Modal CLI rather than embedding the Modal SDK directly so the CLI
handles local-entrypoint orchestration, log streaming, and auth (workspace
`robert-16572`) without extra wiring. This mirrors `02_run_fmriprep_modal.py`.
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
MODAL_APP_PATH = REPO_ROOT / "sapienteval" / "modal" / "finetune_sapient.py"


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
        description=(
            "Submit (or dry-run) a fine-tune job to the sapienteval-finetune Modal app."
        ),
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
        "--run-name",
        default="sapient_v0.5_pilot",
        help="Output checkpoint stem (default: sapient_v0.5_pilot).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs (default: 5).",
    )
    # Dry-run defaults to True; opt out with --no-dry-run to actually launch.
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Print the `modal run` command without executing it (default).",
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually invoke `modal run` (spends GPU time).",
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
        "--run-name",
        args.run_name,
        "--epochs",
        str(args.epochs),
    ]

    print("Command:")
    print("  " + " ".join(cmd))
    print(f"Subjects ({len(subject_list)}): {subject_list}")
    print(f"Run name: {args.run_name}")
    print(f"Epochs:   {args.epochs}")

    if args.dry_run:
        print("[dry-run] Not invoking Modal. Pass --no-dry-run to actually launch.")
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
        print("  modal volume ls sapienteval-checkpoints")
        print(
            f"  modal volume get sapienteval-checkpoints /{args.run_name}.pt ./local-inspection/"
        )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
