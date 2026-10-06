"""
04_align_stimuli.py — Align ds004996 transcripts to BOLD TRs.

For each run, reads the events.tsv from sub-XX/func/*_events.tsv (BIDS standard)
which has columns: onset (seconds), duration (seconds), trial_type, response_time?, …

Bins events into TR-aligned windows (TR = 2.0s). For each TR t (covering seconds
[t * 2.0, (t + 1) * 2.0)), collects the concatenated event-related transcript
text for that window (accounting for hemodynamic response function delay of
~4-6s — we shift event windows 4s backward when binning, so TR t corresponds to
stimuli that played at t * 2.0 - 4 ≤ time < t * 2.0 - 2).

Outputs: sapienteval/data/aligned/sub-XX_task-{task}_run-{run}.json
         {
           "subject_id": "sub-01",
           "task": "engage",
           "run": 1,
           "n_TRs": 245,
           "tr_seconds": 2.0,
           "hrf_shift_seconds": 4.0,
           "trs": [
             {"tr_index": 0, "text": "[turn] hello how are you", "events": [...]},
             ...
           ]
         }

Skips audio alignment for v0.5 (text-only path; see sapienteval/SCIENCE.md §6).

Idempotent. Usage same as 03_parcellate.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "ds004996"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "aligned"

# Default BIDS-ish events.tsv pattern. ds004996 stores per-run events under
# sub-XX/func/, e.g. sub-01_task-engage_run-1_events.tsv.
EVENTS_GLOB = "*_events.tsv"
EVENTS_RE = re.compile(
    r"^(?P<subject>sub-[A-Za-z0-9]+)"
    r"_task-(?P<task>[A-Za-z0-9]+)"
    r"(?:_run-(?P<run>[0-9]+))?"
    r"_events\.tsv$"
)

# TODO: verify against actual ds004996 events.tsv schema once downloaded.
# Observed columns in NeuroEngage aggregate script: onset, duration, event_type.
# The annotation repo (OsLjung/NeuroEngage-annotation-data) holds the actual
# transcripts ("participant"/"operator" text). For the v0.5 pilot we degrade
# gracefully to event_type/trial_type tokens when no transcript column exists.
TEXT_COLUMN_CANDIDATES = ("transcript", "text", "stim_text", "utterance")
TYPE_COLUMN_CANDIDATES = ("event_type", "trial_type", "type")
SPEAKER_COLUMN_CANDIDATES = ("speaker", "role", "actor")


def log(msg: str) -> None:
    print(msg, flush=True)


def err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def discover_subjects(data_dir: Path) -> list[str]:
    """Return all sub-XX directories in the raw BIDS data dir."""
    if not data_dir.exists():
        return []
    return sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )


def parse_subjects(arg: str | None, data_dir: Path) -> list[str]:
    if not arg:
        subs = discover_subjects(data_dir)
        if not subs:
            err(f"ERROR: no sub-XX directories found under {data_dir}")
            sys.exit(2)
        return subs
    subs = [s.strip() for s in arg.split(",") if s.strip()]
    for s in subs:
        if not s.startswith("sub-"):
            err(f"ERROR: subject ID must start with 'sub-': got {s!r}")
            sys.exit(2)
    return subs


def parse_run_metadata(filename: str) -> tuple[str, str, str] | None:
    """Return (subject, task, run) from an events.tsv filename, or None."""
    m = EVENTS_RE.match(filename)
    if not m:
        return None
    return (
        m.group("subject"),
        m.group("task"),
        m.group("run") or "1",
    )


def output_path(out_dir: Path, subject: str, task: str, run: str) -> Path:
    return out_dir / f"{subject}_task-{task}_run-{run}.json"


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first column from `candidates` that exists (case-insensitive)."""
    lower_to_actual = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_actual:
            return lower_to_actual[cand.lower()]
    return None


def event_text(row: dict, text_col: str | None, type_col: str | None, speaker_col: str | None) -> str:
    """Build a single-event text token. Prefers transcript text; falls back to type tag."""
    speaker = ""
    if speaker_col:
        spk = str(row.get(speaker_col, "") or "").strip()
        if spk:
            speaker = f"[{spk}] "

    if text_col:
        txt = str(row.get(text_col, "") or "").strip()
        if txt:
            return f"{speaker}{txt}"

    if type_col:
        tag = str(row.get(type_col, "") or "").strip()
        if tag:
            return f"{speaker}[{tag}]"

    return ""


def estimate_n_trs(events_df, tr_seconds: float) -> int:
    """Estimate the number of TRs covered by the events.

    We don't have the BOLD .nii.gz length here without loading it, so we use
    the latest event end-time (onset + duration) rounded up to the next TR.
    This is an upper bound; alignment with parcellated arrays happens during
    training where we trim to min(n_trs_events, n_trs_bold).
    """
    if events_df is None or len(events_df) == 0:
        return 0
    onsets = events_df["onset"].astype(float)
    durations = events_df["duration"].astype(float).fillna(0.0)
    end = float((onsets + durations).max())
    return int(end // tr_seconds) + 1


def bin_events_to_trs(
    events_df,
    n_trs: int,
    tr_seconds: float,
    hrf_shift: float,
    text_col: str | None,
    type_col: str | None,
    speaker_col: str | None,
) -> list[dict]:
    """Return a list of {tr_index, text, events} dicts, length n_trs.

    Binning rule (HRF-shifted):
      TR t covers stimuli with onset in
        [t * tr_seconds - hrf_shift, (t + 1) * tr_seconds - hrf_shift)
    """
    trs: list[dict] = [
        {"tr_index": t, "text": "", "events": []} for t in range(n_trs)
    ]
    if events_df is None or len(events_df) == 0:
        return trs

    for _, row in events_df.iterrows():
        try:
            onset = float(row["onset"])
        except Exception:
            continue
        # Solve for t: t * tr - hrf <= onset < (t+1) * tr - hrf
        t = int((onset + hrf_shift) // tr_seconds)
        if t < 0 or t >= n_trs:
            continue
        token = event_text(row.to_dict(), text_col, type_col, speaker_col)
        ev_record = {
            "onset": round(onset, 3),
            "duration": round(float(row.get("duration", 0.0) or 0.0), 3),
        }
        if type_col and row.get(type_col) is not None:
            ev_record["type"] = str(row.get(type_col))
        if speaker_col and row.get(speaker_col) is not None:
            ev_record["speaker"] = str(row.get(speaker_col))
        if text_col and row.get(text_col) is not None:
            txt_raw = str(row.get(text_col) or "").strip()
            if txt_raw:
                ev_record["text"] = txt_raw
        trs[t]["events"].append(ev_record)
        if token:
            trs[t]["text"] = (trs[t]["text"] + " " + token).strip() if trs[t]["text"] else token

    return trs


def process_events_file(
    events_path: Path,
    out_dir: Path,
    tr_seconds: float,
    hrf_shift: float,
    dry_run: bool,
) -> tuple[bool, bool, bool]:
    """Process one events.tsv. Returns (processed, skipped, errored) as bools."""
    import pandas as pd  # local import keeps --dry-run fast

    meta = parse_run_metadata(events_path.name)
    if meta is None:
        err(f"[warn] could not parse run metadata from {events_path.name}")
        return (False, False, True)
    subject, task, run = meta

    out_path = output_path(out_dir, subject, task, run)
    if out_path.exists():
        log(f"[skip] {subject} task-{task} run-{run} → already at {out_path.name}")
        return (False, True, False)

    if dry_run:
        log(f"[dry-run] would align {events_path.name} → {out_path.name}")
        return (True, False, False)

    try:
        events_df = pd.read_csv(events_path, sep="\t")
    except Exception as exc:  # noqa: BLE001
        err(f"[fail] {subject} task-{task} run-{run}: read events.tsv: {exc}")
        return (False, False, True)

    if "onset" not in events_df.columns:
        err(f"[fail] {subject} task-{task} run-{run}: events.tsv missing required 'onset' column "
            f"(got: {list(events_df.columns)})")
        return (False, False, True)
    if "duration" not in events_df.columns:
        events_df["duration"] = 0.0

    text_col = pick_column(events_df.columns, TEXT_COLUMN_CANDIDATES)
    type_col = pick_column(events_df.columns, TYPE_COLUMN_CANDIDATES)
    speaker_col = pick_column(events_df.columns, SPEAKER_COLUMN_CANDIDATES)

    n_trs = estimate_n_trs(events_df, tr_seconds)
    trs = bin_events_to_trs(
        events_df=events_df,
        n_trs=n_trs,
        tr_seconds=tr_seconds,
        hrf_shift=hrf_shift,
        text_col=text_col,
        type_col=type_col,
        speaker_col=speaker_col,
    )

    payload = {
        "subject_id": subject,
        "task": task,
        "run": int(run),
        "n_TRs": n_trs,
        "tr_seconds": tr_seconds,
        "hrf_shift_seconds": hrf_shift,
        "text_column": text_col,
        "type_column": type_col,
        "speaker_column": speaker_col,
        "source_events_tsv": str(events_path),
        "trs": trs,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    n_nonempty = sum(1 for tr in trs if tr["text"])
    log(
        f"[align] {subject} task-{task} run-{run} → {n_trs} TRs "
        f"({n_nonempty} with text) saved"
    )
    return (True, False, False)


def process_subject(
    subject: str,
    data_dir: Path,
    out_dir: Path,
    tr_seconds: float,
    hrf_shift: float,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Returns (processed, skipped, errored) counts for this subject."""
    func_dir = data_dir / subject / "func"
    if not func_dir.exists():
        err(f"[warn] {subject}: no func/ dir at {func_dir}")
        return (0, 0, 0)

    events_files = sorted(func_dir.glob(EVENTS_GLOB))
    if not events_files:
        err(f"[warn] {subject}: no events.tsv files matched {EVENTS_GLOB}")
        return (0, 0, 0)

    processed = 0
    skipped = 0
    errored = 0
    for events_path in events_files:
        p, s, e = process_events_file(
            events_path=events_path,
            out_dir=out_dir,
            tr_seconds=tr_seconds,
            hrf_shift=hrf_shift,
            dry_run=dry_run,
        )
        processed += int(p)
        skipped += int(s)
        errored += int(e)
    return (processed, skipped, errored)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Align ds004996 transcripts to BOLD TRs (text-only path).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs (e.g. sub-01,sub-02). Defaults to all sub-XX dirs in data dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to raw BIDS data root (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Where to write aligned .json files (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--tr-seconds",
        type=float,
        default=2.0,
        help="BOLD TR in seconds (default: 2.0 for ds004996).",
    )
    parser.add_argument(
        "--hrf-shift",
        type=float,
        default=4.0,
        help="HRF shift in seconds applied to event onsets when binning (default: 4.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be aligned; do not read events.tsv or write files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    subjects = parse_subjects(args.subjects, data_dir)

    log(f"[plan] data dir: {data_dir}")
    log(f"[plan] out dir: {out_dir}")
    log(f"[plan] subjects ({len(subjects)}): {', '.join(subjects)}")
    log(f"[plan] TR={args.tr_seconds}s  HRF shift={args.hrf_shift}s")
    log(f"[plan] text-only path (audio alignment skipped — see SCIENCE.md §6)")
    log(f"[plan] dry run: {args.dry_run}")

    total_processed = 0
    total_skipped = 0
    total_errored = 0
    for subject in subjects:
        p, s, e = process_subject(
            subject=subject,
            data_dir=data_dir,
            out_dir=out_dir,
            tr_seconds=args.tr_seconds,
            hrf_shift=args.hrf_shift,
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
