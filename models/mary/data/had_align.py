"""HAD (ds004488) clip <-> fMRI event-timeline alignment helpers.

HAD is EVENT-BASED, not a continuous movie. Each run is a 312 s timeline in which
60 distinct 2 s clips are shown (2 s clip + 2 s ISI; a blank trial after every 5
trials, 4 blanks at the start/end of each run). The mapping from a clip file to
its onset within a run lives in the BIDS events.tsv:

    onset   duration  trial_type  response  response_time  stim_file
    12.122  2.010     46.0        -1.0      1.377          Drum_corps/v_..._label_1.mp4
    16.035  2.014     104.0       1.0       2.442          Playing_kickball/v_..._label_1.mp4
    ...

Event files (raw BIDS, fetched onto the volume) live at:
    /data/raw/ds004488/{sub}/ses-action01/func/{sub}_ses-action01_task-action_run-NN_events.tsv

`stim_file` is the clip path relative to /data/raw/ds004488/stimuli, e.g.
`Drum_corps/v_Drum_corps_id_0Qo8FzNJ460_start_28.0_label_1.mp4`.

----------------------------------------------------------------------------------
CLIP KEY (must match every extractor)
----------------------------------------------------------------------------------
Per-clip cached features are written under `_stories/{clip_key}/{stream}.npy`.
`clip_key` = the clip's path relative to the stimuli root, no extension, slashes
→ `__` — i.e. the SAME convention as the video extractors' `_rel_story()`:
    Drum_corps/v_Drum_corps_id_0Qo8FzNJ460_start_28.0_label_1.mp4
      -> "Drum_corps__v_Drum_corps_id_0Qo8FzNJ460_start_28.0_label_1"

----------------------------------------------------------------------------------
2 Hz ONSET ALIGNMENT (the core of the task)
----------------------------------------------------------------------------------
Feature grid: 2 Hz (CONTRACTS §2). A run is 156 TRs × TR(2 s) = 312 s → 624 bins.
For each event with onset `t0` (s) and the clip's per-clip feature block
`F` of shape (n_clip_bins, D):
    start_bin = round(t0 * 2.0)                         # onset → 2 Hz bin index
    grid[start_bin : start_bin + n_clip_bins] = F       # drop the clip in place
All other bins (ISI, blank/fixation, run pad) stay ZERO — i.e. "no stimulus".
A clip is ~2 s → ~4 feature bins; placed back-to-back-with-gaps exactly where the
subject saw it. This yields per-(subject,run) stream timelines on the same 2 Hz
grid the fMRI (resampled to 1 Hz, 312 frames) pairs with via the manifest.

Rounding rationale: onsets are ~4 s apart and clips are ~2 s, so neighbouring
clips never collide after rounding to 0.5 s (2 Hz). We clip the write window to
the grid length so a clip starting near the run end can't overflow.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Layout constants (verified against the volume + the HAD paper / events.tsv).
RAW_ROOT = "/data/raw/ds004488"
STIM_ROOT = f"{RAW_ROOT}/stimuli"
SES = "ses-action01"
TARGET_RATE_HZ = 2.0
TR_SECONDS = 2.0
N_TRS = 156                       # per run (sidecar/bold-confirmed)
RUN_SECONDS = N_TRS * TR_SECONDS  # 312 s
N_GRID_BINS = int(round(RUN_SECONDS * TARGET_RATE_HZ))  # 624 bins @ 2 Hz

_RUN_RE = re.compile(r"run-(\d+)")


def clip_key_from_stim_file(stim_file: str) -> str:
    """`Cat/clip.mp4` -> `Cat__clip` (matches extractors' `_rel_story`)."""
    rel = stim_file.strip().strip("/")
    if rel.lower().endswith(".mp4"):
        rel = rel[:-4]
    return rel.replace("/", "__")


def clip_key_from_path(mp4: Path, stim_root: Path) -> str:
    """Same key, from an absolute clip path under the stimuli root."""
    rel = mp4.relative_to(stim_root).with_suffix("")
    return str(rel).replace("/", "__")


def run_key_from_events_path(p: Path) -> str:
    m = _RUN_RE.search(p.name)
    return f"run-{int(m.group(1)):02d}" if m else "run-00"


def events_path(subject: str, run_key: str, raw_root: str = RAW_ROOT) -> Path:
    """Canonical events.tsv path for a (subject, run-NN)."""
    n = int(run_key.split("-")[1])
    return (Path(raw_root) / subject / SES / "func" /
            f"{subject}_{SES}_task-action_run-{n:02d}_events.tsv")


def list_event_files(subject: str, raw_root: str = RAW_ROOT) -> list[Path]:
    func = Path(raw_root) / subject / SES / "func"
    if not func.exists():
        return []
    return sorted(func.glob(f"{subject}_{SES}_task-action_run-*_events.tsv"))


def read_events(ev_path: Path) -> list[dict]:
    """Parse one events.tsv → [{onset, duration, trial_type, stim_file, clip_key}]."""
    rows: list[dict] = []
    with ev_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            stim = (r.get("stim_file") or "").strip()
            if not stim or stim in ("n/a", "nan"):
                continue
            try:
                onset = float(r["onset"])
            except (KeyError, ValueError):
                continue
            dur = r.get("duration")
            rows.append({
                "onset": onset,
                "duration": float(dur) if dur not in (None, "", "n/a") else None,
                "trial_type": (r.get("trial_type") or "").strip(),
                "stim_file": stim,
                "clip_key": clip_key_from_stim_file(stim),
            })
    return rows


def referenced_clips(subjects: list[str], raw_root: str = RAW_ROOT) -> dict[str, str]:
    """All unique clips shown to `subjects` → {clip_key: stim_file_relpath}.

    Use this so extractors only featurize the ~720/subject clips actually seen,
    not all 21,600 in the stimuli tree.
    """
    out: dict[str, str] = {}
    for sub in subjects:
        for ev in list_event_files(sub, raw_root):
            for r in read_events(ev):
                out.setdefault(r["clip_key"], r["stim_file"])
    return out


def assemble_run_timeline(
    events: list[dict],
    clip_feat_loader,            # clip_key:str -> np.ndarray (n_bins, D) | None
    hidden_dim: int,
    n_grid_bins: int = N_GRID_BINS,
):
    """Build a (n_grid_bins, hidden_dim) 2 Hz stream timeline for one run.

    Each event's per-clip feature block is dropped at `round(onset*2)`; the rest
    (ISI / blanks / pad) stays zero. Returns (grid, stats) where stats records
    placement counts + any missing clips. `import numpy as np` is done lazily so
    this module stays importable on the 8 GB laptop without torch/numpy heavy use.
    """
    import numpy as np

    grid = np.zeros((n_grid_bins, hidden_dim), dtype=np.float16)
    placed, missing, clipped = 0, [], 0
    for ev in events:
        feat = clip_feat_loader(ev["clip_key"])
        if feat is None:
            missing.append(ev["clip_key"])
            continue
        if feat.ndim != 2 or feat.shape[1] != hidden_dim:
            missing.append(f"{ev['clip_key']}(shape={getattr(feat,'shape',None)})")
            continue
        start = int(round(ev["onset"] * TARGET_RATE_HZ))
        if start < 0:
            start = 0
        end = start + feat.shape[0]
        if end > n_grid_bins:                      # clip near run end → truncate
            feat = feat[: n_grid_bins - start]
            end = n_grid_bins
            clipped += 1
        if end > start:
            grid[start:end] = feat.astype(np.float16)
            placed += 1
    stats = {
        "events": len(events),
        "placed": placed,
        "n_missing": len(missing),
        "missing": missing[:20],
        "clipped_at_end": clipped,
        "nonzero_bins": int((np.abs(grid).sum(axis=1) > 0).sum()),
        "grid_bins": n_grid_bins,
    }
    return grid, stats
