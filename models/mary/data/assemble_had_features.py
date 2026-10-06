"""Assemble HAD (ds004488) per-CLIP features into per-(subject,run) 2 Hz timelines.

This is the clip→fMRI-timeline alignment step. The frozen extractors write ONE
feature block per unique clip to
    /data/features/mary/had/_stories/{Category}__{clip}/{stream}.npy   (n_bins, D)
HAD is event-based, so a run is a 312 s timeline in which 60 clips appear at the
onsets recorded in the BIDS events.tsv. For each (subject, run, stream) we build a
zero (n_grid_bins=624 @ 2 Hz) array and drop each clip's block at its event onset:

    start_bin = round(onset_seconds * 2)
    grid[start_bin : start_bin + n_clip_bins] = clip_feat

Output (CONTRACTS §3):
    /data/features/mary/had/{subject}/{run-NN}/{stream}.npy   (624, D_m) float16

The 1 Hz fMRI for the same (subject, run-NN) lives at
    /data/fmri/mary/had/{subject}/{run-NN}.npy   (≈312, 20484)
and the manifest (built later) pairs 2 Hz features ↔ 1 Hz fMRI per the family's
T_2Hz=2×T_TR convention.

Alignment + key logic is documented in data/had_align.py (imported here).

Idempotent: skips an output that already exists unless --overwrite.

Usage:
  modal run data/assemble_had_features.py --limit 1                  # one run, audio streams
  modal run --detach data/assemble_had_features.py                   # sub-01+02, all runs/streams
  modal run data/assemble_had_features.py --subjects sub-01 --streams whisper,beats
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import modal

APP_NAME = "mary-assemble-had-features"

# --- HAD alignment constants + helpers (inlined from data/had_align.py so the
#     Modal container needs no sibling-module mount; see that file for the full
#     rationale of the 2 Hz onset placement). -------------------------------------
RAW_ROOT = "/data/raw/ds004488"
SES = "ses-action01"
TARGET_RATE_HZ = 2.0
N_TRS = 156
N_GRID_BINS = int(round(N_TRS * 2.0 * TARGET_RATE_HZ))  # 156 TRs × 2s × 2 Hz = 624
_RUN_RE = re.compile(r"run-(\d+)")


def _clip_key(stim_file: str) -> str:
    rel = stim_file.strip().strip("/")
    if rel.lower().endswith(".mp4"):
        rel = rel[:-4]
    return rel.replace("/", "__")


def _run_key(p: Path) -> str:
    m = _RUN_RE.search(p.name)
    return f"run-{int(m.group(1)):02d}" if m else "run-00"


def _list_event_files(subject: str, raw_root: str = RAW_ROOT) -> list[Path]:
    func = Path(raw_root) / subject / SES / "func"
    if not func.exists():
        return []
    return sorted(func.glob(f"{subject}_{SES}_task-action_run-*_events.tsv"))


def _read_events(ev_path: Path) -> list[dict]:
    rows: list[dict] = []
    with ev_path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            stim = (r.get("stim_file") or "").strip()
            if not stim or stim in ("n/a", "nan"):
                continue
            try:
                onset = float(r["onset"])
            except (KeyError, ValueError):
                continue
            rows.append({"onset": onset, "clip_key": _clip_key(stim),
                         "stim_file": stim})
    return rows


def _assemble_run_timeline(events, clip_feat_loader, hidden_dim,
                           n_grid_bins=N_GRID_BINS):
    import numpy as np
    grid = np.zeros((n_grid_bins, hidden_dim), dtype=np.float16)
    placed, missing, clipped = 0, [], 0
    for ev in events:
        feat = clip_feat_loader(ev["clip_key"])
        if feat is None or feat.ndim != 2 or feat.shape[1] != hidden_dim:
            missing.append(ev["clip_key"])
            continue
        start = max(0, int(round(ev["onset"] * TARGET_RATE_HZ)))
        end = start + feat.shape[0]
        if end > n_grid_bins:
            feat = feat[: n_grid_bins - start]
            end = n_grid_bins
            clipped += 1
        if end > start:
            grid[start:end] = feat.astype(np.float16)
            placed += 1
    stats = {"events": len(events), "placed": placed, "n_missing": len(missing),
             "missing": missing[:20], "clipped_at_end": clipped,
             "nonzero_bins": int((np.abs(grid).sum(axis=1) > 0).sum())}
    return grid, stats

# Streams to assemble. Audio (whisper, beats) + video (slowfast, qwen_vl) are the
# HAD-active streams; got_ocr is optional (on-screen text). qwen_ctx is absent
# (HAD has no spoken-language transcript) — handled by modality dropout.
ALL_STREAMS = ["slowfast", "qwen_vl", "whisper", "beats", "got_ocr"]
STREAM_DIM = {
    "slowfast": 2304, "qwen_vl": 3584, "whisper": 1280, "beats": 768,
    "got_ocr": 768, "qwen_ctx": 4096,
}

STORIES_ROOT = "/data/features/mary/had/_stories"
OUT_ROOT = "/data/features/mary/had"
DEFAULT_SUBJECTS = ["sub-01", "sub-02"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26,<3")
)
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(image=image, cpu=4.0, memory=16384, timeout=6 * 60 * 60,
              volumes={"/data": volume})
def assemble(subjects: list[str] | None = None,
             streams: list[str] | None = None,
             overwrite: bool = False,
             limit: int | None = None) -> dict:
    import numpy as np

    subjects = subjects or DEFAULT_SUBJECTS
    streams = streams or ALL_STREAMS
    stories = Path(STORIES_ROOT)
    out_root = Path(OUT_ROOT)

    # (subject, run-NN) tasks from events.tsv on the volume
    tasks: list[tuple[str, str, Path]] = []
    for sub in subjects:
        for ev in _list_event_files(sub):
            tasks.append((sub, _run_key(ev), ev))
    tasks.sort()
    if limit:
        tasks = tasks[:limit]
    print(f"[assemble] {len(tasks)} (subject,run) timelines × {len(streams)} streams")

    def make_loader(stream: str):
        dim = STREAM_DIM[stream]

        def loader(clip_key: str):
            p = stories / clip_key / f"{stream}.npy"
            if not p.exists():
                return None
            arr = np.load(p)
            return arr if arr.ndim == 2 and arr.shape[1] == dim else None
        return loader

    results = []
    for i, (sub, rk, ev) in enumerate(tasks):
        events = _read_events(ev)
        print(f"  [{i+1}/{len(tasks)}] {sub}/{rk}: {len(events)} events")
        for stream in streams:
            out = out_root / sub / rk / f"{stream}.npy"
            if out.exists() and not overwrite:
                results.append({"subject": sub, "run": rk, "stream": stream,
                                "status": "cached"})
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            grid, stats = _assemble_run_timeline(
                events, make_loader(stream), STREAM_DIM[stream])
            # If NOTHING for this stream was placed (e.g. video not extracted yet),
            # skip writing an all-zero file so the manifest can mark it missing.
            if stats["placed"] == 0:
                results.append({"subject": sub, "run": rk, "stream": stream,
                                "status": "no_features", **stats})
                print(f"      {stream}: NO features placed (likely not extracted) "
                      f"— skipping write")
                continue
            np.save(out, grid)
            volume.commit()
            results.append({"subject": sub, "run": rk, "stream": stream,
                            "status": "written", "shape": list(grid.shape),
                            **{k: stats[k] for k in
                               ("placed", "n_missing", "nonzero_bins", "clipped_at_end")}})
            print(f"      {stream}: {grid.shape} placed={stats['placed']} "
                  f"missing={stats['n_missing']} nonzero_bins={stats['nonzero_bins']}"
                  + (f"  !! missing e.g. {stats['missing'][:3]}" if stats["missing"] else ""))

    written = [r for r in results if r["status"] == "written"]
    summary = {
        "tasks": len(tasks), "streams": streams,
        "written": len(written),
        "cached": len([r for r in results if r["status"] == "cached"]),
        "no_features": len([r for r in results if r["status"] == "no_features"]),
        "results": results,
    }
    print(f"\nDONE: written={summary['written']} cached={summary['cached']} "
          f"no_features={summary['no_features']}")
    return summary


@app.local_entrypoint()
def main(subjects: str = "sub-01,sub-02", streams: str = "",
         overwrite: bool = False, limit: int = 0) -> None:
    subs = [s for s in subjects.split(",") if s] or None
    strs = [s for s in streams.split(",") if s] or None
    res = assemble.remote(subs, strs, overwrite, limit or None)
    print("\n========== SUMMARY ==========")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    for r in res["results"][:40]:
        print(" ", r.get("subject"), r.get("run"), r.get("stream"),
              r.get("status"), "placed=", r.get("placed"),
              "missing=", r.get("n_missing"))
