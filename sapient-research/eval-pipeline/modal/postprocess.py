"""
sapienteval/modal/postprocess.py — Modal app: sapienteval-postprocess.

Two functions run AFTER fMRIPrep finishes:

  1. parcellate     — applies the Schaefer-400 7-Networks atlas to each
                       preprocessed BOLD run on /data/derivatives, writes
                       (T, 400) .npy arrays to /data/parcellated/.
  2. align_stimuli  — bins NeuroEngage events.tsv files into per-TR text
                       windows (with HRF-shift), writes JSON pairs to
                       /data/aligned/.

Mounted volume: `sapienteval-data` (read raw BIDS + derivatives, write
parcellated/aligned outputs).

Deploy:  modal deploy sapienteval/modal/postprocess.py
Run:     modal run sapienteval/modal/postprocess.py::parcellate_all
         modal run sapienteval/modal/postprocess.py::align_all
         modal run sapienteval/modal/postprocess.py     # runs both for pilot
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "nilearn==0.10.4",
        "nibabel",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "joblib",
    )
)

data_vol = modal.Volume.from_name("sapienteval-data", create_if_missing=True)

app = modal.App("sapienteval-postprocess", image=image)

TR_SECONDS = 2.0
LOW_PASS_HZ = 0.1
HIGH_PASS_HZ = 0.01
HRF_SHIFT_SECONDS = 4.0


# ─────────────────────────── Parcellation ────────────────────────────

@app.function(
    cpu=4,
    memory=8192,
    timeout=2 * 3600,
    volumes={"/data": data_vol},
)
def parcellate_subject(subject_id: str) -> dict:
    """Apply Schaefer-400 7-Networks parcellation to all preprocessed BOLD
    runs for one subject. Writes (T, 400) float32 .npy files."""
    import re
    import numpy as np
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.maskers import NiftiLabelsMasker

    label = subject_id.replace("sub-", "")
    # fmriprep 24.x writes derivatives directly to deriv_root (no /fmriprep/
    # subdir). The earlier path matched fmriprep <23 convention. Both paths
    # checked so this works if fmriprep is upgraded.
    candidates = [
        Path(f"/data/derivatives/sub-{label}/func"),
        Path(f"/data/derivatives/fmriprep/sub-{label}/func"),
    ]
    deriv_func = next((c for c in candidates if c.exists()), candidates[0])
    out_dir = Path("/data/parcellated")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not deriv_func.exists():
        return {"subject_id": subject_id, "status": "missing_fmriprep_output"}

    # Schaefer-400 7-Networks at 2mm (matches fMRIPrep --output-spaces ...:res-2)
    atlas = fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=7, resolution_mm=2,
    )

    bold_glob = "*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    bold_re = re.compile(
        r"^(?P<subject>sub-[A-Za-z0-9]+)"
        r"_task-(?P<task>[A-Za-z0-9]+)"
        r"(?:_run-(?P<run>[0-9]+))?"
        r"_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold\.nii\.gz$"
    )

    t0 = time.time()
    per_run = []
    for bold_path in sorted(deriv_func.glob(bold_glob)):
        m = bold_re.match(bold_path.name)
        if not m:
            continue
        task = m.group("task")
        run = m.group("run") or "1"
        out_name = f"{subject_id}_task-{task}_run-{run}.npy"
        out_path = out_dir / out_name
        if out_path.exists():
            per_run.append({"path": str(bold_path.name), "status": "skipped"})
            continue

        # No bandpass filter: it requires runs longer than ~33 TRs and the
        # Sapient model's data extractor applies its own preprocessing. We
        # keep z-score + detrend (cheap, no length requirement) for
        # numerically stable inputs to the encoder.
        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize="zscore_sample",
            detrend=True,
            t_r=TR_SECONDS,
        )
        ts = masker.fit_transform(str(bold_path))  # (T, 400)
        ts = ts.astype("float32")
        np.save(out_path, ts)
        per_run.append({"path": out_name, "shape": list(ts.shape), "status": "ok"})

    data_vol.commit()
    return {
        "subject_id": subject_id,
        "n_runs": sum(1 for r in per_run if r["status"] == "ok"),
        "n_skipped": sum(1 for r in per_run if r["status"] == "skipped"),
        "duration_s": round(time.time() - t0, 1),
        "runs": per_run,
    }


# ─────────────────────────── Alignment ────────────────────────────

def _load_annotation_transcripts(subject_id: str, run: str) -> tuple[list[tuple[float, str]], str | None]:
    """Look up the OsLjung annotation JSON for this (subject, run).

    Returns (transcript_events, source) where:
      - transcript_events is a sorted list of (timestamp_seconds, text) tuples
        merging participant + operator speech, tagged with speaker prefix
      - source is 'human-human', 'human-robot', or None if not found
    """
    label = subject_id.replace("sub-", "")
    run_padded = str(int(run)).zfill(2)
    # File naming: sub-XX_run-YY_<condition>_<engagement>.json
    # We search both conditions; the same (subject, run) may have files in
    # only one or both directories.
    candidates: list[tuple[Path, str]] = []
    for cond in ("human-human", "human-robot"):
        cond_dir = Path(f"/data/annotations/{cond}")
        if cond_dir.exists():
            for p in cond_dir.glob(f"sub-{label}_run-{run_padded}_*.json"):
                candidates.append((p, cond))

    if not candidates:
        return [], None

    # Prefer human-human if both exist (more natural conversation context)
    candidates.sort(key=lambda c: 0 if c[1] == "human-human" else 1)
    ann_path, source = candidates[0]

    try:
        data = json.loads(ann_path.read_text())
    except Exception:
        return [], None

    ann = data.get("annotation", {})
    out: list[tuple[float, str]] = []
    for t_str, tiers in ann.items():
        try:
            t_sec = float(t_str)
        except (TypeError, ValueError):
            continue
        if not isinstance(tiers, dict):
            continue
        pieces: list[str] = []
        for speaker in ("participant", "operator"):
            tier = tiers.get(speaker)
            if isinstance(tier, dict):
                text = (tier.get("value") or "").strip()
                if text:
                    pieces.append(f"[{speaker}] {text}")
        if pieces:
            out.append((t_sec, " ".join(pieces)))
    out.sort(key=lambda r: r[0])
    return out, source


@app.function(
    cpu=2,
    memory=4096,
    timeout=30 * 60,
    volumes={"/data": data_vol},
)
def align_subject(subject_id: str) -> dict:
    """Bin ds004996 events.tsv into per-TR text windows with HRF shift,
    merging real spoken transcripts from /data/annotations/ when available.

    For each TR, the text is the concatenation of:
      1. Speech (participant + operator) from /data/annotations/{cond}/<file>.json
         that falls inside the TR's effective stimulus window
      2. Event categories from events.tsv (fixation_cross, comprehension,
         silence, etc.) — kept as a fallback signal even when speech exists,
         since they encode the task structure

    Writes JSON files to /data/aligned/.
    """
    import pandas as pd

    label = subject_id.replace("sub-", "")
    raw_func = Path(f"/data/ds004996/sub-{label}/func")
    out_dir = Path("/data/aligned")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_func.exists():
        return {"subject_id": subject_id, "status": "missing_raw_bids"}

    EVENT_COLS = ["trial_type", "event_type"]

    t0 = time.time()
    per_run = []
    for events_path in sorted(raw_func.glob(f"sub-{label}_task-*_run-*_events.tsv")):
        # parse task + run from filename
        name = events_path.stem  # e.g. sub-01_task-conversation_run-01_events
        parts = name.split("_")
        task = next((p[5:] for p in parts if p.startswith("task-")), "task")
        run = next((p[4:] for p in parts if p.startswith("run-")), "1")
        out_name = f"{subject_id}_task-{task}_run-{run}.json"
        out_path = out_dir / out_name
        if out_path.exists():
            per_run.append({"path": out_name, "status": "skipped"})
            continue

        try:
            df = pd.read_csv(events_path, sep="\t")
        except Exception as e:
            per_run.append({"path": events_path.name, "status": "parse_error", "error": str(e)[:200]})
            continue

        # Event-type column (always there for ds004996; used as secondary signal)
        event_col = next((c for c in EVENT_COLS if c in df.columns), None)

        if "onset" not in df.columns:
            per_run.append({"path": events_path.name, "status": "no_onset_col"})
            continue

        # Pull real transcripts from the OsLjung annotation repo
        transcript_events, ann_source = _load_annotation_transcripts(subject_id, run)

        # Bin into TRs with HRF shift
        # TR t covers stimulus times [t*TR - HRF, (t+1)*TR - HRF)
        max_event_onset = float(df["onset"].max()) if not df.empty else 0.0
        max_transcript_onset = transcript_events[-1][0] if transcript_events else 0.0
        max_onset = max(max_event_onset, max_transcript_onset)
        n_trs = int((max_onset + HRF_SHIFT_SECONDS) // TR_SECONDS) + 1
        trs = []
        for t_idx in range(n_trs):
            t_start = t_idx * TR_SECONDS - HRF_SHIFT_SECONDS
            t_end = (t_idx + 1) * TR_SECONDS - HRF_SHIFT_SECONDS

            # Real transcripts in this window
            tr_speech: list[str] = []
            for ts, text in transcript_events:
                if t_start <= ts < t_end:
                    tr_speech.append(text)

            # Event categories in this window (kept as task-structure context)
            tr_events: list[str] = []
            if event_col is not None:
                mask = (df["onset"] >= t_start) & (df["onset"] < t_end)
                chunk = df[mask]
                if not chunk.empty:
                    tr_events = [
                        f"<{x}>" for x in chunk[event_col].fillna("").astype(str) if x
                    ]

            tr_text = " ".join(tr_speech + tr_events)
            trs.append({"tr_index": t_idx, "text": tr_text})

        out_obj = {
            "subject_id": subject_id,
            "task": task,
            "run": int(run),
            "n_TRs": n_trs,
            "tr_seconds": TR_SECONDS,
            "hrf_shift_seconds": HRF_SHIFT_SECONDS,
            "annotation_source": ann_source,
            "event_column_used": event_col,
            "n_transcript_events": len(transcript_events),
            "trs": trs,
        }
        out_path.write_text(json.dumps(out_obj))
        per_run.append({
            "path": out_name,
            "n_TRs": n_trs,
            "annotation_source": ann_source,
            "n_transcript_events": len(transcript_events),
            "status": "ok",
        })

    data_vol.commit()
    return {
        "subject_id": subject_id,
        "n_runs": sum(1 for r in per_run if r["status"] == "ok"),
        "duration_s": round(time.time() - t0, 1),
        "runs": per_run,
    }


# ─────────────────────────── Orchestration ────────────────────────────

@app.local_entrypoint()
def main(subjects: str = "", step: str = "all"):
    """Run parcellation and/or alignment for the given subjects.

    Defaults to pilot subjects from sapienteval/splits.json. Step:
      - "parcellate"  → parcellation only
      - "align"       → alignment only
      - "all"         → both (default)
    """
    if subjects:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
    else:
        splits_path = Path("sapienteval/splits.json")
        if splits_path.exists():
            subject_list = json.loads(splits_path.read_text())["pilot"]["subjects"]
        else:
            subject_list = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]

    print(f"Postprocess subjects: {subject_list}  (step={step})")

    if step in ("parcellate", "all"):
        print("\n=== Parcellation (Schaefer-400) ===")
        results = list(parcellate_subject.map(subject_list))
        print(json.dumps(results, indent=2))

    if step in ("align", "all"):
        print("\n=== Stimulus alignment ===")
        results = list(align_subject.map(subject_list))
        print(json.dumps(results, indent=2))
