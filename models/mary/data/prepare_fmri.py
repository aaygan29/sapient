"""Project ds002345 ("Narratives") raw BOLD onto fsaverage5 → 1 Hz, z-scored.

CONTRACTS.md §0 (WS-D routing) + §1 (20,484 fsaverage5) + §3 (cache paths).
Mirrors sapient1/data/prepare_fmri.py (vol→fsaverage5 via nilearn vol_to_surf,
resample to 1 Hz, z-score per vertex per run) but for the Huth Narratives
dataset, whose BOLD is BIDS *raw* (native EPI space, no `space-` tag) with
TR read from each run's JSON sidecar (BIDS `RepetitionTime`, ≈1.5 s here).

For each `sub-*/func/*_bold.nii.gz`:
  1. Load with nibabel.
  2. Project to fsaverage5 (10,242/hemi → 20,484 total) via vol_to_surf.
     Native-space EPI projects through its own affine; nilearn resamples the
     fsaverage5 pial mesh into the volume's world coordinates.
  3. Resample TR-grid → 1 Hz along time (linear interp, like the family).
  4. Z-score per vertex per run (mean 0 / std 1 over time).
  5. Save `(T_1Hz, 20484)` float32 →
     /data/fmri/mary/huth/{subject}/{story}.npy

The story name is the BIDS `task-` label, normalized so it matches a
`stimuli/<story>_audio.wav` stem (so the feature agent's
`_stories/{story}/{stream}.npy` line up). Multi-run stories are concatenated in
run order into a single per-(subject,story) array (one stimulus → one story).

Usage (Modal):
  modal run data/prepare_fmri.py                       # default slice (8 subjects)
  modal run data/prepare_fmri.py --subjects sub-001 --stories pieman   # one
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FSAVERAGE_MESH = "fsaverage5"
N_VERTICES = 20484                     # fsaverage5: 10,242/hemi × 2 (LH⊕RH)
DATASET = "huth"                       # CONTRACTS §3 dataset slug for ds002345
RAW_ROOT_DEFAULT = "/data/raw/ds002345"
OUT_ROOT_DEFAULT = "/data/fmri/mary/huth"
DEFAULT_TR = 1.5                       # fallback if a sidecar is missing

TASK_RE = re.compile(r"task-([A-Za-z0-9]+)")
RUN_RE = re.compile(r"run-(\d+)")
SES_RE = re.compile(r"ses-(\d+)")

# ---------------------------------------------------------------------------
# CNeuroMod / Algonauts-2025 dataset config (movie10 + friends).
#   Layout: fmriprep/{movie10,friends}/sub-XX/ses-XXX/func/
#           sub-XX_ses-XXX_task-<label>_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
#   The story key matches the feature-extractor key (CONTRACTS §3):
#     task-bourne05 (under movie10/) -> "movie10_bourne05"
#     task-s01e24a  (under friends/)  -> "friends_s01e24a"
#   so /data/fmri/mary/cneuromod/<subject>/<story>.npy pairs with
#      /data/features/mary/cneuromod/_stories/<story>/<stream>.npy.
#   CNeuroMod movie TR = 1.49 s. Each task usually has a single run; if multiple
#   (run-1, run-2 across sessions) they are concatenated in (session, run) order.
CNEUROMOD_BASE = "/data/raw/cneuromod/fmriprep"
CNEUROMOD_OUT = "/data/fmri/mary/cneuromod"
CNEUROMOD_GROUPS = ("movie10", "friends")
CNEUROMOD_TR = 1.49
CNEUROMOD_SPACE = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"


# ---------------------------------------------------------------------------
# Lebel2023 / ds003020 ("Lebel/Huth Narratives", raw BIDS, SESSIONS).
#   Layout: /data/raw/ds003020/sub-UTSXX/ses-NN/func/
#           sub-UTSXX_ses-NN_task-<story>[_run-N]_bold.nii.gz
#   BOLD is *raw* (native EPI, no `space-` tag) like ds002345 → TR from the JSON
#   sidecar (RepetitionTime ≈ 2.0 s). The story key is the raw BIDS `task-<label>`
#   — it already matches the feature dirs at
#       /data/features/mary/lebel2023/_stories/<story>/<stream>.npy
#   (verified: 84/84 story tasks overlap; only the 5 *Localizer tasks have no
#   features and are naturally dropped by the manifest). So NO normalization is
#   applied here — we keep the raw task label as the story key. A story may be
#   repeated across sessions/runs; those are concatenated in (session, run) order
#   into one per-(subject,story) array (one stimulus → one story, family conv.).
LEBEL_BASE = "/data/raw/ds003020"
LEBEL_OUT = "/data/fmri/mary/lebel2023"
LEBEL_TR = 2.0  # fallback; real TR read from each run's sidecar


# ---------------------------------------------------------------------------
# Wen2017 (Purdue PURR 2809) — Subject-1 video-fMRI, MNI volumetric.
#   Layout (after unzip): /data/raw/wen2017/video_fmri_dataset/subject1/fmri/
#       {seg1..seg18, test1..test5}/mni/*_mni.nii.gz   (4D MNI152 volumetric)
#   The run key is the segment dir name (seg1..seg18, test1..test5), which
#   matches the stimulus stem (stimuli/seg1.mp4 ...) so the video features at
#   /data/features/mary/wen2017/_stories/{seg1..test5}/{stream}.npy line up.
#   Video-only & silent → only the 3 video streams exist; audio/text streams are
#   recorded as missing per CONTRACTS §4 (modality dropout handles them).
#   TR = 2.0 s. `test*` segments were acquired with 10 repeats; the published
#   *_mni.nii.gz under test*/mni is the repeat-averaged 4D series (one per test).
WEN_BASE = "/data/raw/wen2017/video_fmri_dataset/subject1/fmri"
WEN_OUT = "/data/fmri/mary/wen2017"
WEN_TR = 2.0
WEN_SUBJECT = "subject1"


# ---------------------------------------------------------------------------
# Story name normalization: BIDS task label -> stimulus wav stem
# ---------------------------------------------------------------------------
def normalize_story(task: str, wav_stems: set[str]) -> str:
    """Map a BIDS `task-<label>` to the matching `<story>_audio.wav` stem.

    Exact match wins. Otherwise fall back to a known alias (milkyway has
    multiple variants on disk) or a unique prefix match; else return the raw
    task (the manifest will then record no shared feature dir for it).
    """
    if task in wav_stems:
        return task
    aliases = {"milkyway": "milkywayoriginal"}
    if task in aliases and aliases[task] in wav_stems:
        return aliases[task]
    # Unique prefix (e.g. notthefallintact already matches exactly; this is a
    # guard for any task whose wav stem is task+suffix).
    pref = [s for s in wav_stems if s == task or s.startswith(task)]
    if len(pref) == 1:
        return pref[0]
    return task


# ---------------------------------------------------------------------------
# Core (pure) functions — usable locally or on Modal
# ---------------------------------------------------------------------------
def read_tr(bold_path: Path) -> float:
    """Read RepetitionTime from the run's JSON sidecar; fall back to DEFAULT_TR."""
    side = bold_path.with_name(bold_path.name.replace("_bold.nii.gz", "_bold.json"))
    if side.exists():
        try:
            tr = float(json.loads(side.read_text()).get("RepetitionTime", DEFAULT_TR))
            if tr > 0:
                return tr
        except Exception:
            pass
    return DEFAULT_TR


def project_to_fsaverage5(bold_path: Path, fsaverage) -> np.ndarray:  # noqa: F821
    """Project a 4D BOLD file onto fsaverage5 → (T, 20484) float32 (LH⊕RH)."""
    import nibabel as nib
    from nilearn import surface

    img = nib.load(str(bold_path))
    lh = surface.vol_to_surf(img, fsaverage.pial_left, radius=3.0, kind="auto")
    rh = surface.vol_to_surf(img, fsaverage.pial_right, radius=3.0, kind="auto")
    out = np.concatenate([lh.T, rh.T], axis=1).astype(np.float32)  # (T, V)
    if out.shape[1] != N_VERTICES:
        raise RuntimeError(f"Expected {N_VERTICES} vertices, got {out.shape[1]}")
    return out


def resample_to_1hz(arr: np.ndarray, tr_seconds: float) -> np.ndarray:  # noqa: F821
    """Linear-interpolate (T, V) at native TR → (T_1hz, V) at 1 Hz."""
    from scipy.interpolate import interp1d

    n_in = arr.shape[0]
    if n_in < 2:
        return arr.astype(np.float32)
    t_in = np.arange(n_in) * tr_seconds
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0)
    f = interp1d(t_in, arr, axis=0, kind="linear", fill_value="extrapolate")
    return f(t_out).astype(np.float32)


def zscore_per_vertex(arr: np.ndarray) -> np.ndarray:  # noqa: F821
    """Z-score each vertex over time. Constant vertices → 0 (no NaNs)."""
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = (arr - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def discover_runs(raw_root: Path, subjects: list[str] | None,
                  stories: list[str] | None, wav_stems: set[str]) -> dict:
    """Group bold files by (subject, story) -> ordered list of run paths.

    Multi-run stories are concatenated in run order.
    """
    grouped: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    sub_dirs = sorted(raw_root.glob("sub-*"))
    if subjects:
        keep = set(subjects)
        sub_dirs = [d for d in sub_dirs if d.name in keep]
    for sub_dir in sub_dirs:
        func = sub_dir / "func"
        if not func.exists():
            continue
        for nii in sorted(func.glob("*_bold.nii.gz")):
            m = TASK_RE.search(nii.name)
            if not m:
                continue
            story = normalize_story(m.group(1), wav_stems)
            if stories and story not in stories and m.group(1) not in stories:
                continue
            rm = RUN_RE.search(nii.name)
            run_idx = int(rm.group(1)) if rm else 0
            grouped[(sub_dir.name, story)].append((run_idx, nii))
    # sort each group's runs
    return {k: [p for _, p in sorted(v)] for k, v in grouped.items()}


def discover_cneuromod_runs(base: Path, groups: list[str],
                            subjects: list[str] | None,
                            stories: list[str] | None) -> dict:
    """Group CNeuroMod fmriprep BOLD by (subject, story) -> ordered run paths.

    Layout: {base}/{group}/sub-XX/ses-XXX/func/
            sub-XX_ses-XXX_task-<label>[_run-N]_<CNEUROMOD_SPACE>
    story = "{group}_{task}" (matches the feature-extractor key). Runs are
    ordered by (session, run) so multi-run / multi-session tasks concatenate in
    temporal order.
    """
    grouped: dict[tuple[str, str], list[tuple[int, int, Path]]] = defaultdict(list)
    sub_filter = set(subjects) if subjects else None
    story_filter = set(stories) if stories else None
    for group in groups:
        gdir = base / group
        if not gdir.is_dir():
            print(f"  (skip missing group dir {gdir})")
            continue
        for sub_dir in sorted(gdir.glob("sub-*")):
            if sub_filter and sub_dir.name not in sub_filter:
                continue
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                func = ses_dir / "func"
                if not func.is_dir():
                    continue
                for nii in sorted(func.glob(f"*_{CNEUROMOD_SPACE}")):
                    tm = TASK_RE.search(nii.name)
                    if not tm:
                        continue
                    task = tm.group(1)
                    story = f"{group}_{task}"
                    if story_filter and not (
                        story in story_filter or task in story_filter
                        or group in story_filter
                        or any(s in story for s in story_filter)
                    ):
                        continue
                    sm = SES_RE.search(nii.name)
                    rm = RUN_RE.search(nii.name)
                    ses_idx = int(sm.group(1)) if sm else 0
                    run_idx = int(rm.group(1)) if rm else 0
                    grouped[(sub_dir.name, story)].append((ses_idx, run_idx, nii))
    return {k: [p for *_, p in sorted(v)] for k, v in grouped.items()}


def discover_lebel_runs(base: Path, subjects: list[str] | None,
                        stories: list[str] | None,
                        feature_stories: set[str] | None = None) -> dict:
    """Group ds003020 raw BOLD by (subject, story) -> ordered run paths.

    Session-aware (reuses the CNeuroMod `sub-*/ses-*/func` walk) but for RAW
    BIDS bold (no `space-` tag) and with story = raw `task-<label>` (no group
    prefix, matching the lebel feature dirs). Runs ordered by (session, run).

    If `feature_stories` is given, tasks with no matching feature dir (e.g. the
    *Localizer tasks) are skipped at discovery time so we don't waste compute
    projecting BOLD that can never enter the manifest.
    """
    grouped: dict[tuple[str, str], list[tuple[int, int, Path]]] = defaultdict(list)
    sub_filter = set(subjects) if subjects else None
    story_filter = set(stories) if stories else None
    for sub_dir in sorted(base.glob("sub-*")):
        if sub_filter and sub_dir.name not in sub_filter:
            continue
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            func = ses_dir / "func"
            if not func.is_dir():
                continue
            for nii in sorted(func.glob("*_bold.nii.gz")):
                tm = TASK_RE.search(nii.name)
                if not tm:
                    continue
                story = tm.group(1)            # raw task label == feature key
                if feature_stories is not None and story not in feature_stories:
                    continue                   # drops localizers (no features)
                if story_filter and story not in story_filter:
                    continue
                sm = SES_RE.search(nii.name)
                rm = RUN_RE.search(nii.name)
                ses_idx = int(sm.group(1)) if sm else 0
                run_idx = int(rm.group(1)) if rm else 0
                grouped[(sub_dir.name, story)].append((ses_idx, run_idx, nii))
    return {k: [p for *_, p in sorted(v)] for k, v in grouped.items()}


def process_one(subject: str, story: str, run_paths: list[Path],
                out_root: Path, fsaverage,
                tr_override: float | None = None) -> dict:  # noqa: F821
    """Project + resample + z-score one (subject, story); concat multi-run.

    `tr_override` forces the TR (CNeuroMod movie runs are 1.49 s) instead of
    reading the BIDS sidecar.
    """
    out = out_root / subject / f"{story}.npy"
    if out.exists():
        n = int(np.load(out, mmap_mode="r").shape[0])
        return {"subject": subject, "story": story, "n_trs": n,
                "path": str(out), "status": "cached"}
    out.parent.mkdir(parents=True, exist_ok=True)

    parts: list[np.ndarray] = []
    for rp in run_paths:
        tr = tr_override if tr_override else read_tr(rp)
        surf = project_to_fsaverage5(rp, fsaverage)     # (T_tr, V)
        surf = resample_to_1hz(surf, tr)                # (T_1hz, V)
        parts.append(surf)
    arr = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    arr = zscore_per_vertex(arr)                        # per (subject,story) run
    np.save(out, arr)
    return {"subject": subject, "story": story, "n_trs": int(arr.shape[0]),
            "n_runs": len(run_paths), "path": str(out),
            "finite": bool(np.isfinite(arr).all()),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "status": "written"}


# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------
APP_NAME = "mary-prepare-fmri"

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "nilearn>=0.10.4",
        "nibabel>=5.2",
        "numpy>=1.26,<3",
        "scipy>=1.13",
        "scikit-learn>=1.4",
    )
)
_volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
_hf_secret = modal.Secret.from_name("hf-token")
_app = modal.App(APP_NAME)


@_app.function(
    image=_image,
    cpu=4.0,
    memory=16384,
    timeout=6 * 60 * 60,
    volumes={"/data": _volume},
    secrets=[_hf_secret],
)
def prepare_remote(
    raw_root: str = RAW_ROOT_DEFAULT,
    out_root: str = OUT_ROOT_DEFAULT,
    subjects: list[str] | None = None,
    stories: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Project ds002345 BOLD → fsaverage5/1Hz/z-scored on the sapient-data volume."""
    global np
    import numpy as np  # noqa: F811
    from nilearn import datasets as nl_datasets

    raw = Path(raw_root)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    wav_stems = {p.name.replace("_audio.wav", "")
                 for p in (raw / "stimuli").glob("*_audio.wav")}

    grouped = discover_runs(raw, subjects, stories, wav_stems)
    keys = sorted(grouped.keys())
    if limit:
        keys = keys[:limit]
    print(f"Found {len(keys)} (subject,story) groups to process.")

    print("Fetching fsaverage5 mesh ...")
    fsaverage = nl_datasets.fetch_surf_fsaverage(mesh=FSAVERAGE_MESH)

    results = []
    for i, (sub, story) in enumerate(keys):
        run_paths = grouped[(sub, story)]
        print(f"  [{i + 1}/{len(keys)}] {sub} / {story}  ({len(run_paths)} run(s))")
        try:
            r = process_one(sub, story, run_paths, out, fsaverage)
            print(f"      -> {r['status']} n_trs={r.get('n_trs')} "
                  f"mean={r.get('mean')} std={r.get('std')}")
            results.append(r)
        except Exception as e:
            print(f"      !! FAILED: {e}")
            results.append({"subject": sub, "story": story,
                            "status": "error", "error": str(e)})
        _volume.commit()

    written = [r for r in results if r["status"] == "written"]
    cached = [r for r in results if r["status"] == "cached"]
    errors = [r for r in results if r["status"] == "error"]
    summary = {
        "groups": len(keys),
        "written": len(written),
        "cached": len(cached),
        "errors": len(errors),
        "results": results,
    }
    print(f"\nDONE: written={len(written)} cached={len(cached)} errors={len(errors)}")
    return summary


@_app.function(
    image=_image,
    cpu=4.0,
    memory=32768,                 # MNI BOLD volumes are large; headroom for vol_to_surf
    timeout=12 * 60 * 60,
    volumes={"/data": _volume},
    secrets=[_hf_secret],
)
def prepare_cneuromod_remote(
    base: str = CNEUROMOD_BASE,
    out_root: str = CNEUROMOD_OUT,
    groups: list[str] | None = None,
    subjects: list[str] | None = None,
    stories: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Project CNeuroMod fmriprep BOLD (sub-*/ses-*/func) → fsaverage5/1Hz/z-scored.

    MNI152NLin2009cAsym volumetric → vol_to_surf → 1 Hz → z-score → (T,20484).
    Output: /data/fmri/mary/cneuromod/<subject>/<group>_<task>.npy (idempotent).
    """
    global np
    import numpy as np  # noqa: F811
    from nilearn import datasets as nl_datasets

    base_p = Path(base)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    groups = groups or list(CNEUROMOD_GROUPS)

    grouped = discover_cneuromod_runs(base_p, groups, subjects, stories)
    keys = sorted(grouped.keys())
    if limit:
        keys = keys[:limit]
    print(f"CNeuroMod: {len(keys)} (subject,story) groups to process "
          f"(groups={groups}).")

    print("Fetching fsaverage5 mesh ...")
    fsaverage = nl_datasets.fetch_surf_fsaverage(mesh=FSAVERAGE_MESH)

    results = []
    for i, (sub, story) in enumerate(keys):
        run_paths = grouped[(sub, story)]
        print(f"  [{i + 1}/{len(keys)}] {sub} / {story}  ({len(run_paths)} run(s))",
              flush=True)
        try:
            r = process_one(sub, story, run_paths, out, fsaverage,
                            tr_override=CNEUROMOD_TR)
            print(f"      -> {r['status']} n_trs={r.get('n_trs')} "
                  f"mean={r.get('mean')} std={r.get('std')}", flush=True)
            results.append(r)
        except Exception as e:
            print(f"      !! FAILED: {e}", flush=True)
            results.append({"subject": sub, "story": story,
                            "status": "error", "error": str(e)})
        _volume.commit()

    written = [r for r in results if r["status"] == "written"]
    cached = [r for r in results if r["status"] == "cached"]
    errors = [r for r in results if r["status"] == "error"]
    print(f"\nDONE: written={len(written)} cached={len(cached)} "
          f"errors={len(errors)}")
    return {"groups": len(keys), "written": len(written),
            "cached": len(cached), "errors": len(errors), "results": results}


@_app.function(
    image=_image,
    cpu=4.0,
    memory=24576,                 # one (subject,story); raw EPI vol headroom
    timeout=2 * 60 * 60,
    volumes={"/data": _volume},
    secrets=[_hf_secret],
)
def prepare_lebel_one(subject: str, story: str, run_path_strs: list[str],
                      out_root: str = LEBEL_OUT) -> dict:
    """Worker: project ONE lebel2023 (subject, story) → fsaverage5/1Hz/z.

    Runs in its OWN container so many (subject,story) jobs execute in parallel
    (fan-out via `.starmap`). TR read per-run from the BIDS sidecar (~2.0 s).
    """
    global np
    import numpy as np  # noqa: F811
    from nilearn import datasets as nl_datasets

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    fsaverage = nl_datasets.fetch_surf_fsaverage(mesh=FSAVERAGE_MESH)
    run_paths = [Path(p) for p in run_path_strs]
    try:
        r = process_one(subject, story, run_paths, out, fsaverage)
    except Exception as e:
        r = {"subject": subject, "story": story, "status": "error", "error": str(e)}
    _volume.commit()
    print(f"  {subject}/{story}: {r.get('status')} n_trs={r.get('n_trs')} "
          f"mean={r.get('mean')} std={r.get('std')} {r.get('error','')}", flush=True)
    return r


@_app.function(
    image=_image,
    cpu=2.0,
    memory=8192,
    timeout=12 * 60 * 60,
    volumes={"/data": _volume},
    secrets=[_hf_secret],
)
def prepare_lebel_remote(
    base: str = LEBEL_BASE,
    out_root: str = LEBEL_OUT,
    subjects: list[str] | None = None,
    stories: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Orchestrate ds003020 (lebel2023) projection — FANS OUT per (subject,story).

    Session-aware discovery, then `.starmap` over `prepare_lebel_one` so each
    (subject,story) projects in its OWN container (massively parallel). TR read
    per-run from the BIDS sidecar. Story key = raw `task-<label>` (matches
    /data/features/mary/lebel2023/_stories/<story>/). Localizer tasks (no feature
    dir) are dropped at discovery. Output /data/fmri/mary/lebel2023/<sub>/<story>.npy.
    """
    import numpy as np  # noqa: F401  (mesh fetch lives in workers)

    base_p = Path(base)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    feat_root = Path("/data/features/mary/lebel2023/_stories")
    feature_stories = ({d.name for d in feat_root.iterdir() if d.is_dir()}
                       if feat_root.is_dir() else None)
    print(f"Known feature stories: "
          f"{len(feature_stories) if feature_stories else 'ALL (no _stories dir)'}")

    grouped = discover_lebel_runs(base_p, subjects, stories, feature_stories)
    keys = sorted(grouped.keys())
    if limit:
        keys = keys[:limit]
    print(f"lebel2023: fanning out {len(keys)} (subject,story) jobs to parallel "
          f"containers ...", flush=True)

    args = [(sub, story, [str(p) for p in grouped[(sub, story)]], out_root)
            for (sub, story) in keys]
    results = list(prepare_lebel_one.starmap(args))

    written = [r for r in results if r.get("status") == "written"]
    cached = [r for r in results if r.get("status") == "cached"]
    errors = [r for r in results if r.get("status") == "error"]
    print(f"\nDONE: written={len(written)} cached={len(cached)} "
          f"errors={len(errors)}")
    return {"groups": len(keys), "written": len(written),
            "cached": len(cached), "errors": len(errors), "results": results}


def discover_wen_runs(base: Path, stories: list[str] | None) -> dict:
    """Group Wen2017 Subject-1 MNI BOLD by (subject, run) -> [ALL repeat paths].

    Layout: {base}/{seg|test}{N}/mni/{run}_{rep}_mni.nii.gz (a few use a `.mni`
    separator instead of `_mni`). The run key is the segment dir name
    (seg1..seg18, test1..test5), matching the stimulus stem. Each segment has
    MULTIPLE repeated viewings (seg*: 2 reps, test*: 10 reps); we return all
    repeat files so the projector can repeat-AVERAGE them (the Wen2017 / ORCLE
    convention — averaging across viewings raises SNR toward the noise ceiling).
    """
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    story_filter = set(stories) if stories else None
    if not base.is_dir():
        print(f"  (wen2017 base missing: {base})")
        return {}
    for seg_dir in sorted(base.iterdir()):
        if not seg_dir.is_dir():
            continue
        run = seg_dir.name                      # seg1..seg18, test1..test5
        if story_filter and run not in story_filter:
            continue
        mni = seg_dir / "mni"
        if not mni.is_dir():
            continue
        # Accept both `*_mni.nii.gz` and the stray `*.mni.nii.gz` spelling.
        niis = sorted(set(mni.glob("*_mni.nii.gz")) | set(mni.glob("*.mni.nii.gz")))
        if not niis:
            niis = sorted(mni.glob("*.nii.gz"))
        if niis:
            grouped[(WEN_SUBJECT, run)] = list(niis)   # ALL repeats
    return dict(grouped)


def process_wen_one(subject: str, run: str, rep_paths: list[Path],
                    out_root: Path, fsaverage) -> dict:  # noqa: F821
    """Project + repeat-AVERAGE one Wen2017 (subject, run).

    Each rep is projected to fsaverage5 then resampled to 1 Hz (TR=2.0 s); the
    repeats are truncated to the common length and averaged BEFORE z-scoring
    (standard Wen2017 multi-repeat denoising). Idempotent: skips if output exists.
    """
    out = out_root / subject / f"{run}.npy"
    if out.exists():
        n = int(np.load(out, mmap_mode="r").shape[0])
        return {"subject": subject, "story": run, "n_trs": n,
                "n_reps": len(rep_paths), "path": str(out), "status": "cached"}
    out.parent.mkdir(parents=True, exist_ok=True)

    rep_arrs = []
    for rp in rep_paths:
        surf = project_to_fsaverage5(rp, fsaverage)     # (T_tr, V)
        surf = resample_to_1hz(surf, WEN_TR)            # (T_1hz, V)
        rep_arrs.append(surf)
    if not rep_arrs:
        raise RuntimeError(f"no repeats for {subject}/{run}")
    # Average across repeats on the common (shortest) length.
    min_t = min(a.shape[0] for a in rep_arrs)
    stacked = np.stack([a[:min_t] for a in rep_arrs], axis=0)  # (R, T, V)
    avg = stacked.mean(axis=0)                                 # (T, V)
    arr = zscore_per_vertex(avg)
    np.save(out, arr)
    return {"subject": subject, "story": run, "n_trs": int(arr.shape[0]),
            "n_reps": len(rep_paths), "path": str(out),
            "finite": bool(np.isfinite(arr).all()),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "status": "written"}


@_app.function(
    image=_image,
    cpu=4.0,
    memory=32768,                 # MNI BOLD 4D volumes; headroom for vol_to_surf
    timeout=12 * 60 * 60,
    volumes={"/data": _volume},
    secrets=[_hf_secret],
)
def prepare_wen_remote(
    base: str = WEN_BASE,
    out_root: str = WEN_OUT,
    stories: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Project Wen2017 Subject-1 MNI BOLD → fsaverage5/1Hz/z-scored.

    MNI152 volumetric → vol_to_surf → 1 Hz (from TR=2.0 s) → z-score → (T,20484).
    Output: /data/fmri/mary/wen2017/subject1/{seg1..test5}.npy (idempotent).
    """
    global np
    import numpy as np  # noqa: F811
    from nilearn import datasets as nl_datasets

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    grouped = discover_wen_runs(Path(base), stories)
    keys = sorted(grouped.keys())
    if limit:
        keys = keys[:limit]
    print(f"Wen2017: {len(keys)} (subject,run) groups to process.", flush=True)

    print("Fetching fsaverage5 mesh ...")
    fsaverage = nl_datasets.fetch_surf_fsaverage(mesh=FSAVERAGE_MESH)

    results = []
    for i, (sub, run) in enumerate(keys):
        rep_paths = grouped[(sub, run)]
        print(f"  [{i + 1}/{len(keys)}] {sub} / {run}  ({len(rep_paths)} repeat(s))",
              flush=True)
        try:
            r = process_wen_one(sub, run, rep_paths, out, fsaverage)
            print(f"      -> {r['status']} n_trs={r.get('n_trs')} "
                  f"n_reps={r.get('n_reps')} mean={r.get('mean')} std={r.get('std')}",
                  flush=True)
            results.append(r)
        except Exception as e:
            print(f"      !! FAILED: {e}", flush=True)
            results.append({"subject": sub, "story": run,
                            "status": "error", "error": str(e)})
        _volume.commit()

    written = [r for r in results if r["status"] == "written"]
    cached = [r for r in results if r["status"] == "cached"]
    errors = [r for r in results if r["status"] == "error"]
    print(f"\nDONE: written={len(written)} cached={len(cached)} "
          f"errors={len(errors)}")
    return {"groups": len(keys), "written": len(written),
            "cached": len(cached), "errors": len(errors), "results": results}


@_app.local_entrypoint()
def wen(
    base: str = WEN_BASE,
    out_root: str = WEN_OUT,
    stories: str = "",
    limit: int = 0,
) -> None:
    """Project Wen2017 Subject-1 MNI BOLD → fsaverage5.

    `modal run [--detach] data/prepare_fmri.py::wen [--stories seg1,test1] [--limit N]`
    """
    sts = [s for s in stories.split(",") if s] or None
    res = prepare_wen_remote.remote(base, out_root, sts, limit or None)
    print("\n========== WEN2017 SUMMARY ==========")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    for r in res["results"][:40]:
        print(" ", r.get("subject"), r.get("story"), r.get("status"),
              "n_trs=", r.get("n_trs"))


# ~6-8 subjects spread across high-coverage stories for the 24h sprint.
DEFAULT_SLICE_SUBJECTS = [
    "sub-001", "sub-002", "sub-003", "sub-004",
    "sub-005", "sub-006", "sub-007", "sub-008",
]


@_app.local_entrypoint()
def main(
    raw_root: str = RAW_ROOT_DEFAULT,
    out_root: str = OUT_ROOT_DEFAULT,
    subjects: str = ",".join(DEFAULT_SLICE_SUBJECTS),
    stories: str = "",
    limit: int = 0,
) -> None:
    """`modal run data/prepare_fmri.py [--subjects a,b] [--stories x,y] [--limit N]`"""
    subs = [s for s in subjects.split(",") if s] or None
    sts = [s for s in stories.split(",") if s] or None
    res = prepare_remote.remote(raw_root, out_root, subs, sts,
                                limit or None)
    print("\n========== SUMMARY ==========")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    for r in res["results"][:40]:
        print(" ", r.get("subject"), r.get("story"), r.get("status"),
              "n_trs=", r.get("n_trs"))


@_app.local_entrypoint()
def cneuromod(
    base: str = CNEUROMOD_BASE,
    out_root: str = CNEUROMOD_OUT,
    groups: str = ",".join(CNEUROMOD_GROUPS),
    subjects: str = "sub-01,sub-02",
    stories: str = "",
    limit: int = 0,
) -> None:
    """Project CNeuroMod fmriprep BOLD → fsaverage5.

    `modal run data/prepare_fmri.py::cneuromod [--groups movie10] \
        [--subjects sub-01,sub-02] [--stories movie10_bourne05] [--limit N]`
    """
    grps = [g for g in groups.split(",") if g] or None
    subs = [s for s in subjects.split(",") if s] or None
    sts = [s for s in stories.split(",") if s] or None
    res = prepare_cneuromod_remote.remote(base, out_root, grps, subs, sts,
                                          limit or None)
    print("\n========== CNEUROMOD SUMMARY ==========")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    for r in res["results"][:40]:
        print(" ", r.get("subject"), r.get("story"), r.get("status"),
              "n_trs=", r.get("n_trs"))


@_app.local_entrypoint()
def lebel(
    base: str = LEBEL_BASE,
    out_root: str = LEBEL_OUT,
    subjects: str = "sub-UTS01,sub-UTS02",
    stories: str = "",
    limit: int = 0,
) -> None:
    """Project ds003020 (lebel2023) raw BOLD → fsaverage5 (session-aware).

    `modal run [--detach] data/prepare_fmri.py::lebel [--subjects sub-UTS01] \
        [--stories christmas1940,life] [--limit N]`
    """
    subs = [s for s in subjects.split(",") if s] or None
    sts = [s for s in stories.split(",") if s] or None
    res = prepare_lebel_remote.remote(base, out_root, subs, sts, limit or None)
    print("\n========== LEBEL2023 SUMMARY ==========")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    for r in res["results"][:40]:
        print(" ", r.get("subject"), r.get("story"), r.get("status"),
              "n_trs=", r.get("n_trs"))


# Local CLI shim (not used on Modal but kept for parity with the family).
def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=RAW_ROOT_DEFAULT)
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--stories", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit(
        "This script runs on Modal; use `modal run data/prepare_fmri.py`. "
        f"(args={args})"
    )


if __name__ == "__main__":
    _cli()
