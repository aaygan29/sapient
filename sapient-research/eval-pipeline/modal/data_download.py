"""
sapienteval/modal/data_download.py

Modal CPU app that pulls the NeuroEngage BIDS dataset (OpenNeuro ds004996)
directly onto the sapienteval-data volume — no local disk required.

Why this exists:
  Running `datalad get sub-01..sub-05` locally requires ~30GB free disk +
  a datalad install. Doing it inside a Modal container avoids both, keeps
  raw data colocated with the rest of the Track B pipeline (fMRIPrep,
  parcellation, training), and the cost is negligible (a few minutes of
  CPU compute, ~$0.50).

Dataset: ds004996 (NeuroEngage, Torubarova et al., HRI 2025, CC0 license).
  - 50 subjects: 30 human-partner, 20 robot-partner (Furhat)
  - 3 BOLD runs per subject + anat + fieldmaps
  - Total: ~30GB for all 50 subjects, ~5GB for the 5-subject pilot

Layout written:
    /data/ds004996/
      participants.tsv
      sub-01/
        anat/sub-01_T1w.nii.gz
        anat/sub-01_T1w.json
        func/sub-01_task-engage_run-1_bold.nii.gz
        func/sub-01_task-engage_run-1_bold.json
        func/sub-01_task-engage_run-1_events.tsv
        fmap/...
      sub-02/...

Run (after deploy):
    modal run sapienteval/modal/data_download.py
    modal run sapienteval/modal/data_download.py --subjects sub-01,sub-02
    modal run sapienteval/modal/data_download.py --all   # all 50 subjects
"""

import modal
import json
import time
from pathlib import Path

data_vol = modal.Volume.from_name("sapienteval-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git")
    # openneuro-py downloads files via HTTPS directly (no git-annex symlink
    # indirection). datalad was tried first but its get-by-symlink model
    # writes 208-byte pointers into the .nii path instead of real bytes,
    # which made fMRIPrep fail with FileNotFoundError downstream.
    .pip_install("openneuro-py>=2024.2.0", "aiohttp", "tqdm")
)

app = modal.App("sapienteval-data-download", image=image)

DS_REPO = "https://github.com/OpenNeuroDatasets/ds004996.git"
DS_NAME = "ds004996"


@app.function(
    cpu=4,
    memory=8192,
    timeout=4 * 3600,
    volumes={"/data": data_vol},
)
def download_subjects(subject_ids: list[str], include_fieldmaps: bool = True) -> dict:
    """
    Pull a list of NeuroEngage (ds004996) subjects via openneuro-py.

    openneuro-py downloads via the OpenNeuro HTTPS API — real bytes land on
    disk, no git-annex symlink indirection. Idempotent: openneuro.download
    skips files that already exist with the expected size.

    Returns: {status, n_subjects, files_fetched, bytes_fetched, duration_s,
              per_subject_results}
    """
    import openneuro

    t0 = time.time()
    root = Path("/data") / DS_NAME
    root.mkdir(parents=True, exist_ok=True)

    # First, fetch dataset-level metadata + participants.tsv for the splits
    # verification step.
    try:
        openneuro.download(
            dataset=DS_NAME,
            target_dir=str(root),
            include=["participants.tsv", "dataset_description.json", "README"],
        )
    except Exception as e:
        print(f"[meta] participants.tsv fetch failed (continuing): {e}")

    per_subject: list[dict] = []
    total_files = 0
    total_bytes = 0

    for sid in subject_ids:
        # openneuro-py's `include` filter accepts BIDS-style path prefixes.
        # We grab everything under `sub-XX/` then optionally drop fieldmaps.
        include = [f"{sid}/"]
        exclude: list[str] = []
        if not include_fieldmaps:
            exclude.append(f"{sid}/fmap/")

        print(f"[download] {sid} (include={include}, exclude={exclude})")
        sub_t0 = time.time()
        try:
            openneuro.download(
                dataset=DS_NAME,
                target_dir=str(root),
                include=include,
                exclude=exclude if exclude else None,
            )
            sub_dir = root / sid
            sub_files = list(sub_dir.rglob("*")) if sub_dir.exists() else []
            sub_files = [p for p in sub_files if p.is_file()]
            sub_bytes = sum(p.stat().st_size for p in sub_files)
            total_files += len(sub_files)
            total_bytes += sub_bytes
            per_subject.append({
                "subject_id": sid,
                "status": "ok",
                "n_files": len(sub_files),
                "bytes": sub_bytes,
                "duration_s": round(time.time() - sub_t0, 1),
            })
        except Exception as e:
            per_subject.append({
                "subject_id": sid,
                "status": "error",
                "error": str(e)[:500],
                "duration_s": round(time.time() - sub_t0, 1),
            })

    data_vol.commit()

    return {
        "status": "ok",
        "n_subjects": len(subject_ids),
        "files_fetched": total_files,
        "bytes_fetched": total_bytes,
        "duration_s": round(time.time() - t0, 2),
        "per_subject_results": per_subject,
    }


@app.function(
    cpu=2,
    memory=2048,
    timeout=20 * 60,
    volumes={"/data": data_vol},
)
def download_annotations() -> dict:
    """Clone OsLjung/NeuroEngage-annotation-data to /data/annotations/.

    The events.tsv files in ds004996 only contain coarse event categories
    (fixation_cross / comprehension / production / silence / turn_initiation).
    The real spoken transcripts live in this separate annotation repo, keyed
    by timestamp under each per-(subject, run) JSON file.

    After this runs, align_subject() in postprocess.py can merge per-timestamp
    transcripts into the per-TR text windows.
    """
    import subprocess

    t0 = time.time()
    target = Path("/data/annotations")
    repo_url = "https://github.com/OsLjung/NeuroEngage-annotation-data.git"

    if target.exists() and any(target.iterdir()):
        # Pull updates if already cloned
        result = subprocess.run(
            ["git", "-C", str(target), "pull", "--ff-only"],
            capture_output=True, text=True,
        )
        action = "pulled"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(target)],
            capture_output=True, text=True,
        )
        action = "cloned"

    if result.returncode != 0:
        return {
            "status": "error",
            "action": action,
            "stderr": result.stderr[-500:],
        }

    # Inventory: count JSON files per subdirectory
    inventory: dict[str, int] = {}
    for sub in sorted(target.iterdir()):
        if sub.is_dir():
            json_files = list(sub.rglob("*.json"))
            inventory[sub.name] = len(json_files)

    # Sample one file to confirm transcript content
    sample_files = list(target.rglob("*.json"))
    sample = None
    if sample_files:
        sample_path = sample_files[0]
        sample = {
            "path": str(sample_path.relative_to(target)),
            "size_bytes": sample_path.stat().st_size,
        }
        try:
            data = json.loads(sample_path.read_text())
            ann = data.get("annotation", {})
            participant_turns = sum(
                1 for tiers in ann.values()
                if isinstance(tiers, dict) and tiers.get("participant", {}).get("value")
            )
            sample["subject_id"] = data.get("subject_id")
            sample["run"] = data.get("run")
            sample["n_annotation_keys"] = len(ann)
            sample["n_participant_turns"] = participant_turns
        except Exception as e:
            sample["parse_error"] = str(e)[:200]

    data_vol.commit()
    return {
        "status": "ok",
        "action": action,
        "target": str(target),
        "inventory": inventory,
        "n_total_json_files": sum(inventory.values()),
        "sample_file": sample,
        "duration_s": round(time.time() - t0, 2),
    }


@app.function(
    cpu=2,
    memory=2048,
    volumes={"/data": data_vol},
)
def list_dataset() -> dict:
    """Quick inventory of what's currently on the sapienteval-data volume."""
    root = Path("/data") / DS_NAME
    if not root.exists():
        return {"status": "empty", "message": f"{root} not present"}

    subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    files_per_subject = {}
    for s in subjects:
        sd = root / s
        files_per_subject[s] = {
            "n_T1w": len(list(sd.glob("anat/*_T1w.nii.gz"))),
            "n_bold": len(list(sd.glob("func/*_bold.nii.gz"))),
            "n_events": len(list(sd.glob("func/*_events.tsv"))),
            "n_fmap": len(list(sd.glob("fmap/*.nii.gz"))),
        }

    return {
        "status": "ok",
        "n_subjects": len(subjects),
        "subjects": subjects,
        "files_per_subject": files_per_subject,
        "has_participants_tsv": (root / "participants.tsv").exists(),
    }


@app.local_entrypoint()
def main(subjects: str = "", all: bool = False, include_fieldmaps: bool = True):
    """
    Pull NeuroEngage subjects onto the sapienteval-data volume.

    Defaults to the pilot subject list from sapienteval/splits.json.
    """
    if all:
        subject_list = [f"sub-{i:02d}" for i in range(1, 51)]
    elif subjects:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
    else:
        splits_path = Path("sapienteval/splits.json")
        if splits_path.exists():
            subject_list = json.loads(splits_path.read_text())["pilot"]["subjects"]
        else:
            subject_list = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]

    print(f"Downloading {len(subject_list)} subjects: {subject_list}")
    print(f"Include fieldmaps: {include_fieldmaps}")
    print(f"Target volume: sapienteval-data")

    result = download_subjects.remote(subject_list, include_fieldmaps)
    print(json.dumps(result, indent=2))

    print("\n=== Post-download inventory ===")
    inventory = list_dataset.remote()
    print(json.dumps(inventory, indent=2))
