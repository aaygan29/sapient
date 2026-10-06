"""Read-only audit of the `sapient-data` Modal volume for Sapient-1.

Answers the founder's key question: how much CNeuroMod (Friends/movie10) is
ACTUALLY downloaded (real bytes, not git-annex pointers) and feature-extracted,
vs what the shipped first-light checkpoint actually trained on.

Cheap CPU job. No writes. Run:
    modal run sapient1/scripts/inspect_data_state.py
"""
from __future__ import annotations

import modal

app = modal.App("sapient-1-inspect-data")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("sapient-data", create_if_missing=False)

# git-annex unfetched pointers are tiny symlinks/files (~150-300 bytes).
ANNEX_POINTER_MAX = 4096


@app.function(image=image, cpu=2.0, memory=4096, timeout=3600,
              volumes={"/data": volume})
def inspect() -> dict:
    import os
    from pathlib import Path

    def du(path: str) -> int:
        total = 0
        p = Path(path)
        if not p.exists():
            return -1
        for root, _dirs, files in os.walk(path):
            if "/.git/" in root + "/" or root.endswith("/.git"):
                # skip git internals EXCEPT annex objects (real fetched bytes)
                if "/annex/objects" not in root:
                    continue
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if os.path.islink(fp):
                        continue
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    def count_real_videos(root: str, pattern_suffix: str = ".mkv") -> dict:
        """Count .mkv files that are REAL (size > pointer threshold) vs pointers.
        git-annex stores real content as symlinks into annex/objects; an
        unfetched file is a dangling symlink. A resolved symlink with real
        target = present. We resolve and stat."""
        p = Path(root)
        if not p.exists():
            return {"exists": False}
        present, absent, present_bytes = 0, 0, 0
        for fp in p.rglob(f"*{pattern_suffix}"):
            if "/.git/" in str(fp):
                continue
            try:
                real = fp.resolve()
                if real.exists() and not real.is_symlink():
                    sz = real.stat().st_size
                    if sz > ANNEX_POINTER_MAX:
                        present += 1
                        present_bytes += sz
                    else:
                        absent += 1
                else:
                    absent += 1
            except OSError:
                absent += 1
        return {"exists": True, "present": present, "absent": absent,
                "present_GB": round(present_bytes / 1e9, 2)}

    def count_npy(root: str) -> dict:
        p = Path(root)
        if not p.exists():
            return {"exists": False}
        files = [f for f in p.rglob("*.npy") if "/.git/" not in str(f)]
        total = sum(f.stat().st_size for f in files if f.exists())
        return {"exists": True, "count": len(files),
                "total_MB": round(total / 1e6, 1),
                "sample": sorted(f.name for f in files)[:6]}

    report: dict = {}

    # 0. Realness of CNeuroMod BOLD: are the *_bold.nii.gz resolved (real
    #    content) or dangling annex pointers? Count per subject for friends.
    def bold_realness(task: str):
        root = Path(f"/data/raw/cneuromod/fmriprep/{task}")
        if not root.exists():
            return {"exists": False}
        per_sub = {}
        for sub in ["sub-01", "sub-02", "sub-03", "sub-05"]:
            sd = root / sub
            if not sd.exists():
                continue
            present, absent, gb = 0, 0, 0.0
            for fp in sd.rglob("*_bold.nii.gz"):
                if "/.git/" in str(fp):
                    continue
                try:
                    real = fp.resolve()
                    if real.exists() and real.stat().st_size > ANNEX_POINTER_MAX:
                        present += 1
                        gb += real.stat().st_size / 1e9
                    else:
                        absent += 1
                except OSError:
                    absent += 1
            per_sub[sub] = {"bold_present": present, "bold_absent": absent,
                            "present_GB": round(gb, 1)}
        return per_sub
    report["cneuromod_friends_bold_realness"] = bold_realness("friends")
    report["cneuromod_movie10_bold_realness"] = bold_realness("movie10")

    # 1. Raw CNeuroMod (the download_cneuromod.py target).
    report["raw_cneuromod_friends_bold_GB"] = round(
        du("/data/raw/cneuromod/fmriprep/friends") / 1e9, 2)
    report["raw_cneuromod_movie10_bold_GB"] = round(
        du("/data/raw/cneuromod/fmriprep/movie10") / 1e9, 2)
    report["raw_cneuromod_friends_stimuli"] = count_real_videos(
        "/data/raw/cneuromod/fmriprep/friends/sourcedata/friends")

    # 2. Algonauts 2025 (the OTHER, possibly-complete corpus).
    report["algonauts_friends_videos"] = count_real_videos(
        "/data/raw/algonauts2025/stimuli/movies/friends")
    report["algonauts_movie10_videos"] = count_real_videos(
        "/data/raw/algonauts2025/stimuli/movies/movie10")
    report["algonauts_fmri_GB"] = round(
        du("/data/raw/algonauts2025/fmri") / 1e9, 2)

    # 3. Processed fMRI actually used by training (per dataset.py/manifest).
    for sub in ["sub-01", "sub-02", "sub-03", "sub-05"]:
        report[f"fmri_cneuromod_{sub}"] = count_npy(
            f"/data/fmri/cneuromod/{sub}")

    # 4. Extracted features (the real cost driver — V-JEPA2 + W2V-BERT).
    report["features_vjepa2_cneuromod"] = count_npy(
        "/data/features/vjepa2/cneuromod")
    report["features_w2vbert_cneuromod"] = count_npy(
        "/data/features/w2vbert/cneuromod")

    # 5. Manifest used by sapient-1 training.
    for mf in ["manifest_sapient1.json", "manifest.json"]:
        mp = Path("/data") / mf
        if mp.exists():
            import json
            try:
                data = json.loads(mp.read_text())
                if isinstance(data, list):
                    report[f"{mf}_entries"] = len(data)
                elif isinstance(data, dict):
                    report[f"{mf}_keys"] = list(data.keys())[:10]
                    for k in ("train", "val", "test", "samples", "windows"):
                        if k in data and isinstance(data[k], list):
                            report[f"{mf}_{k}"] = len(data[k])
            except Exception as e:
                report[f"{mf}_error"] = str(e)

    # 6. Shipped checkpoints.
    ck = Path("/data/checkpoints")
    if ck.exists():
        report["checkpoints"] = sorted(
            p.name for p in ck.rglob("*.pt"))[:20]

    return report


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(inspect.remote(), indent=2))
