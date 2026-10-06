"""Confirm CNeuroMod BOLD .nii.gz on the volume are REAL fetched content,
not dangling git-annex pointers. Fast (no full du walk). CPU, read-only.

    modal run sapient1/scripts/check_bold_realness.py
"""
from __future__ import annotations
import modal

app = modal.App("sapient-1-bold-realness")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("sapient-data", create_if_missing=False)
POINTER_MAX = 4096


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800,
              volumes={"/data": volume})
def check() -> dict:
    from pathlib import Path
    out = {}
    for task in ["friends", "movie10"]:
        root = Path(f"/data/raw/cneuromod/fmriprep/{task}")
        if not root.exists():
            out[task] = {"exists": False}
            continue
        per_sub = {}
        for sub in ["sub-01", "sub-02", "sub-03", "sub-05"]:
            sd = root / sub
            if not sd.exists():
                continue
            present = absent = 0
            gb = 0.0
            stems = set()
            for fp in sd.rglob("*_bold.nii.gz"):
                if "/.git/" in str(fp):
                    continue
                try:
                    real = fp.resolve()
                    sz = real.stat().st_size if real.exists() else 0
                    if sz > POINTER_MAX:
                        present += 1
                        gb += sz / 1e9
                        # task token e.g. task-s03e05a
                        name = fp.name
                        if "task-" in name:
                            stems.add(name.split("task-")[1].split("_")[0])
                    else:
                        absent += 1
                except OSError:
                    absent += 1
            seasons = sorted({s[:3] for s in stems if s.startswith("s")})
            per_sub[sub] = {
                "bold_runs_present": present, "bold_runs_absent": absent,
                "present_GB": round(gb, 1),
                "n_distinct_stimuli": len(stems),
                "seasons_or_movies": seasons[:12],
            }
        out[task] = per_sub
    return out


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(check.remote(), indent=2))
