"""Quick status of HAD processing on the volume: referenced clip count, a sample
feature shape, fMRI outputs, and assembled per-run timelines."""
from __future__ import annotations
import modal

app = modal.App("mary-check-had")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26,<3")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)


@app.function(image=image, cpu=2.0, memory=4096, timeout=15 * 60,
              volumes={"/data": vol})
def check() -> dict:
    import csv
    from pathlib import Path
    import numpy as np

    raw = Path("/data/raw/ds004488")
    out: dict = {}

    # referenced clips per subject + union
    ses = "ses-action01"
    union: set[str] = set()
    per_sub = {}
    for sub in ("sub-01", "sub-02"):
        func = raw / sub / ses / "func"
        rels: set[str] = set()
        for ev in sorted(func.glob(f"{sub}_{ses}_task-action_run-*_events.tsv")):
            with ev.open() as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    s = (r.get("stim_file") or "").strip()
                    if s and s not in ("n/a", "nan"):
                        rels.add(s)
        per_sub[sub] = len(rels)
        union |= rels
    out["referenced_clips_per_subject"] = per_sub
    out["referenced_clips_union"] = len(union)
    print("referenced clips:", per_sub, "union", len(union))

    # per-stream _stories coverage
    stories = Path("/data/features/mary/had/_stories")
    for stream in ("slowfast", "qwen_vl", "whisper", "beats", "got_ocr"):
        n = sum(1 for d in stories.glob("*") if (d / f"{stream}.npy").exists()) \
            if stories.exists() else 0
        # sample shape
        shape = None
        if stories.exists():
            for d in stories.glob("*"):
                p = d / f"{stream}.npy"
                if p.exists():
                    shape = list(np.load(p, mmap_mode="r").shape)
                    break
        out[f"stories_{stream}"] = {"n_clips_with_feature": n, "sample_shape": shape}
        print(f"  {stream}: {n} clips, sample {shape}")

    # fMRI outputs
    fmri = Path("/data/fmri/mary/had")
    fr = {}
    if fmri.exists():
        for sub in sorted(fmri.glob("sub-*")):
            runs = sorted(p.name for p in sub.glob("run-*.npy"))
            fr[sub.name] = runs
            if runs:
                sh = list(np.load(sub / runs[0], mmap_mode="r").shape)
                out[f"fmri_{sub.name}_sample_shape"] = sh
    out["fmri_runs"] = fr
    print("fmri runs:", {k: len(v) for k, v in fr.items()})

    # assembled per-run timelines
    feat = Path("/data/features/mary/had")
    asm = {}
    for sub in ("sub-01", "sub-02"):
        sd = feat / sub
        if sd.exists():
            asm[sub] = {rd.name: sorted(p.name for p in rd.glob("*.npy"))
                        for rd in sorted(sd.glob("run-*"))}
    out["assembled"] = asm
    return out


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(check.remote(), indent=2)[:5000])
