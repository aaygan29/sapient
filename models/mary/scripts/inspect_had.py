"""Quick Modal inspection of the HAD dataset (ds004488) on sapient-data.

Confirms layout for processing into Mary inputs:
  - stimuli/{ActionCategory}/*.mp4 clip tree (count, sample names, sizes)
  - derivatives/fmriprep/sub-01|02 func bold (space-MNI152..._desc-preproc_bold.nii.gz)
  - events.tsv columns + sample rows (the clip<->onset alignment source)
  - bold JSON sidecar (RepetitionTime / TR)

Run:  modal run scripts/inspect_had.py
"""

from __future__ import annotations

import modal

APP_NAME = "mary-inspect-had"

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26,<3")
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
app = modal.App(APP_NAME)

RAW = "/data/raw/ds004488"
FMRIPREP = "/data/raw/ds004488/derivatives/fmriprep"


@app.function(image=image, cpu=2.0, memory=4096, timeout=20 * 60,
              volumes={"/data": volume})
def inspect() -> dict:
    import json
    from collections import defaultdict
    from pathlib import Path

    root = Path(RAW)
    out: dict = {}
    if not root.exists():
        return {"error": f"{root} does not exist"}

    print("=== top-level ds004488 ===")
    for p in sorted(root.iterdir()):
        print(" ", p.name, "(dir)" if p.is_dir() else f"({p.stat().st_size} B)")

    # ---- stimuli tree ----
    stim = root / "stimuli"
    cats = sorted([d.name for d in stim.iterdir() if d.is_dir()]) if stim.exists() else []
    mp4s = sorted(stim.rglob("*.mp4")) if stim.exists() else []
    print(f"\n=== stimuli: {len(cats)} categories, {len(mp4s)} mp4 clips ===")
    print("  categories:", cats[:60])
    for m in mp4s[:8]:
        print("   sample clip:", m.relative_to(stim), m.stat().st_size, "B")
    out["n_categories"] = len(cats)
    out["categories"] = cats
    out["n_clips"] = len(mp4s)
    out["sample_clips"] = [str(m.relative_to(stim)) for m in mp4s[:20]]

    # ---- fmriprep subjects ----
    fp = Path(FMRIPREP)
    print(f"\n=== fmriprep at {fp} ===")
    if not fp.exists():
        print("  MISSING")
    subs = sorted([p.name for p in fp.glob("sub-*") if p.is_dir()]) if fp.exists() else []
    out["fmriprep_subjects"] = subs
    print("  subjects:", subs)

    bold_info = defaultdict(list)
    sample_sidecar = None
    for sub in subs:
        func = fp / sub / "func"
        if not func.exists():
            continue
        for nii in sorted(func.glob("*bold.nii.gz")):
            bold_info[sub].append({"name": nii.name, "size": nii.stat().st_size})
        if sample_sidecar is None:
            for js in sorted(func.glob("*_bold.json")):
                try:
                    sample_sidecar = {"name": js.name, "json": json.loads(js.read_text())}
                except Exception as e:
                    sample_sidecar = {"name": js.name, "error": str(e)}
                break

    print("\n=== bold files per subject (all spaces) ===")
    for sub in subs:
        files = bold_info.get(sub, [])
        print(f"  {sub}: {len(files)} *bold.nii.gz")
        for f in files[:20]:
            print(f"      {f['name']}  {f['size']} B")
    out["bold_counts"] = {s: len(bold_info.get(s, [])) for s in subs}
    out["bold_sample"] = {s: [f["name"] for f in bold_info.get(s, [])[:20]] for s in subs}

    # MNI preproc bold specifically
    mni_glob = "*task-action*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz"
    mni_counts = {}
    for sub in subs:
        func = fp / sub / "func"
        mni = sorted(func.glob(mni_glob)) if func.exists() else []
        mni_counts[sub] = [m.name for m in mni]
    out["mni_preproc_bold"] = mni_counts
    print("\n=== MNI preproc task-action bold ===")
    for s, lst in mni_counts.items():
        print(f"  {s}: {len(lst)}")
        for n in lst:
            print("     ", n)

    # ---- events.tsv (raw BIDS func, not derivatives) ----
    print("\n=== events.tsv (raw func) ===")
    ev_samples = {}
    for sub in subs:
        # events live in the raw BIDS tree, not derivatives
        cand_funcs = [root / sub / "func", fp / sub / "func"]
        evs = []
        for cf in cand_funcs:
            if cf.exists():
                evs += sorted(cf.glob("*task-action*_events.tsv"))
        if not evs:
            continue
        ev = evs[0]
        text = ev.read_text()
        lines = text.splitlines()
        ev_samples[sub] = {
            "path": str(ev),
            "n_event_files": len(evs),
            "header": lines[0] if lines else "",
            "first_rows": lines[1:6],
            "n_rows": len(lines) - 1,
            "all_event_files": [str(e) for e in evs[:30]],
        }
        print(f"  {sub}: {len(evs)} events.tsv files; sample {ev.name}")
        print(f"     header: {lines[0] if lines else ''}")
        for r in lines[1:6]:
            print(f"     row:    {r}")
    out["events"] = ev_samples

    print("\n=== sample bold sidecar ===")
    print(json.dumps(sample_sidecar, indent=2)[:1500] if sample_sidecar else "NONE")
    out["sample_sidecar"] = sample_sidecar

    return out


@app.local_entrypoint()
def main() -> None:
    import json
    res = inspect.remote()
    print("\n========== RESULT ==========")
    print(json.dumps(res, indent=2)[:8000])
