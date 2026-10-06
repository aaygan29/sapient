"""Quick Modal inspection of ds002345 on the sapient-data volume.

Confirms sub-*/func/*_bold.nii.gz are real bytes, lists stories
(stimuli/*_audio.wav), and which subjects have which tasks/runs.

Run:  modal run scripts/inspect_ds002345.py
"""

from __future__ import annotations

import modal

APP_NAME = "mary-inspect-ds002345"

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26,<3")
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
app = modal.App(APP_NAME)

ROOT = "/data/raw/ds002345"


@app.function(image=image, cpu=2.0, memory=4096, timeout=20 * 60,
              volumes={"/data": volume})
def inspect() -> dict:
    import json
    import os
    from collections import defaultdict
    from pathlib import Path

    root = Path(ROOT)
    if not root.exists():
        return {"error": f"{root} does not exist"}

    print("=== top-level ===")
    for p in sorted(root.iterdir()):
        print(" ", p.name, "(dir)" if p.is_dir() else f"({p.stat().st_size} B)")

    # Stimuli / stories
    stim_dir = root / "stimuli"
    wavs = sorted(stim_dir.glob("*_audio.wav")) if stim_dir.exists() else []
    stories = [w.name.replace("_audio.wav", "") for w in wavs]
    print(f"\n=== stimuli: {len(wavs)} *_audio.wav ===")
    for w in wavs[:50]:
        print("  ", w.name, w.stat().st_size, "B")

    # Subjects + func files
    subs = sorted([p.name for p in root.glob("sub-*") if p.is_dir()])
    print(f"\n=== {len(subs)} subjects ===")

    bold_by_sub: dict[str, list] = defaultdict(list)
    sample_sidecar = None
    sample_bold_stat = None
    for sub in subs:
        func = root / sub / "func"
        if not func.exists():
            continue
        for nii in sorted(func.glob("*_bold.nii.gz")):
            st = nii.stat()
            bold_by_sub[sub].append({"name": nii.name, "size": st.st_size})
            if sample_bold_stat is None:
                sample_bold_stat = {"name": nii.name, "size": st.st_size}
        if sample_sidecar is None:
            for js in sorted(func.glob("*_bold.json")):
                try:
                    sample_sidecar = {"name": js.name,
                                      "json": json.loads(js.read_text())}
                except Exception as e:
                    sample_sidecar = {"name": js.name, "error": str(e)}
                break

    print("\n=== bold files per subject ===")
    for sub in subs:
        files = bold_by_sub.get(sub, [])
        sizes = [f["size"] for f in files]
        print(f"  {sub}: {len(files)} bold; sizes {min(sizes) if sizes else 0}..{max(sizes) if sizes else 0} B")
        for f in files:
            print(f"      {f['name']}  {f['size']} B")

    # Try to discover task names from filenames
    import re
    task_re = re.compile(r"task-([A-Za-z0-9]+)")
    tasks_by_sub: dict[str, set] = defaultdict(set)
    for sub, files in bold_by_sub.items():
        for f in files:
            m = task_re.search(f["name"])
            if m:
                tasks_by_sub[sub].add(m.group(1))
    print("\n=== tasks per subject ===")
    for sub in subs:
        print(f"  {sub}: {sorted(tasks_by_sub.get(sub, []))}")

    # task -> subject coverage
    cov: dict[str, list] = defaultdict(list)
    for sub, tks in tasks_by_sub.items():
        for t in tks:
            cov[t].append(sub)
    print("\n=== task coverage (n subjects) ===")
    for t in sorted(cov, key=lambda k: -len(cov[k])):
        print(f"  {t}: {len(cov[t])} subjects")

    print("\n=== sample sidecar ===")
    print(json.dumps(sample_sidecar, indent=2)[:2000] if sample_sidecar else "NONE")

    return {
        "n_subjects": len(subs),
        "subjects": subs,
        "n_stories": len(stories),
        "stories": stories,
        "bold_counts": {s: len(bold_by_sub.get(s, [])) for s in subs},
        "tasks_by_sub": {s: sorted(tasks_by_sub.get(s, [])) for s in subs},
        "task_coverage": {t: sorted(v) for t, v in cov.items()},
        "sample_bold_stat": sample_bold_stat,
        "sample_sidecar": sample_sidecar,
    }


@app.local_entrypoint()
def main() -> None:
    import json
    res = inspect.remote()
    print("\n========== RESULT ==========")
    print(json.dumps(res, indent=2)[:6000])
