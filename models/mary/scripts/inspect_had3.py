"""Hunt for HAD event/design info anywhere on the volume + confirm bold geometry.

The fmriprep tree has NO *_events.tsv. HAD (ds004488) event timing might live:
  - in the raw BIDS root (a separate ds004488 raw, not just derivatives)
  - as *_events.tsv / *_events.json with a different task tag
  - in a sourcedata/ or design/ dir
  - encoded in confounds tsv (no) or a top-level participants/scans tsv
Also: load one bold to read affine/shape/TR and confirm it is volumetric MNI.
"""

from __future__ import annotations

import modal

app = modal.App("mary-inspect-had3")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.26,<3", "nibabel>=5.2"
)
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)


@app.function(image=image, cpu=2.0, memory=8192, timeout=20 * 60,
              volumes={"/data": volume})
def inspect() -> dict:
    from pathlib import Path
    import nibabel as nib

    data = Path("/data")
    out: dict = {}

    # 1) Top-level of /data/raw to see if there's a separate raw HAD tree
    print("=== /data/raw top-level ===")
    for p in sorted((data / "raw").iterdir()):
        print("  ", p.name, "(dir)" if p.is_dir() else "(file)")

    ds = data / "raw" / "ds004488"
    print("\n=== ds004488 FULL top-level (files+dirs) ===")
    for p in sorted(ds.iterdir()):
        kind = "dir" if p.is_dir() else f"{p.stat().st_size}B"
        print(f"  {p.name}  ({kind})")

    # 2) Any tsv/json/tsv.gz anywhere that could carry timing — look at names
    print("\n=== all *.tsv under ds004488 (names only, capped 80) ===")
    tsvs = []
    for p in ds.rglob("*.tsv"):
        rel = str(p.relative_to(ds))
        tsvs.append(rel)
    for r in sorted(tsvs)[:80]:
        print("  ", r)
    out["tsv_count"] = len(tsvs)
    # categorize
    kinds = {}
    for r in tsvs:
        base = r.split("/")[-1]
        key = base.split("_")[-1]
        kinds[key] = kinds.get(key, 0) + 1
    out["tsv_kinds"] = kinds
    print("  tsv kinds:", kinds)

    print("\n=== events/design/onset-ish files (any ext) ===")
    hits = []
    for p in ds.rglob("*"):
        n = p.name.lower()
        if any(k in n for k in ("event", "onset", "design", "stim", "trial", "log")):
            if p.is_file():
                hits.append((str(p.relative_to(ds)), p.stat().st_size))
    for h in hits[:60]:
        print("  ", h)
    out["timing_candidate_files"] = hits[:60]

    # 3) Inspect one stimuli category file listing as ordered (clip name carries
    #    youtube id + start; maybe run order is in a separate manifest)
    print("\n=== look for run/clip-order manifests (json/tsv/csv at root or code/) ===")
    for sub in ("", "code", "sourcedata", "stimuli"):
        d = ds / sub if sub else ds
        if not d.exists():
            continue
        for p in sorted(d.glob("*")):
            if p.is_file() and p.suffix.lower() in (".json", ".tsv", ".csv", ".txt", ".mat", ".m"):
                print(f"  {p.relative_to(ds)}  {p.stat().st_size}B")

    # 4) Confirm bold geometry
    bold = next(ds.rglob("sub-01_task-action_run-1_desc-preproc_bold.nii.gz"), None)
    if bold is not None:
        img = nib.load(str(bold))
        out["bold_shape"] = list(img.shape)
        out["bold_zooms"] = [float(z) for z in img.header.get_zooms()]
        out["bold_affine"] = [[float(x) for x in row] for row in img.affine]
        print("\n=== bold geometry ===")
        print("  shape:", img.shape)
        print("  zooms:", img.header.get_zooms())
        print("  affine:\n", img.affine)

    return out


@app.local_entrypoint()
def main() -> None:
    import json
    res = inspect.remote()
    print("\n========== RESULT ==========")
    print(json.dumps(res, indent=2)[:6000])
