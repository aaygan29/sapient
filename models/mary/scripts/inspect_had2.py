"""Deep walk of HAD sub-01/sub-02 trees (raw + fmriprep) to locate bold + events."""

from __future__ import annotations

import modal

app = modal.App("mary-inspect-had2")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26,<3")
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)

RAW = "/data/raw/ds004488"


@app.function(image=image, cpu=2.0, memory=4096, timeout=20 * 60,
              volumes={"/data": volume})
def inspect() -> dict:
    import json
    from pathlib import Path

    root = Path(RAW)
    out: dict = {}

    # Full recursive listing of the two subjects in BOTH raw and derivatives.
    targets = [
        root / "sub-01",
        root / "sub-02",
        root / "derivatives" / "fmriprep" / "sub-01",
        root / "derivatives" / "fmriprep" / "sub-02",
    ]
    for t in targets:
        print(f"\n===== walk {t} (exists={t.exists()}) =====")
        if not t.exists():
            continue
        entries = []
        for p in sorted(t.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(t))
                sz = p.stat().st_size
                entries.append((rel, sz))
                print(f"  {rel}  {sz} B")
        out[str(t)] = entries[:200]

    # Look for ANY events.tsv anywhere under the dataset (capped)
    print("\n===== any *_events.tsv under ds004488 =====")
    evs = []
    for p in root.rglob("*_events.tsv"):
        evs.append(str(p.relative_to(root)))
        if len(evs) <= 20:
            print("  ", p.relative_to(root))
    out["all_events_count"] = len(evs)
    out["events_sample"] = evs[:20]

    # Print one events.tsv header + rows if present
    if evs:
        ev0 = root / evs[0]
        lines = ev0.read_text().splitlines()
        print(f"\n===== {evs[0]} =====")
        print("  header:", lines[0] if lines else "")
        for r in lines[1:8]:
            print("  row:", r)
        out["events_header"] = lines[0] if lines else ""
        out["events_rows"] = lines[1:8]
        out["events_nrows"] = len(lines) - 1

    # Find one bold + sidecar
    print("\n===== any *bold.nii.gz (first 20) =====")
    bolds = []
    for p in root.rglob("*bold.nii.gz"):
        bolds.append(str(p.relative_to(root)))
        if len(bolds) <= 20:
            print("  ", p.relative_to(root), p.stat().st_size, "B")
    out["bold_count"] = len(bolds)
    out["bold_sample"] = bolds[:20]

    # sidecar
    for p in root.rglob("*task-action*_bold.json"):
        try:
            out["sidecar"] = {"name": p.name, "json": json.loads(p.read_text())}
        except Exception as e:
            out["sidecar"] = {"name": p.name, "error": str(e)}
        print("\n===== sidecar", p.name, "=====")
        print(json.dumps(out["sidecar"], indent=2)[:1500])
        break

    return out


@app.local_entrypoint()
def main() -> None:
    import json
    res = inspect.remote()
    print("\n========== RESULT (truncated) ==========")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("/")},
                     indent=2)[:6000])
