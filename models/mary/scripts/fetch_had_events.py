"""Fetch HAD (ds004488) raw BIDS *_events.tsv (+ events.json) for sub-01/sub-02
from public OpenNeuro S3 onto the sapient-data volume, then print their format.

Only the derivatives/fmriprep tree was synced earlier (no events). The clip<->onset
alignment lives in the raw BIDS func/ events.tsv. These files are tiny (KBs).

Lands at: /data/raw/ds004488/sub-01/func/*_task-action_*_events.tsv (+ .json)

Run:  modal run scripts/fetch_had_events.py
"""

from __future__ import annotations

import subprocess
import modal

app = modal.App("mary-fetch-had-events")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").apt_install("awscli")

S3 = "s3://openneuro.org/ds004488"
DEST = "/data/raw/ds004488"
SUBS = ["sub-01", "sub-02"]


def _sh(cmd):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout[-2000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1000:])
    return r


@app.function(image=image, cpu=2.0, timeout=30 * 60, volumes={"/data": vol})
def fetch() -> dict:
    from pathlib import Path

    out: dict = {"synced": {}}

    # top-level events sidecar if present (task-action_events.json)
    _sh(["aws", "s3", "cp", "--no-sign-request",
         f"{S3}/task-action_events.json", f"{DEST}/task-action_events.json"])
    _sh(["aws", "s3", "cp", "--no-sign-request",
         f"{S3}/task-action_bold.json", f"{DEST}/task-action_bold.json"])

    # Raw BIDS HAD nests runs under ses-action01/func/. Sync only the tiny
    # events.tsv from there (keep BIDS layout intact on the volume).
    SES = "ses-action01"
    for sub in SUBS:
        src = f"{S3}/{sub}/{SES}/func/"
        dst = f"{DEST}/{sub}/{SES}/func/"
        r = _sh(["aws", "s3", "sync", "--no-sign-request", src, dst,
                 "--exclude", "*", "--include", "*_events.tsv"])
        out["synced"][f"{sub}"] = r.returncode == 0
        vol.commit()

    # Report what landed + show one events.tsv in full-ish
    print("\n===== events.tsv landed =====")
    sample = None
    counts = {}
    for sub in SUBS:
        func = Path(DEST) / sub / SES / "func"
        evs = sorted(func.glob("*_events.tsv")) if func.exists() else []
        counts[sub] = [e.name for e in evs]
        print(f"  {sub}: {len(evs)} events.tsv")
        for e in evs:
            print("     ", e.name, e.stat().st_size, "B")
        if evs and sample is None:
            sample = evs[0]
    out["events_files"] = counts

    if sample is not None:
        lines = sample.read_text().splitlines()
        print(f"\n===== FULL sample {sample.name} ({len(lines)-1} rows) =====")
        print("HEADER:", lines[0])
        for r in lines[1:25]:
            print("  ", r)
        print("  ...")
        for r in lines[-5:]:
            print("  ", r)
        out["sample_name"] = sample.name
        out["sample_header"] = lines[0]
        out["sample_first25"] = lines[1:25]
        out["sample_last5"] = lines[-5:]
        out["sample_nrows"] = len(lines) - 1

    # events.json sidecar contents (column definitions)
    ej = Path(DEST) / "task-action_events.json"
    if ej.exists():
        out["events_json"] = ej.read_text()[:3000]
        print("\n===== task-action_events.json =====")
        print(out["events_json"])

    return out


@app.local_entrypoint()
def main() -> None:
    import json
    res = fetch.remote()
    print("\n========== RESULT ==========")
    print(json.dumps(res, indent=2)[:5000])
