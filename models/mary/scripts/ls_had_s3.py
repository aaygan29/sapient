"""List the raw ds004488 S3 tree for sub-01 to locate events.tsv (sessions?)."""
from __future__ import annotations
import subprocess
import modal

app = modal.App("mary-ls-had-s3")
image = modal.Image.debian_slim(python_version="3.11").apt_install("awscli")
S3 = "s3://openneuro.org/ds004488"


def _sh(cmd):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print("ERR", r.stderr[-500:])
    return r.stdout


@app.function(image=image, cpu=2.0, timeout=15 * 60)
def ls() -> dict:
    print("=== top level ds004488 ===")
    _sh(["aws", "s3", "ls", "--no-sign-request", f"{S3}/"])
    print("\n=== sub-01/ ===")
    _sh(["aws", "s3", "ls", "--no-sign-request", f"{S3}/sub-01/"])
    print("\n=== recursive sub-01 (events/tsv/json only via grep-ish full list) ===")
    out = _sh(["aws", "s3", "ls", "--no-sign-request", "--recursive", f"{S3}/sub-01/"])
    events = [l for l in out.splitlines() if "events" in l.lower()]
    print("\n=== events lines ===")
    for e in events[:40]:
        print("  ", e)
    return {"events_lines": events[:40], "n_events_lines": len(events)}


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(ls.remote(), indent=2)[:4000])
