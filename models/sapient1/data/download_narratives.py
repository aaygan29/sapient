"""Download ds002345 (Narratives, Nastase et al. 2021) directly to the `sapient-data` Modal volume.

ARCHITECTURAL RULE: every download writes to the shared Modal volume.
Nothing lands on a laptop / workstation. OpenNeuro public buckets stream at
cluster-internal bandwidth when we sync from inside Modal.

Modal app: sapient-1-download-narratives (CPU, ~30 min – 2 h, < $0.50).
"""

from __future__ import annotations

import modal

APP_NAME = "sapient-1-download-narratives"
ACCESSION = "ds002345"
S3_PATH = "s3://openneuro.org/ds002345"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("awscli", "ca-certificates")
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=6 * 60 * 60,
    volumes={"/data": volume},
)
def download() -> dict:
    """Sync ds002345 from OpenNeuro into /data/raw/ds002345."""
    import json
    import subprocess
    from pathlib import Path

    dest = Path("/data/raw") / ACCESSION
    dest.mkdir(parents=True, exist_ok=True)

    print(f"==> aws s3 sync --no-sign-request {S3_PATH} {dest}")
    subprocess.run(
        ["aws", "s3", "sync", "--no-sign-request", S3_PATH, str(dest)],
        check=True,
    )

    # Narratives is the documented non-CC0 exception per spec §3:
    # fMRI is openly shared, stimulus audio is copyrighted (radio shows / published
    # readings). We use audio features-only via frozen encoders and DO NOT
    # redistribute raw audio with our model release. release.py is responsible for
    # excluding the audio from release_artifacts/. So this script just logs the
    # license rather than aborting.
    desc_files = list(dest.rglob("dataset_description.json"))
    if not desc_files:
        raise RuntimeError(f"ABORT: no dataset_description.json found under {dest}")
    license = json.load(desc_files[0].open()).get("License", "")
    print(f"  dataset_description.json License: {license!r}")
    print(f"  (using audio as frozen features only; will not redistribute)")

    # Footprint
    size = subprocess.check_output(["du", "-sh", str(dest)], text=True).split()[0]
    n_files = len(list(dest.rglob("*")))
    print(f"  footprint: {size}, {n_files} files")

    volume.commit()
    return {"accession": ACCESSION, "size": size, "n_files": int(n_files)}


@app.local_entrypoint()
def main() -> None:
    """`modal run data/download_narratives.py`"""
    result = download.remote()
    print(f"\nDone. {result}")
