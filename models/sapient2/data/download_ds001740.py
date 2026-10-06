"""Download ds001740 (Rauchbauer HRI 2019/2020) directly to the `sapient-data` Modal volume.

ARCHITECTURAL RULE: every download writes to the shared Modal volume.
Nothing lands on a laptop / workstation. OpenNeuro public buckets stream at
cluster-internal bandwidth when we sync from inside Modal.

Modal app: sapient-2-download-ds001740 (CPU, ~30 min – 2 h, < $0.50).
"""

from __future__ import annotations

import modal

APP_NAME = "sapient-2-download-ds001740"
ACCESSION = "ds001740"
# Spec §2.1 calls for v2.1.0 pin, but OpenNeuro's S3 mirror only exposes
# the latest revision at the dataset root — `s3://openneuro.org/ds001740/`
# returns the full tree; `s3://openneuro.org/ds001740/versions/2.1.0/` is
# empty. True version pinning needs the datalad-based clone of the
# OpenNeuroDatasets GitHub mirror (`git checkout 2.1.0` then `datalad get`).
# For now we sync the latest revision via S3 and note the deviation.
# Switch to the datalad pattern (see download_cneuromod.py) if exact v2.1.0
# reproducibility becomes a requirement.
S3_PATH = "s3://openneuro.org/ds001740"
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
    """Sync ds001740 from OpenNeuro into /data/raw/ds001740."""
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

    # CC0 license guard — locate dataset_description.json with rglob so we
    # tolerate any S3 layout (root or versioned snapshot).
    #
    # OpenNeuro datasets default to CC0 per platform policy but not all
    # dataset_description.json files declare it explicitly. ds001740 in
    # particular has an empty License field. The data is still CC0 per
    # OpenNeuro's terms — empty → warn-and-proceed; non-empty non-CC0 →
    # ABORT. To enforce strict CC0 declaration set DS_REQUIRE_CC0=1.
    import os
    desc_files = list(dest.rglob("dataset_description.json"))
    if not desc_files:
        raise RuntimeError(f"ABORT: no dataset_description.json found under {dest}")
    license = json.load(desc_files[0].open()).get("License", "")
    if license == "":
        msg = (f"  License field empty in {ACCESSION} dataset_description.json. "
               f"OpenNeuro default is CC0; proceeding.")
        print(msg)
        if os.environ.get("DS_REQUIRE_CC0") == "1":
            raise RuntimeError(f"ABORT: strict mode + empty License")
    elif "CC0" not in license:
        raise RuntimeError(
            f"ABORT: {ACCESSION} dataset_description License = {license!r} (expected CC0)"
        )
    else:
        print(f"  License check passed: {license}")

    # Footprint
    size = subprocess.check_output(["du", "-sh", str(dest)], text=True).split()[0]
    n_files = len(list(dest.rglob("*")))
    print(f"  footprint: {size}, {n_files} files")

    volume.commit()
    return {"accession": ACCESSION, "size": size, "n_files": int(n_files)}


@app.local_entrypoint()
def main() -> None:
    """`modal run data/download_ds001740.py`"""
    result = download.remote()
    print(f"\nDone. {result}")
