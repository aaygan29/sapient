"""One-shot diagnostic: can CNeuroMod fmriprep annex content actually be fetched?

Root-cause probe for the Sapient-1 "download completes but files are 147-byte
annex pointers" bug. We pick ONE small MNI BOLD file in friends/sub-01 and:
  1. `git annex whereis` it (which remotes claim to have it)
  2. enumerate the dataset's git-annex special remotes + their config
  3. try `datalad get` it with full annex debug, run from the *friends* dataset
  4. report the resulting on-disk size (>100 KB == real bytes landed)

CPU-only, single file, a couple minutes — cheap. Read-only except for the one
file it tries to fetch.
"""

from __future__ import annotations

import subprocess

import modal

app = modal.App("sapient-1-diag-annex")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-annex", "wget", "curl", "ca-certificates")
    .pip_install("datalad>=1.0,<2")
)

FRIENDS = "/data/raw/cneuromod/fmriprep/friends"


def sh(cmd, cwd=None, timeout=900):
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  !! TIMEOUT", flush=True)
        return None
    if r.stdout:
        print(r.stdout[-4000:], flush=True)
    if r.stderr:
        print("STDERR:", r.stderr[-4000:], flush=True)
    print(f"  rc={r.returncode}", flush=True)
    return r


@app.function(image=image, cpu=4.0, timeout=3600, volumes={"/data": vol})
def diag() -> dict:
    import glob
    import os

    sh(["git", "config", "--global", "user.email", "build@the-sapient-company.com"])
    sh(["git", "config", "--global", "user.name", "Sapient Build"])
    sh(["git", "config", "--global", "--add", "safe.directory", "*"])

    # 1. pick a small MNI BOLD file under friends/sub-01
    cands = sorted(glob.glob(
        f"{FRIENDS}/sub-01/**/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        recursive=True))
    if not cands:
        return {"error": "no candidate file found"}
    target = cands[0]
    rel = os.path.relpath(target, FRIENDS)
    print(f"TARGET: {rel}  (size now={os.path.getsize(target) if os.path.exists(target) else 'MISSING'})")

    # 2. enumerate special remotes
    sh(["git", "annex", "info", "--fast"], cwd=FRIENDS)
    sh(["git", "annex", "enableremote", "conp-ria-storage-http"], cwd=FRIENDS)

    # 3. whereis
    sh(["git", "annex", "whereis", rel], cwd=FRIENDS)

    # 4. try datalad get from the friends dataset directly
    sh(["datalad", "-C", FRIENDS, "get", "-J", "2", rel], timeout=1800)

    # 4b. fall back to raw git annex get with explicit http remote
    sh(["git", "annex", "get", "--from", "conp-ria-storage-http", rel],
       cwd=FRIENDS, timeout=1800)

    # check the annex object actually exists on the volume (the real persistence test)
    sh(["git", "annex", "whereis", rel], cwd=FRIENDS)
    obj = subprocess.run(
        ["du", "-sh", f"{FRIENDS}/.git/annex/objects"],
        capture_output=True, text=True).stdout.strip()
    print(f"ANNEX OBJECTS DIR: {obj}")

    size = os.path.getsize(target) if os.path.exists(target) else 0
    landed = size > 100_000
    vol.commit()
    print(f"\n=== RESULT: size={size} bytes  landed={landed} ===")
    return {"target": rel, "size_bytes": size, "real_content": landed}


@app.local_entrypoint()
def main() -> None:
    print(diag.remote())
