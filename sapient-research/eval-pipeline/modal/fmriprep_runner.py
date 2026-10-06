"""Modal app: sapienteval-fmriprep — run fMRIPrep on NeuroEngage BIDS subjects.

Purpose
-------
Preprocess raw BIDS functional MRI from OpenNeuro ds004996 (NeuroEngage) so
downstream stages get motion-corrected BOLD in MNI152 volumetric space, ready
for Schaefer-400 parcellation and Sapient Cognitive Eval (Track B) fine-tuning.

Dataset
-------
OpenNeuro ds004996 (NeuroEngage). The raw BIDS tree is expected at
`/data/ds004996/` inside the Modal volume (populated by
`sapienteval/01_download_neuroengage.py` running locally or via a separate
upload step; this runner does not re-download).

Modal isolation
---------------
- App name:   `sapienteval-fmriprep`  (kept separate from `sapient-v2` and
              `sapient-scan-trigger`; do not consolidate).
- Volume:     `sapienteval-data`      (raw BIDS + derivatives; isolated from
              the video-scan pipeline; re-downloaded fresh, not shared).
- Workspace:  robert-16572            (existing user workspace).

Why no FreeSurfer
-----------------
Schaefer-400 parcellation operates on volumetric MNI152 BOLD, so cortical
surface reconstruction is unnecessary and would dominate runtime (~hours of
recon-all per subject). We pass `--fs-no-reconall` and restrict outputs to
`--output-spaces MNI152NLin2009cAsym:res-2`. Pure volumetric pipeline.

Container
---------
Base image is `nipreps/fmriprep:24.1.1` from Docker Hub. fMRIPrep is already
installed inside the image; `add_python="3.11"` makes the image compatible
with Modal's function dispatch (Modal needs a Python interpreter on PATH to
launch the function entrypoint).

Resources per subject
---------------------
- 8 vCPU, 16 GB RAM
- 6-hour hard timeout (fMRIPrep itself capped at 5.5h via `subprocess` timeout
  so we get a clean error tail before Modal kills the container)
- ~$1/hr × ~4h × 5 subjects run in parallel ≈ ~$20 for the pilot

Deploy
------
    modal deploy sapienteval/modal/fmriprep_runner.py

Run (pilot defaults: sub-01..sub-05 in parallel)
------------------------------------------------
    modal run sapienteval/modal/fmriprep_runner.py
    modal run sapienteval/modal/fmriprep_runner.py --subjects sub-01,sub-02

Inspect outputs
---------------
    modal volume ls sapienteval-data
    modal volume get sapienteval-data /derivatives/fmriprep/sub-01/func/ ./local-inspection/
"""

from __future__ import annotations

import json

import modal

# --------------------------------------------------------------------------- #
# Image: fMRIPrep 24.1.1 from Docker Hub, with Python 3.11 added so Modal can
# launch the function entrypoint inside the container.
# --------------------------------------------------------------------------- #
image = (
    modal.Image.from_registry(
        "nipreps/fmriprep:24.1.1",
        add_python="3.11",
    )
    # The nipreps image sets ENTRYPOINT ["fmriprep"] which collides with
    # Modal's own runner dispatch (the runner needs to exec python). Clearing
    # the entrypoint lets Modal launch its dispatcher; we invoke fmriprep
    # explicitly via subprocess.run(["fmriprep", ...]) below.
    .entrypoint([])
)

# --------------------------------------------------------------------------- #
# Volume: holds the raw BIDS dataset and the derivatives we write.
# Isolated from any other Sapient pipeline.
# --------------------------------------------------------------------------- #
data_vol = modal.Volume.from_name("sapienteval-data", create_if_missing=True)

# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = modal.App("sapienteval-fmriprep", image=image)


@app.function(
    cpu=8,                       # 8 vCPU
    memory=16384,                # 16 GB RAM
    timeout=6 * 3600,            # 6h hard timeout (Modal will kill at 6h)
    volumes={"/data": data_vol},
    # FreeSurfer license: required by fmriprep even with --fs-no-reconall.
    # The Modal secret `freesurfer-license` exposes FS_LICENSE_TEXT (the
    # text contents of the user's license.txt). We write it to a file at
    # request time and pass via --fs-license-file.
    secrets=[modal.Secret.from_name("freesurfer-license")],
)
def preprocess_subject(subject_id: str) -> dict:
    """Run fMRIPrep on one BIDS subject.

    Reads from  : /data/ds004996/sub-XX/
    Writes to   : /data/derivatives/fmriprep/sub-XX/
    Returns     : {subject_id, status, duration_s, n_runs_processed, error?}

    `subject_id` may be passed as "sub-01" or just "01"; both are normalized.
    If `desc-preproc_bold.nii.gz` files already exist for the subject, the
    function returns `status="skipped_already_processed"` without re-running.
    """
    import os
    import subprocess
    import time

    bids_root = "/data/ds004996"
    deriv_root = "/data/derivatives"
    os.makedirs(deriv_root, exist_ok=True)

    label = subject_id.replace("sub-", "")
    expected_out = f"{deriv_root}/fmriprep/sub-{label}/func"

    # Skip if already processed (idempotent re-runs)
    if os.path.isdir(expected_out) and any(
        f.endswith("desc-preproc_bold.nii.gz") for f in os.listdir(expected_out)
    ):
        return {
            "subject_id": subject_id,
            "status": "skipped_already_processed",
        }

    # Materialize the FreeSurfer license from the Modal secret onto disk so
    # fmriprep can read it. The secret exposes either FS_LICENSE_TEXT (full
    # license body) or FREESURFER_LICENSE (legacy name); we accept either.
    license_text = os.environ.get("FS_LICENSE_TEXT") or os.environ.get("FREESURFER_LICENSE", "")
    if not license_text:
        return {
            "subject_id": subject_id,
            "status": "error",
            "error": "FreeSurfer license missing. Set Modal secret `freesurfer-license` with key FS_LICENSE_TEXT.",
        }
    license_path = "/tmp/fs_license.txt"
    with open(license_path, "w") as f:
        f.write(license_text.strip() + "\n")

    t0 = time.time()
    cmd = [
        "fmriprep",
        bids_root,
        deriv_root,
        "participant",
        "--participant-label", label,
        "--fs-license-file", license_path,
        "--fs-no-reconall",                              # no cortical surface recon
        "--output-spaces", "MNI152NLin2009cAsym:res-2",  # volumetric MNI only
        "--skip-bids-validation",
        "--nthreads", "8",
        "--mem-mb", "14000",
        "--stop-on-first-crash",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(5.5 * 3600),  # leave 30 min headroom before Modal's 6h kill
        )
    except subprocess.TimeoutExpired as e:
        duration_s = time.time() - t0
        return {
            "subject_id": subject_id,
            "status": "timeout",
            "duration_s": duration_s,
            "stderr_tail": (e.stderr or b"")[-2000:].decode("utf-8", errors="replace")
            if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")[-2000:],
            "stdout_tail": (e.stdout or b"")[-2000:].decode("utf-8", errors="replace")
            if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")[-2000:],
        }

    duration_s = time.time() - t0

    if result.returncode != 0:
        return {
            "subject_id": subject_id,
            "status": "error",
            "duration_s": duration_s,
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-2000:],
            "stdout_tail": result.stdout[-2000:],
        }

    # Count preprocessed BOLD runs that landed on disk
    n_runs = 0
    if os.path.isdir(expected_out):
        n_runs = sum(
            1 for f in os.listdir(expected_out) if f.endswith("desc-preproc_bold.nii.gz")
        )

    # Persist writes to the volume so subsequent stages can read them
    data_vol.commit()

    return {
        "subject_id": subject_id,
        "status": "ok",
        "duration_s": duration_s,
        "n_runs_processed": n_runs,
    }


@app.local_entrypoint()
def main(
    subjects: str = "sub-01,sub-02,sub-03,sub-04,sub-05",
    wait: bool = False,
):
    """Submit one fMRIPrep container per subject, fire-and-forget by default.

    Containers run independently on Modal — they survive after this CLI exits.
    Use `--wait` to block until all complete (only useful for short subject
    lists; for 5+ subjects you almost always want fire-and-forget).

    Side effect: writes derivatives to /data/derivatives/fmriprep/sub-XX/ on
    the sapienteval-data volume. Check progress with `modal app list`,
    `modal app history sapienteval-fmriprep`, or by listing the volume.

    Usage
    -----
        modal run sapienteval/modal/fmriprep_runner.py
        modal run sapienteval/modal/fmriprep_runner.py --subjects sub-01,sub-02
        modal run sapienteval/modal/fmriprep_runner.py --wait              # block
    """
    subject_list = [s.strip() for s in subjects.split(",") if s.strip()]

    if wait:
        # Synchronous map — local CLI must stay connected for ~4h
        print(
            f"Submitting {len(subject_list)} fMRIPrep jobs in parallel (BLOCKING): {subject_list}"
        )
        results = list(preprocess_subject.map(subject_list))
        print(json.dumps(results, indent=2))
        return

    # Fire-and-forget: each subject becomes an independent FunctionCall that
    # survives after this entrypoint exits. Modal's `.map()` over a detached
    # app can be cancelled when the local caller disconnects, so we use
    # `.spawn()` per subject instead.
    print(
        f"Spawning {len(subject_list)} independent fMRIPrep containers..."
    )
    call_ids = []
    for sid in subject_list:
        call = preprocess_subject.spawn(sid)
        call_ids.append({"subject_id": sid, "call_id": call.object_id})
        print(f"  {sid:8s} -> call_id={call.object_id}")
    print()
    print(json.dumps({"spawned": call_ids}, indent=2))
    print()
    print("Containers are running independently. They will finish in ~4h.")
    print("Check progress:")
    print("  modal app history sapienteval-fmriprep")
    print("  modal volume ls sapienteval-data /derivatives/fmriprep/")
