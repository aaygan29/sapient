"""Run the official fMRIPrep container on Modal for a small ds004996 subset.

Expert plan (see CODEBASE_MAP / handoff): the sapient-2 pipeline's
`prepare_fmri.py` consumes MNI152NLin2009cAsym *volumes* and projects them to
fsaverage5 with nilearn `vol_to_surf`. So we do NOT need FreeSurfer surface
recon. We run fMRIPrep with:

  --fs-no-reconall                      (skip the expensive recon-all)
  --output-spaces MNI152NLin2009cAsym:res-2
  --skip-bids-validation

which produces exactly `*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
per run — the file `find_bold_runs` looks for.

License gate: fMRIPrep needs a FreeSurfer license.txt even with
--fs-no-reconall. We get it from the Modal secret `freesurfer-license`
(env var FS_LICENSE_TEXT) and write it into the container at runtime.

ARCHITECTURAL RULE: BIDS in + derivatives out both live on the `sapient-data`
Modal volume. Nothing touches a laptop. Work dir is on fast local scratch
(NOT the volume) so fMRIPrep's heavy intermediate I/O stays off the network FS.

Modal app: sapient-2-fmriprep  (CPU, high-mem; ~1-3 h for 4 subjects).

Usage:
  modal run --detach data/run_fmriprep.py            # first 4 subjects
  modal run --detach data/run_fmriprep.py --subjects "01 02 03 04"
"""

from __future__ import annotations

import modal

APP_NAME = "sapient-2-fmriprep"
FMRIPREP_TAG = "25.2.5"

# Official nipreps image. We override the entrypoint so Modal can run our
# Python function inside it; we then call the `fmriprep` CLI via subprocess.
image = (
    modal.Image.from_registry(
        f"nipreps/fmriprep:{FMRIPREP_TAG}",
        add_python=None,  # the image already ships python in the fmriprep env
    )
    .entrypoint([])  # clear the fmriprep ENTRYPOINT so Modal's runner works
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
fs_secret = modal.Secret.from_name("freesurfer-license")

app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=8.0,
    memory=32768,          # fMRIPrep is RAM-hungry; ~8-16 GB/subject, run 1 at a time
    timeout=12 * 60 * 60,
    volumes={"/data": volume},
    secrets=[fs_secret],
    ephemeral_disk=512 * 1024,  # fast local scratch for the work dir (MB; Modal min 512 GiB)
)
def fmriprep_subject(subject_label: str) -> dict:
    """Run fMRIPrep on a single ds004996 subject (e.g. '01').

    One subject per container keeps peak RAM bounded and lets the 4-subject
    subset run as a parallel .map() fan-out instead of a 4x-long serial job.
    """
    import os
    import subprocess
    import time
    from pathlib import Path

    bids_dir = Path("/data/raw/ds004996")
    out_dir = Path("/data/derivatives/fmriprep")
    work_dir = Path("/scratch/work")  # ephemeral local disk
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- License gate: materialize the FreeSurfer license from the secret. ----
    fs_text = os.environ.get("FS_LICENSE_TEXT")
    if not fs_text:
        raise RuntimeError(
            "FS_LICENSE_TEXT not present — freesurfer-license secret missing/empty."
        )
    license_path = Path("/tmp/fs_license.txt")
    license_path.write_text(fs_text if fs_text.endswith("\n") else fs_text + "\n")

    cmd = [
        "fmriprep",
        str(bids_dir),
        str(out_dir),
        "participant",
        "--participant-label", subject_label,
        "--fs-no-reconall",
        "--output-spaces", "MNI152NLin2009cAsym:res-2",
        "--skip-bids-validation",
        "--fs-license-file", str(license_path),
        "--work-dir", str(work_dir),
        "--nprocs", "8",
        "--omp-nthreads", "4",
        "--mem-mb", "30000",
        "--stop-on-first-crash",
        "--notrack",
    ]
    print(f"==> sub-{subject_label}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    dt = time.time() - t0

    # Commit whatever derivatives exist regardless of exit code so partial
    # progress survives.
    volume.commit()

    sub_out = out_dir / f"sub-{subject_label}"
    preproc = sorted(sub_out.rglob("*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))
    result = {
        "subject": f"sub-{subject_label}",
        "returncode": proc.returncode,
        "minutes": round(dt / 60, 1),
        "n_preproc_bold": len(preproc),
        "preproc_bold": [str(p) for p in preproc],
    }
    print(f"<== sub-{subject_label} done: {result}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fMRIPrep failed for sub-{subject_label} (rc={proc.returncode}); "
            f"produced {len(preproc)} preproc BOLD files."
        )
    return result


@app.local_entrypoint()
def main(subjects: str = "01 02 03 04") -> None:
    """`modal run --detach data/run_fmriprep.py --subjects "01 02 03 04"`"""
    labels = subjects.split()
    print(f"Launching fMRIPrep for {len(labels)} subjects: {labels}")
    results = list(fmriprep_subject.map(labels, return_exceptions=True))
    print("\n==== fMRIPrep subset summary ====")
    total_runs = 0
    for lbl, r in zip(labels, results):
        if isinstance(r, Exception):
            print(f"  sub-{lbl}: ERROR {r!r}")
        else:
            total_runs += r["n_preproc_bold"]
            print(f"  sub-{lbl}: rc={r['returncode']} "
                  f"runs={r['n_preproc_bold']} ({r['minutes']} min)")
    print(f"Total preproc BOLD runs produced: {total_runs}")
