"""Download CNeuroMod CC0 subset directly to the `sapient-data` Modal volume.

ARCHITECTURAL RULE: every download writes to the shared Modal volume.
Nothing lands on a laptop / workstation. CNeuroMod alone is hundreds of GB;
running datalad locally is a non-starter.

This script runs as a Modal CPU job (no GPU needed). datalad + git-annex
fetch from CONP/GitHub into /data/raw/cneuromod inside the container,
volume.commit() persists it, and every downstream Modal app (feature
extraction, training, eval) reads from the same volume.

ROOT-CAUSE NOTES (2026-06-07)
-----------------------------
The original version of this script had two independent failure modes that
together meant it "completed" while fetching ZERO bytes:

  1. SILENT NO-OP (`get (notneeded: 2)`). It ran
        datalad get -r fmriprep/friends/sub-01
     from the *cneuromod superdataset*. git-annex content for these
     fmriprep datasets lives on a separate special remote
     (`conp-ria-storage-http`, the public CONP RIA HTTP store) that is NOT
     auto-enabled, and the recursive get from the superdataset only
     "ensured presence" of the parent datasets without descending to fetch
     annexed content — so every *_bold.nii.gz stayed a ~149-byte annex
     pointer. FIX: enable `conp-ria-storage-http` on each leaf dataset and
     run `datalad get` from *inside* that leaf dataset (-C).

  2. PREEMPTION. The @app.function had no `retries`, so a single Modal
     worker preemption ("Runner interrupted due to worker preemption")
     killed the whole multi-hour job with no resume. FIX: retries=10 +
     per-(task,subject) chunked gets that are individually short and
     idempotent (annex skips content already present), so a restart picks
     up exactly where it left off.

Also note: git identity MUST be set at RUNTIME. git-annex shells out to
`git commit-tree` while recording state; without user.name/user.email it
fails with "Author identity unknown" and the get aborts.

Modal app: sapient-1-download-cneuromod (CPU, retried, resumable).
"""

from __future__ import annotations

import modal

APP_NAME = "sapient-1-download-cneuromod"
REPO = "https://github.com/courtois-neuromod/cneuromod.processed.git"
CC0_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
TASKS = ["friends", "movie10"]
# Public CONP RIA HTTP store that actually serves the annex content without
# CNeuroMod S3 credentials. Confirmed via `git annex whereis` (uuid ce44986a).
PUBLIC_ANNEX_REMOTE = "conp-ria-storage-http"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "curl", "ca-certificates",
        "build-essential", "python3-dev",
        "sudo",   # datalad-installer invokes `sudo` even when running as root
    )
    .pip_install("datalad>=1.0,<2", "datalad-installer>=1.1")
    # apt's git-annex on Debian Slim lags well behind the version datalad
    # 1.x expects, which causes opaque `datalad install` failures.
    # datalad-installer pulls a known-working binary from the
    # datalad/git-annex release builds and drops it in /usr/local/bin.
    .run_commands(
        # git annex commits during `datalad install`; commits need identity.
        "git config --global user.email 'build@the-sapient-company.com'",
        "git config --global user.name 'Sapient Build'",
        "datalad-installer --sudo ok git-annex -m datalad/git-annex:release",
    )
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=24 * 60 * 60,
    # PREEMPTION RESILIENCE: a preempted worker restarts with the same input;
    # the per-chunk gets below are idempotent so it resumes cleanly.
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0,
                          initial_delay=10.0),
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def download() -> dict:
    """Pull CNeuroMod CC0 subjects' fMRI + stimuli into /data/raw/cneuromod/."""
    import subprocess
    import sys
    from pathlib import Path

    raw_root = Path("/data/raw")
    raw_root.mkdir(parents=True, exist_ok=True)
    dest = raw_root / "cneuromod"

    def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
        """Run a subprocess and capture/log stderr loudly. Modal's default
        propagation drops stderr; we route it through stdout so it lands in
        the Modal logs and the run dashboard."""
        print(f"  $ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout[-4000:], flush=True)
        if result.stderr:
            print(f"  [stderr] {result.stderr[-4000:]}", file=sys.stderr, flush=True)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"rc={result.returncode}: {' '.join(cmd)}\nSTDERR: {result.stderr}"
            )
        return result.returncode

    # RUNTIME git identity — git-annex commit-tree needs it, and --global from
    # image build does not always survive into the function's HOME.
    run(["git", "config", "--global", "user.email",
         "build@the-sapient-company.com"], check=False)
    run(["git", "config", "--global", "user.name", "Sapient Build"], check=False)
    run(["git", "config", "--global", "--add", "safe.directory", "*"], check=False)

    # 0. Tool versions — surfaces version skew loudly.
    print("=== Tool versions ===")
    run(["git", "--version"], check=False)
    run(["git-annex", "version"], check=False)
    run(["datalad", "--version"], check=False)
    print()

    # 1. Install the superdataset (metadata only, fast). Idempotent.
    if not dest.exists():
        run(["datalad", "install", "--source", REPO, str(dest)])
    else:
        print(f"  (cneuromod already installed at {dest})")

    # 2. Install the fmriprep submodule's metadata (-n = no data, just the tree).
    run(["datalad", "get", "-n", "fmriprep"], cwd=str(dest))

    def enable_public_remote(ds_dir: Path) -> None:
        """Make the public CONP RIA HTTP store usable for annex get.
        Without this, annex only knows about the credentialed S3 remote and
        the laptop-scratch peers (offline), so get silently no-ops."""
        run(["git", "annex", "enableremote", PUBLIC_ANNEX_REMOTE],
            cwd=str(ds_dir), check=False)

    def get_leaf(ds_dir: Path, label: str) -> str:
        """Fetch annex content for a fully-installed leaf dataset, from inside
        it. Idempotent: annex skips content already present, so a preemption
        restart resumes here cheaply."""
        enable_public_remote(ds_dir)
        rc = run(["datalad", "-C", str(ds_dir), "get", "-r", "-J", "6", "."],
                 check=False)
        return "ok" if rc == 0 else f"failed: rc={rc}"

    # 3. Per-(task, subject) recursive get. Each is a short, resumable chunk.
    #    -n first installs the leaf subject subdataset tree, THEN we get content
    #    from inside it so annex actually descends and fetches bytes.
    summary: dict = {}
    for task in TASKS:
        task_dir = dest / "fmriprep" / task
        # ensure the task dataset tree is installed (metadata)
        run(["datalad", "get", "-n", "-r", f"fmriprep/{task}"],
            cwd=str(dest), check=False)
        enable_public_remote(task_dir)
        for sub in CC0_SUBJECTS:
            sub_dir = task_dir / sub
            if not sub_dir.exists():
                print(f"  (skip {task}/{sub}: not in this task)")
                continue
            print(f"\n=== get {task}/{sub} ===")
            summary[f"{task}/{sub}"] = get_leaf(sub_dir, f"{task}/{sub}")
            volume.commit()   # persist after each subject so progress survives

    # 4. Stimuli (videos + transcripts) — live under sourcedata/<task>.
    for task in TASKS:
        stim_dir = dest / "fmriprep" / task / "sourcedata" / task
        run(["datalad", "get", "-n", "-r",
             f"fmriprep/{task}/sourcedata/{task}"], cwd=str(dest), check=False)
        if stim_dir.exists():
            print(f"\n=== get stimuli {task} ===")
            summary[f"stimuli/{task}"] = get_leaf(stim_dir, f"stimuli/{task}")
            volume.commit()

    # 5. Footprint report — measures REAL bytes (annex object store), not the
    #    symlink farm. A subject with content shows hundreds of MB / GB here.
    print("\n=== Real footprint (annex objects) by task ===")
    footprint: dict = {}
    for task in TASKS:
        objs = dest / "fmriprep" / task / ".git" / "annex" / "objects"
        if objs.exists():
            size = subprocess.check_output(["du", "-sh", str(objs)],
                                           text=True).split()[0]
            footprint[task] = size
            print(f"  {task}: {size} of annexed content")

    volume.commit()
    return {"summary": summary, "footprint": footprint}


@app.local_entrypoint()
def main() -> None:
    """`modal run --detach data/download_cneuromod.py`"""
    result = download.remote()
    print("\n=== Download summary ===")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print("\n=== Real footprint ===")
    for k, v in result.get("footprint", {}).items():
        print(f"  {k}: {v}")
