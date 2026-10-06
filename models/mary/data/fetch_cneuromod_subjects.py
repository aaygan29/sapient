"""Materialize CNeuroMod movie10 BOLD for additional subjects (sub-03, sub-05)
via git-annex from the PUBLIC conp-ria / s3 remotes (no DUA token needed —
verified by probe: `git annex get` of a sub-03 BOLD succeeded over
conp-ria-storage-http).

The fmriprep derivatives live in the datalad repo at
/data/raw/cneuromod/fmriprep/movie10 (git-annex). Their BOLD files are dangling
symlinks (annex pointers) until pulled. This app runs `git annex get` for each
sub-XX movie10 `*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`, which
downloads the real bytes into .git/annex/objects and makes the symlink resolve.

Idempotent: git-annex skips files already present. Git identity is set so the
post-get state commit doesn't error (the probe's only failure was a cosmetic
"Author identity unknown" on the state commit — the download itself was fine).

`modal run --detach data/fetch_cneuromod_subjects.py --subjects sub-03,sub-05`
`modal run data/fetch_cneuromod_subjects.py::report --subjects sub-03,sub-05`
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "git", "git-annex", "openssh-client", "curl", "wget", "ca-certificates"
)
app = modal.App("mary-fetch-cneuromod-subjects")

REPO = "/data/raw/cneuromod/fmriprep/movie10"
CN_SPACE = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"


def _materialized(p: Path) -> bool:
    try:
        return p.stat().st_size > 1_000_000
    except OSError:
        return False


@app.function(image=image, cpu=2, memory=4096, timeout=12 * 3600,
              volumes={"/data": vol})
def fetch(subjects: str = "sub-03,sub-05") -> dict:
    subs = [s.strip() for s in subjects.split(",") if s.strip()]
    repo = Path(REPO)
    # Set a git identity so annex's post-get state commit doesn't abort.
    subprocess.run(["git", "config", "user.email", "mary@sapient.local"],
                   cwd=repo, check=False)
    subprocess.run(["git", "config", "user.name", "mary-data"],
                   cwd=repo, check=False)

    results = {}
    for sub in subs:
        sdir = repo / sub
        bolds = sorted(sdir.rglob(f"*_{CN_SPACE}"))
        todo = [b for b in bolds if not _materialized(b)]
        print(f"[{sub}] {len(bolds)} BOLD files, {len(todo)} to fetch", flush=True)
        got, failed = 0, []
        t0 = time.time()
        for i, b in enumerate(todo):
            rel = str(b.relative_to(repo))
            r = subprocess.run(["git", "annex", "get", rel],
                               cwd=repo, capture_output=True, text=True)
            ok = _materialized(b)
            if ok:
                got += 1
            else:
                failed.append({"file": rel, "rc": r.returncode,
                               "err": (r.stdout + r.stderr)[-300:]})
            if (i + 1) % 5 == 0 or not ok:
                print(f"  [{sub}] {i+1}/{len(todo)} got={got} "
                      f"({rel.split('/')[-1]}) ok={ok}", flush=True)
                vol.commit()
        vol.commit()
        n_now = sum(1 for b in bolds if _materialized(b))
        results[sub] = {"n_bold": len(bolds), "fetched_this_run": got,
                        "n_materialized_now": n_now, "n_failed": len(failed),
                        "failed_sample": failed[:5],
                        "secs": round(time.time() - t0, 1)}
        print(f"[{sub}] DONE materialized={n_now}/{len(bolds)} "
              f"in {results[sub]['secs']}s", flush=True)
    return results


@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60,
              volumes={"/data": vol})
def report(subjects: str = "sub-03,sub-05") -> dict:
    import re
    subs = [s.strip() for s in subjects.split(",") if s.strip()]
    repo = Path(REPO)
    out = {}
    for sub in subs:
        bolds = sorted((repo / sub).rglob(f"*_{CN_SPACE}"))
        mat = [b for b in bolds if _materialized(b)]
        tasks = sorted({re.search(r"task-([A-Za-z0-9]+)", b.name).group(1)
                        for b in mat if re.search(r"task-([A-Za-z0-9]+)", b.name)})
        out[sub] = {"n_bold": len(bolds), "n_materialized": len(mat),
                    "n_tasks_materialized": len(tasks), "tasks": tasks}
    return out


@app.local_entrypoint()
def main(subjects: str = "sub-03,sub-05"):
    # Blocking .remote() under `modal run --detach` keeps this app alive
    # server-side after the local CLI disconnects (survives laptop sleep).
    print(json.dumps(fetch.remote(subjects), indent=2))


@app.local_entrypoint()
def report_ep(subjects: str = "sub-03,sub-05"):
    print(json.dumps(report.remote(subjects), indent=2))
