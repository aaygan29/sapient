"""Fetch the CC0 Algonauts-2025 (CNeuroMod) MOVIE STIMULI via DataLad to the volume.

We only need the stimuli (.mkv + .tsv transcripts) — the fMRI we already have as
CNeuroMod fMRIPrep on the volume (which we project to fsaverage5). Repo:
  github.com/courtois-neuromod/algonauts_2025.competitors  (CC0)

Pulls a chosen subset of stimuli/movies/{movie10,ood,friends/sX}. movie10 includes
the Bourne Ultimatum (Paper B OOD) and matches our cneuromod fmriprep tasks.

Run DETACHED so it survives a laptop crash/sleep:
  modal run --detach data/fetch_algonauts_stimuli.py --subset movie10,ood
"""
import subprocess
import modal

app = modal.App("mary-fetch-algonauts")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "git-annex", "wget")
         .pip_install("datalad>=1.0"))
REPO = "https://github.com/courtois-neuromod/algonauts_2025.competitors.git"


def _sh(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.stdout: print(r.stdout[-2000:], flush=True)
    if r.returncode != 0: print("STDERR:", r.stderr[-2000:], flush=True)
    return r


@app.function(image=image, cpu=8.0, timeout=12 * 60 * 60, volumes={"/data": vol})
def fetch(subset: str = "movie10,ood"):
    root = "/data/raw/algonauts2025"
    # datalad needs git identity + annex
    _sh(["git", "config", "--global", "user.email", "mary@sapient.local"])
    _sh(["git", "config", "--global", "user.name", "mary"])
    _sh(["datalad", "install", "-r", "-s", REPO, root])
    got = {}
    for part in [s for s in subset.split(",") if s]:
        path = f"{root}/stimuli/movies/{part}"
        r = _sh(["datalad", "get", "-r", "-J", "8", path], cwd=root)
        got[part] = r.returncode == 0
        vol.commit()
    # report what landed
    r = _sh(["bash", "-lc", f"find {root}/stimuli -name '*.mkv' | head -40 ; "
             f"echo '---count---' ; find {root}/stimuli -name '*.mkv' | wc -l ; "
             f"du -sh {root}/stimuli 2>/dev/null"])
    return {"subset": subset, "got": got}


@app.local_entrypoint()
def main(subset: str = "movie10,ood"):
    print(fetch.remote(subset))
