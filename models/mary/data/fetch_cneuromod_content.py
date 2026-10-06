"""Materialize CNeuroMod / Algonauts-2025 git-annex content on the volume.

WHY THIS EXISTS
---------------
The Algonauts-2025 (`raw/algonauts2025`) and CNeuroMod fmriprep
(`raw/cneuromod/fmriprep`) trees are present on the `sapient-data` volume only
as DataLad/git-annex POINTERS — the actual bytes were never fetched (the
`.git/annex/objects/` stores are empty; every `.mkv`/`.tsv`/`_bold.nii.gz` is a
broken symlink of ~135–149 bytes). Feature extraction and fMRI projection cannot
run until the content is materialized. This job runs `datalad get` on the exact
paths Mary needs, then verifies the bytes landed. It is the prerequisite for:
  data/extract_{slowfast,qwen_vl,got_ocr,whisper,beats,qwen_ctx}.py --dataset cneuromod
  data/prepare_fmri.py::cneuromod

WHAT IT FETCHES (sub-01, sub-02; movie10 + a few friends + ood)
  STIMULI  (raw/algonauts2025/stimuli):
    - transcripts/movie10/**          (.tsv, tiny — qwen_ctx text stream)
    - movies/movie10/**               (.mkv — the 4 movie10 films incl. Bourne)
    - movies/ood/**                   (.mkv — OOD eval clips; optional)
    - movies/friends/s1/<first eps>   (.mkv — a few Friends episodes)
    - transcripts/{ood,friends}/**    (.tsv for the above)
  fMRI     (raw/cneuromod/fmriprep):
    - {movie10,friends}/sub-01 + sub-02 — MNI152NLin2009cAsym preproc BOLD

Idempotent: `datalad get` skips already-present files; re-running is safe.
DETACHED launch (survives laptop sleep):
  modal run --detach data/fetch_cneuromod_content.py                 # default slice
  modal run --detach data/fetch_cneuromod_content.py --what stimuli  # just stimuli
  modal run --detach data/fetch_cneuromod_content.py --what fmri     # just fMRI
"""

from __future__ import annotations

import subprocess

import modal

app = modal.App("mary-fetch-cneuromod")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-annex", "wget", "curl")
    .pip_install("datalad>=1.0")
)

ALG = "/data/raw/algonauts2025"
FMRIPREP = "/data/raw/cneuromod/fmriprep"

# Stimulus paths to materialize (relative to ALG). movie10 first (the priority +
# Bourne); ood + a few friends episodes round out the slice.
STIM_PATHS = [
    "stimuli/transcripts/movie10",
    "stimuli/movies/movie10",
    "stimuli/transcripts/ood",
    "stimuli/movies/ood",
    "stimuli/transcripts/friends/s1",
    # a handful of Friends s1 episodes (a/b half-episodes) to seed the friends slice
    "stimuli/movies/friends/s1/friends_s01e01a.mkv",
    "stimuli/movies/friends/s1/friends_s01e01b.mkv",
    "stimuli/movies/friends/s1/friends_s01e02a.mkv",
    "stimuli/movies/friends/s1/friends_s01e02b.mkv",
]

# fMRI: movie10 + friends fmriprep, sub-01 + sub-02 only (the sprint cohort).
FMRI_GROUPS = ["movie10", "friends"]
FMRI_SUBJECTS = ["sub-01", "sub-02"]


def _sh(cmd, cwd=None, timeout=None):
    print("+", " ".join(cmd), flush=True)
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  !! TIMEOUT", flush=True)
        return None
    if r.stdout:
        print(r.stdout[-2500:], flush=True)
    if r.returncode != 0 and r.stderr:
        print("STDERR:", r.stderr[-2500:], flush=True)
    return r


def _git_setup():
    _sh(["git", "config", "--global", "user.email", "mary@sapient.local"])
    _sh(["git", "config", "--global", "user.name", "mary"])
    _sh(["git", "config", "--global", "--add", "safe.directory", "*"])


def _verify(path_glob: str, label: str, root: str):
    """Report how many files under `root` matching `path_glob` are real bytes."""
    import glob
    import os
    real = small = 0
    for f in glob.glob(os.path.join(root, path_glob), recursive=True):
        try:
            if os.path.getsize(f) > 100_000:   # >100 KB = real content
                real += 1
            else:
                small += 1
        except OSError:
            small += 1
    print(f"  [{label}] materialized={real} still-pointer/small={small}", flush=True)
    return real, small


@app.function(image=image, cpu=8.0, timeout=12 * 60 * 60,
              volumes={"/data": vol}, secrets=[modal.Secret.from_name("hf-token")])
def fetch(what: str = "all", friends_only: str = "") -> dict:
    import os

    _git_setup()
    did: dict[str, bool] = {}

    if what in ("all", "stimuli"):
        if not os.path.exists(ALG):
            print(f"!! {ALG} missing — run fetch_algonauts_stimuli.py first.")
        else:
            # Ensure the movie subdatasets are installed (no-op if already).
            _sh(["datalad", "-C", ALG, "get", "-n", "-r",
                 "stimuli/movies", "stimuli/transcripts"], timeout=1800)
            paths = STIM_PATHS
            if friends_only:
                paths = [p for p in STIM_PATHS if "friends" in p] or STIM_PATHS
            for rel in paths:
                target = os.path.join(ALG, rel)
                r = _sh(["datalad", "-C", ALG, "get", "-r", "-J", "6", rel],
                        timeout=6 * 3600)
                did[f"stim:{rel}"] = bool(r and r.returncode == 0)
                vol.commit()
            _verify("stimuli/movies/movie10/**/*.mkv", "movie10 mkv", ALG)
            _verify("stimuli/movies/ood/**/*.mkv", "ood mkv", ALG)
            _verify("stimuli/movies/friends/**/*.mkv", "friends mkv", ALG)
            _verify("stimuli/transcripts/movie10/**/*.tsv", "movie10 tsv", ALG)

    if what in ("all", "fmri"):
        if not os.path.exists(FMRIPREP):
            print(f"!! {FMRIPREP} missing.")
        else:
            for group in FMRI_GROUPS:
                gdir = os.path.join(FMRIPREP, group)
                # install the group subdataset (metadata) if needed
                _sh(["datalad", "-C", FMRIPREP, "get", "-n", group], timeout=1800)
                for sub in FMRI_SUBJECTS:
                    rel = f"{group}/{sub}"
                    if not os.path.exists(os.path.join(FMRIPREP, rel)):
                        # subject may live one level down after subdataset install
                        pass
                    # only the MNI preproc BOLD (skip T1w/fsLR to save space/time)
                    r = _sh(["datalad", "-C", FMRIPREP, "get", "-r", "-J", "6",
                             f"{rel}"], timeout=10 * 3600)
                    did[f"fmri:{rel}"] = bool(r and r.returncode == 0)
                    vol.commit()
            for group in FMRI_GROUPS:
                _verify(f"{group}/sub-01/**/*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                        f"{group} sub-01 bold", FMRIPREP)

    vol.commit()
    print("\nDONE.", {k: v for k, v in did.items()})
    return {"what": what, "results": did}


@app.local_entrypoint()
def main(what: str = "all", friends_only: str = "") -> None:
    print(fetch.remote(what=what, friends_only=friends_only))
