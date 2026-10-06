"""Fetch specific OpenNeuro datasets/subjects (+stimuli) to the sapient-data volume
via public S3 (no credentials). Probe mode lists sizes first so we don't pull TBs.

ORCLE training datasets to add (paper §4.1):
  ds003020  Lebel2023        sub-EN057, sub-EN058   audio+text   CC0
  ds004488  HAD (actions)    sub-01, sub-02         audio+video  CC-BY
  (Huth ds002345 already on volume; Wen2017 = Purdue PURR, separate script.)

Usage:
  modal run data/fetch_openneuro.py --dataset ds003020 --probe
  modal run data/fetch_openneuro.py --dataset ds003020 --subjects sub-EN057,sub-EN058
"""
import subprocess
import modal

app = modal.App("mary-fetch-openneuro")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("awscli"))

S3 = "s3://openneuro.org"


def _sh(cmd):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


@app.function(image=image, cpu=4.0, timeout=6 * 60 * 60, volumes={"/data": vol})
def fetch(dataset: str, subjects: str = "", probe: bool = False,
          include_stimuli: bool = True, deriv_subjects: str = ""):
    base = f"{S3}/{dataset}"
    if probe:
        # top-level listing + sizes of a couple key prefixes
        print("=== top level ===")
        print(_sh(["aws", "s3", "ls", "--no-sign-request", f"{base}/"]).stdout)
        for pref in ["stimuli/", "derivatives/"]:
            r = _sh(["aws", "s3", "ls", "--no-sign-request", "--summarize",
                     "--recursive", "--human-readable", f"{base}/{pref}"])
            tail = "\n".join(r.stdout.splitlines()[-4:])
            print(f"=== {pref} size ===\n{tail}")
        return {"dataset": dataset, "mode": "probe"}

    subs = [s for s in subjects.split(",") if s]
    dest = f"/data/raw/{dataset}"
    # always grab the small top-level metadata
    _sh(["aws", "s3", "cp", "--no-sign-request", f"{base}/dataset_description.json",
         f"{dest}/dataset_description.json"])
    results = {}
    for sub in subs:
        r = _sh(["aws", "s3", "sync", "--no-sign-request",
                 f"{base}/{sub}", f"{dest}/{sub}"])
        ok = r.returncode == 0
        print(f"  {sub}: {'OK' if ok else 'FAIL ' + r.stderr[-300:]}")
        results[sub] = ok
        vol.commit()
    for sub in [s for s in deriv_subjects.split(",") if s]:
        r = _sh(["aws", "s3", "sync", "--no-sign-request",
                 f"{base}/derivatives/fmriprep/{sub}", f"{dest}/derivatives/fmriprep/{sub}"])
        print(f"  deriv {sub}: rc={r.returncode}")
        results[f"deriv:{sub}"] = r.returncode == 0
        vol.commit()
    if include_stimuli:
        r = _sh(["aws", "s3", "sync", "--no-sign-request",
                 f"{base}/stimuli", f"{dest}/stimuli"])
        print(f"  stimuli: rc={r.returncode}")
        results["stimuli"] = r.returncode == 0
        vol.commit()
    return {"dataset": dataset, "subjects": results}


@app.local_entrypoint()
def main(dataset: str, subjects: str = "", probe: bool = False,
         include_stimuli: bool = True, deriv_subjects: str = ""):
    print(fetch.remote(dataset, subjects, probe, include_stimuli, deriv_subjects))
