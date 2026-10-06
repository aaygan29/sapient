"""Materialize Wen2017 Subject-1 fMRI on the sapient-data volume.

PURR (HUBzero) double-wraps the data: the DOI zip contains
`10_4231_R7X63K3M/bundle.zip`, and THAT inner bundle contains
`video_fmri_dataset/subject1/fmri/{seg1..18,test1..5}/{mni,cifti,raw}/...`.

This app, in one shot:
  1. (re)downloads the 15.3 GB outer DOI zip if absent/wrong-size (wget -c resume),
  2. extracts it fully -> 10_4231_R7X63K3M/bundle.zip,
  3. extracts the inner bundle fully -> video_fmri_dataset/subject1/fmri/...,
  4. verifies enough segments materialized, THEN reclaims space.

Idempotent + safe: it only deletes source zips after confirming the expected
segment/MNI counts, so an interrupted run never destroys the only copy.

ALWAYS run under `modal run --detach data/unzip_wen2017.py::main` and DO NOT stop
it mid-extraction (a half-written inner bundle is unrecoverable without re-download).

`modal run --detach data/unzip_wen2017.py::main`     # subject-1 fMRI (+ stimuli)
`modal run data/unzip_wen2017.py::report_ep`         # inventory only
"""
from __future__ import annotations

import json
import subprocess
import time
import zipfile
from pathlib import Path

import modal

vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "wget", "unzip", "p7zip-full"
)
app = modal.App("mary-unzip-wen2017")

ROOT = "/data/raw/wen2017"
OUTER_ZIP = f"{ROOT}/10_4231_R7X63K3M.zip"
OUTER_URL = "https://purr.purdue.edu/publications/2805/serve/1?render=archive"
OUTER_EXPECTED = 15_297_443_422
INNER_BUNDLE = f"{ROOT}/10_4231_R7X63K3M/bundle.zip"
STIM_BUNDLE = f"{ROOT}/10_4231_R71Z42KK/bundle.zip"
DS = f"{ROOT}/video_fmri_dataset"
S1_FMRI = f"{DS}/subject1/fmri"
# 18 seg + 5 test = 23 segment dirs expected; require a healthy majority before
# deleting sources (allow a couple of stragglers, but never the 2-of-23 truncation).
MIN_SEGMENTS_OK = 20


def _unzip(zip_path: Path, dest: Path) -> dict:
    """Extract an archive. Python zipfile first; fall back to 7z for ZIP64."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            n = len(zf.namelist())
            zf.extractall(dest)
        return {"tool": "zipfile", "n_members": n, "ok": True}
    except Exception as e:
        py_err = str(e)[:200]
    r = subprocess.run(["7z", "x", "-y", f"-o{dest}", str(zip_path)],
                       capture_output=True, text=True)
    return {"tool": "7z", "ok": r.returncode == 0, "rc": r.returncode,
            "tail": (r.stdout + r.stderr)[-500:], "zipfile_error": py_err}


def _seg_counts() -> dict:
    f = Path(S1_FMRI)
    if not f.is_dir():
        return {"n_seg": 0, "n_test": 0, "n_mni": 0}
    return {"n_seg": len(list(f.glob("seg*"))),
            "n_test": len(list(f.glob("test*"))),
            "n_mni": len(list(f.rglob("*_mni.nii.gz")))}


@app.function(cpu=4, memory=16384, timeout=6 * 3600, image=image,
              volumes={"/data": vol})
def materialize(target: str = "all") -> dict:
    out: dict = {}

    # ---- stimuli (idempotent) ----
    if target in ("all", "stimuli"):
        stim = Path(DS) / "stimuli"
        if stim.is_dir() and list(stim.glob("*.mp4")):
            out["stimuli"] = {"status": "already", "n_mp4": len(list(stim.glob("*.mp4")))}
        elif Path(STIM_BUNDLE).exists():
            r = _unzip(Path(STIM_BUNDLE), Path(ROOT))
            vol.commit()
            out["stimuli"] = {"status": "extracted",
                              "n_mp4": len(list(stim.glob("*.mp4"))), **r}
        else:
            out["stimuli"] = {"status": "no_bundle"}

    # ---- subject-1 fMRI ----
    if target in ("all", "subject1"):
        c = _seg_counts()
        if c["n_seg"] + c["n_test"] >= MIN_SEGMENTS_OK and c["n_mni"] >= MIN_SEGMENTS_OK:
            out["subject1"] = {"status": "already", **c}
            return out

        # Step A: ensure the outer DOI zip is present + full-size.
        outer = Path(OUTER_ZIP)
        inner = Path(INNER_BUNDLE)
        if not inner.exists() or inner.stat().st_size < 1_000_000_000:
            cur = outer.stat().st_size if outer.exists() else 0
            if cur != OUTER_EXPECTED:
                print(f"[s1] downloading outer ({cur:,}/{OUTER_EXPECTED:,})", flush=True)
                t0 = time.time()
                rc = subprocess.run(
                    ["wget", "--continue", "--tries=30", "--timeout=120",
                     "--waitretry=15", "--no-verbose", "--progress=dot:giga",
                     "-O", str(outer), OUTER_URL], check=False).returncode
                vol.commit()
                got = outer.stat().st_size if outer.exists() else 0
                out["outer_fetch"] = {"rc": rc, "bytes": got, "expected": OUTER_EXPECTED,
                                      "secs": round(time.time() - t0, 1)}
                print(f"[s1] wget rc={rc} {got:,} bytes", flush=True)
                if got != OUTER_EXPECTED:
                    return {**out, "subject1": {"status": "outer_download_incomplete",
                                                "bytes": got}}
            # Step B: extract outer -> 10_4231_R7X63K3M/bundle.zip
            print("[s1] extracting outer DOI zip ...", flush=True)
            out["outer_extract"] = _unzip(outer, Path(ROOT))
            vol.commit()
            if not inner.exists() or inner.stat().st_size < 1_000_000_000:
                return {**out, "subject1": {
                    "status": "inner_bundle_too_small_or_missing",
                    "inner_bytes": inner.stat().st_size if inner.exists() else 0}}

        # Step C: extract inner bundle -> video_fmri_dataset/subject1/fmri/...
        print(f"[s1] extracting inner bundle ({inner.stat().st_size:,} B) ...", flush=True)
        out["inner_extract"] = _unzip(inner, Path(ROOT))
        vol.commit()
        c = _seg_counts()
        out["subject1"] = {"status": "extracted", **c}
        print(f"[s1] segments: {c}", flush=True)

        # Step D: reclaim space ONLY if a healthy majority materialized.
        if c["n_seg"] + c["n_test"] >= MIN_SEGMENTS_OK and c["n_mni"] >= MIN_SEGMENTS_OK:
            inner.unlink(missing_ok=True)
            outer.unlink(missing_ok=True)
            try:
                Path(INNER_BUNDLE).parent.rmdir()
            except OSError:
                pass
            vol.commit()
            out["subject1"]["reclaimed"] = True
        else:
            out["subject1"]["status"] = "incomplete_kept_sources"
    return out


@app.function(cpu=2, memory=4096, timeout=20 * 60, image=image,
              volumes={"/data": vol})
def report() -> dict:
    dsp = Path(DS)
    out = {"video_fmri_dataset_exists": dsp.exists()}
    stim = dsp / "stimuli"
    if stim.is_dir():
        out["stimuli"] = {"n_mp4": len(list(stim.glob("*.mp4")))}
    f = Path(S1_FMRI)
    if f.is_dir():
        mni = sorted(f.rglob("*_mni.nii.gz"))
        out["subject1"] = {
            "seg_dirs": sorted(p.name for p in f.iterdir() if p.is_dir()),
            "n_mni": len(mni),
            "sample_mni": str(mni[0].relative_to(dsp)) if mni else None,
            "sample_mni_bytes": mni[0].stat().st_size if mni else None,
        }
    for z in (OUTER_ZIP, INNER_BUNDLE):
        if Path(z).exists():
            out.setdefault("leftover_zips", {})[z] = Path(z).stat().st_size
    return out


@app.local_entrypoint()
def main(target: str = "all"):
    print(json.dumps(materialize.remote(target), indent=2))


@app.local_entrypoint()
def report_ep():
    print(json.dumps(report.remote(), indent=2))
