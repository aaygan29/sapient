"""Project HAD (ds004488) preprocessed BOLD → fsaverage5 → 1 Hz → z-score.

CONTRACTS.md §0 (WS-D routing) + §1 (20,484 fsaverage5) + §3 (cache paths).
Sibling of `data/prepare_fmri.py` (the Huth/ds002345 projector); same core math
(`vol_to_surf` → resample → z-score) but adapted to HAD's layout & convention:

  * BOLD lives in the fMRIPrep derivatives, FLAT-named (no `ses-`/`space-` tag):
    `derivatives/fmriprep/sub-XX/sub-XX_task-action_run-N_desc-preproc_bold.nii.gz`
    These are volumetric MNI152NLin2009cAsym (2 mm, affine origin ~ -81/-83/-75,
    shape (83,89,77,T)). `vol_to_surf` resamples the fsaverage5 pial mesh into the
    volume's world space, so the standard projection applies unchanged.
  * TR = 2.0 s (sidecar `RepetitionTime`), 156 TRs/run = 312 s.
  * HAD is EVENT-BASED, not a continuous movie: each run is an INDEPENDENT 312 s
    timeline of 2 s clips + ISIs. So — unlike the Huth projector — runs are NEVER
    concatenated; we emit ONE array per (subject, run):
        /data/fmri/mary/had/{subject}/{run}.npy   (T_1Hz≈312, 20484) float32
    `run` key = `run-NN` (zero-padded, 1..12) so it lines up with the feature
    assembly step and the events.tsv (`*_run-NN_events.tsv`).

After projection: resample TR-grid → 1 Hz (linear interp, family convention) then
z-score per vertex per run. ~312 frames/run @ 1 Hz.

Usage:
  modal run data/prepare_fmri_had.py --limit 1                 # validate one run
  modal run --detach data/prepare_fmri_had.py                  # sub-01 + sub-02, all runs
  modal run data/prepare_fmri_had.py --subjects sub-01 --runs run-01
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

FSAVERAGE_MESH = "fsaverage5"
N_VERTICES = 20484
DATASET = "had"
FMRIPREP_ROOT_DEFAULT = "/data/raw/ds004488/derivatives/fmriprep"
OUT_ROOT_DEFAULT = "/data/fmri/mary/had"
DEFAULT_TR = 2.0

RUN_RE = re.compile(r"run-(\d+)")
# desc-preproc bold, FLAT derivative names (no space- tag present on these files)
BOLD_GLOB = "*task-action*desc-preproc_bold.nii.gz"


def read_tr(bold_path: Path) -> float:
    side = bold_path.with_name(bold_path.name.replace(".nii.gz", ".json"))
    if side.exists():
        try:
            tr = float(json.loads(side.read_text()).get("RepetitionTime", DEFAULT_TR))
            if tr > 0:
                return tr
        except Exception:
            pass
    return DEFAULT_TR


def run_key(bold_path: Path) -> str:
    """`run-NN` zero-padded so it matches events.tsv + feature assembly."""
    m = RUN_RE.search(bold_path.name)
    n = int(m.group(1)) if m else 0
    return f"run-{n:02d}"


def project_to_fsaverage5(bold_path: Path, fsaverage) -> "np.ndarray":  # noqa: F821
    import nibabel as nib
    from nilearn import surface

    img = nib.load(str(bold_path))
    lh = surface.vol_to_surf(img, fsaverage.pial_left, radius=3.0, kind="auto")
    rh = surface.vol_to_surf(img, fsaverage.pial_right, radius=3.0, kind="auto")
    out = np.concatenate([lh.T, rh.T], axis=1).astype(np.float32)  # (T, V)
    if out.shape[1] != N_VERTICES:
        raise RuntimeError(f"Expected {N_VERTICES} vertices, got {out.shape[1]}")
    return out


def resample_to_1hz(arr: "np.ndarray", tr_seconds: float) -> "np.ndarray":  # noqa: F821
    from scipy.interpolate import interp1d

    n_in = arr.shape[0]
    if n_in < 2:
        return arr.astype(np.float32)
    t_in = np.arange(n_in) * tr_seconds
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0)
    f = interp1d(t_in, arr, axis=0, kind="linear", fill_value="extrapolate")
    return f(t_out).astype(np.float32)


def zscore_per_vertex(arr: "np.ndarray") -> "np.ndarray":  # noqa: F821
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = (arr - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def discover_runs(fmriprep_root: Path, subjects: list[str] | None,
                  runs: list[str] | None) -> dict[tuple[str, str], Path]:
    """(subject, run-NN) -> bold path. One bold per run (event-based; no concat)."""
    found: dict[tuple[str, str], Path] = {}
    sub_dirs = sorted(fmriprep_root.glob("sub-*"))
    if subjects:
        keep = set(subjects)
        sub_dirs = [d for d in sub_dirs if d.name in keep]
    for sub_dir in sub_dirs:
        func = sub_dir / "func"
        search_dir = func if func.exists() else sub_dir   # HAD deriv is flat in sub dir
        for nii in sorted(search_dir.glob(BOLD_GLOB)):
            rk = run_key(nii)
            if runs and rk not in runs:
                continue
            found[(sub_dir.name, rk)] = nii
    return found


def process_one(subject: str, rk: str, bold_path: Path, out_root: Path,
                fsaverage) -> dict:
    out = out_root / subject / f"{rk}.npy"
    if out.exists():
        n = int(np.load(out, mmap_mode="r").shape[0])
        return {"subject": subject, "run": rk, "n_trs": n,
                "path": str(out), "status": "cached"}
    out.parent.mkdir(parents=True, exist_ok=True)

    tr = read_tr(bold_path)
    surf = project_to_fsaverage5(bold_path, fsaverage)    # (T_tr, V)
    surf = resample_to_1hz(surf, tr)                      # (T_1hz, V)
    arr = zscore_per_vertex(surf)
    np.save(out, arr)
    return {"subject": subject, "run": rk, "n_trs": int(arr.shape[0]),
            "tr": tr, "path": str(out),
            "finite": bool(np.isfinite(arr).all()),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "status": "written"}


APP_NAME = "mary-prepare-fmri-had"

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "nilearn>=0.10.4", "nibabel>=5.2", "numpy>=1.26,<3",
        "scipy>=1.13", "scikit-learn>=1.4",
    )
)
_volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
_hf_secret = modal.Secret.from_name("hf-token")
_app = modal.App(APP_NAME)


@_app.function(
    image=_image, cpu=4.0, memory=16384, timeout=12 * 60 * 60,
    volumes={"/data": _volume}, secrets=[_hf_secret],
)
def prepare_remote(
    fmriprep_root: str = FMRIPREP_ROOT_DEFAULT,
    out_root: str = OUT_ROOT_DEFAULT,
    subjects: list[str] | None = None,
    runs: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    global np
    import numpy as np  # noqa: F811
    from nilearn import datasets as nl_datasets

    root = Path(fmriprep_root)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    found = discover_runs(root, subjects, runs)
    keys = sorted(found.keys())
    if limit:
        keys = keys[:limit]
    print(f"Found {len(keys)} (subject,run) HAD bold runs to process.")

    print("Fetching fsaverage5 mesh ...")
    fsaverage = nl_datasets.fetch_surf_fsaverage(mesh=FSAVERAGE_MESH)

    results = []
    for i, (sub, rk) in enumerate(keys):
        print(f"  [{i+1}/{len(keys)}] {sub} / {rk}")
        try:
            r = process_one(sub, rk, found[(sub, rk)], out, fsaverage)
            print(f"      -> {r['status']} n_trs={r.get('n_trs')} "
                  f"mean={r.get('mean')} std={r.get('std')}")
            results.append(r)
        except Exception as e:
            print(f"      !! FAILED: {e}")
            results.append({"subject": sub, "run": rk, "status": "error",
                            "error": str(e)})
        _volume.commit()

    summary = {
        "groups": len(keys),
        "written": len([r for r in results if r["status"] == "written"]),
        "cached": len([r for r in results if r["status"] == "cached"]),
        "errors": len([r for r in results if r["status"] == "error"]),
        "results": results,
    }
    print(f"\nDONE: written={summary['written']} cached={summary['cached']} "
          f"errors={summary['errors']}")
    return summary


@_app.local_entrypoint()
def main(
    fmriprep_root: str = FMRIPREP_ROOT_DEFAULT,
    out_root: str = OUT_ROOT_DEFAULT,
    subjects: str = "sub-01,sub-02",
    runs: str = "",
    limit: int = 0,
) -> None:
    subs = [s for s in subjects.split(",") if s] or None
    rns = [r for r in runs.split(",") if r] or None
    res = prepare_remote.remote(fmriprep_root, out_root, subs, rns, limit or None)
    print("\n========== SUMMARY ==========")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    for r in res["results"][:30]:
        print(" ", r.get("subject"), r.get("run"), r.get("status"),
              "n_trs=", r.get("n_trs"))
