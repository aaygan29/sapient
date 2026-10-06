"""Project preprocessed BOLD onto fsaverage5 and build the Schaefer-1000 atlas.

Per spec §2.2 and decision #10 (Schaefer atlas is built ONCE offline and used
only in post-processing — never at training time).

For each BIDS-style preprocessed BOLD run (MNI152NLin2009cAsym space):
  1. Load with nibabel.
  2. Project to fsaverage5 surface (10,242 vertices/hemisphere → 20,484 total).
  3. Resample to 1 Hz along the time axis.
  4. Save as a memory-mapped .npy of shape (T_1hz, 20484), grouped by subject.

Once per workspace (run with --build-parcellation), build
``data/parcellate.npz``: a sparse (1000, 20484) vertex-to-parcel averaging
matrix derived from the Schaefer-2018 7-network 1000-parcel atlas.

Usage:
  python -m data.prepare_fmri \
      --in data/raw/cneuromod --out data/fmri/cneuromod \
      --tr 1.49 --build-parcellation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import datasets, surface
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, save_npz

FSAVERAGE_MESH = "fsaverage5"
N_VERTICES = 20484          # fsaverage5: 10,242 per hemisphere × 2
N_PARCELS = 1000            # Schaefer-2018 7-networks 1k
PARCELLATION_PATH = Path("data/parcellate.npz")
CNEUROMOD_CC0_SUBJECTS = {"sub-01", "sub-02", "sub-03", "sub-05"}
BOLD_SUFFIX = "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"


def find_bold_runs(raw_root: Path, cc0_only: bool = True) -> list[Path]:
    """Discover all fmriprep BOLD runs in MNI space."""
    runs: list[Path] = []
    for sub_dir in sorted(raw_root.glob("fmriprep/*/sub-*")):
        if cc0_only and sub_dir.name not in CNEUROMOD_CC0_SUBJECTS:
            continue
        for nii in sub_dir.rglob(f"*{BOLD_SUFFIX}"):
            runs.append(nii)
    return runs


def project_to_fsaverage5(bold_nii: Path, fsaverage) -> np.ndarray:
    """Project a 4D MNI BOLD file onto fsaverage5 (LH ⊕ RH concatenated)."""
    img = nib.load(str(bold_nii))
    lh = surface.vol_to_surf(img, fsaverage.pial_left, radius=3.0, kind="auto")
    rh = surface.vol_to_surf(img, fsaverage.pial_right, radius=3.0, kind="auto")
    # nilearn returns (V, T); concat along vertex axis after transposing to (T, V).
    out = np.concatenate([lh.T, rh.T], axis=1).astype(np.float32)
    if out.shape[1] != N_VERTICES:
        raise RuntimeError(f"Expected {N_VERTICES} vertices, got {out.shape[1]}")
    return out


def resample_to_1hz(arr: np.ndarray, tr_seconds: float) -> np.ndarray:
    """Linear-interpolate (T, V) at native TR to (T_1hz, V) at 1 Hz."""
    n_in = arr.shape[0]
    t_in = np.arange(n_in) * tr_seconds
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0)
    f = interp1d(t_in, arr, axis=0, kind="linear", fill_value="extrapolate")
    return f(t_out).astype(np.float32)


def build_schaefer_parcellation_matrix(out_path: Path) -> None:
    """Build a sparse (1000, 20484) vertex-to-parcel averaging matrix.

    Rows sum to 1 by averaging the vertices that fall inside each parcel.
    Computed by nearest-neighbor projection of the volumetric Schaefer-2018
    atlas onto fsaverage5 (the surface .annot release is gated; volumetric
    is shipped with nilearn).
    """
    print(f"Building Schaefer-1000 vertex→parcel matrix → {out_path}")
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_PARCELS, yeo_networks=7, resolution_mm=2
    )
    fsaverage = datasets.fetch_surf_fsaverage(mesh=FSAVERAGE_MESH)
    atlas_img = nib.load(atlas.maps)

    lh_labels = surface.vol_to_surf(
        atlas_img, fsaverage.pial_left, radius=3.0, interpolation="nearest"
    )
    rh_labels = surface.vol_to_surf(
        atlas_img, fsaverage.pial_right, radius=3.0, interpolation="nearest"
    )
    labels = np.concatenate([lh_labels, rh_labels]).astype(np.int32)  # (20484,)
    if labels.shape[0] != N_VERTICES:
        raise RuntimeError(
            f"label shape {labels.shape} does not match {N_VERTICES} vertices"
        )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    n_filled = 0
    for parcel_id in range(1, N_PARCELS + 1):
        idx = np.where(labels == parcel_id)[0]
        if len(idx) == 0:
            continue
        w = 1.0 / len(idx)
        rows.extend([parcel_id - 1] * len(idx))
        cols.extend(idx.tolist())
        data.extend([w] * len(idx))
        n_filled += 1
    M = csr_matrix(
        (data, (rows, cols)), shape=(N_PARCELS, N_VERTICES), dtype=np.float32
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out_path, M)
    print(f"  shape={M.shape}, nnz={M.nnz}, parcels_filled={n_filled}/{N_PARCELS}")


def _project_all(raw_root: Path, out_root: Path, tr: float,
                 cc0_only: bool, build_parcellation: bool,
                 parcellation_path: Path = PARCELLATION_PATH) -> dict:
    """Core projection loop. Pure function — used by both CLI and Modal entry points."""
    out_root.mkdir(parents=True, exist_ok=True)
    if build_parcellation:
        build_schaefer_parcellation_matrix(parcellation_path)

    fsaverage = datasets.fetch_surf_fsaverage(mesh=FSAVERAGE_MESH)
    runs = find_bold_runs(raw_root, cc0_only=cc0_only)
    print(f"Found {len(runs)} BOLD runs.")
    if not runs:
        return {"runs_processed": 0, "warning": "no runs found"}

    n_done = n_skip = 0
    for run in runs:
        sub = next((p for p in run.parts if p.startswith("sub-")), "sub-unknown")
        out_name = run.name.replace(BOLD_SUFFIX, ".npy")
        out = out_root / sub / out_name
        if out.exists():
            print(f"  skip {out} (already cached)")
            n_skip += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"  project {run.name} → {out}")
        arr = project_to_fsaverage5(run, fsaverage)
        arr = resample_to_1hz(arr, tr)
        np.save(out, arr)
        n_done += 1
    return {"runs_processed": int(n_done), "runs_skipped": int(n_skip)}


def main() -> None:
    """Local CLI entry point. Works against any path the local Python can see."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="raw_root", type=Path, required=True,
                    help="data/raw/<dataset> (or /data/raw/<dataset> on Modal)")
    ap.add_argument("--out", dest="out_root", type=Path, required=True,
                    help="data/fmri/<dataset> (or /data/fmri/<dataset> on Modal)")
    ap.add_argument("--tr", type=float, required=True,
                    help="Native TR in seconds (e.g. 1.49 for CNeuroMod)")
    ap.add_argument("--cc0-only", action="store_true", default=True,
                    help="Restrict to the 4 CC0 CNeuroMod subjects (default)")
    ap.add_argument("--all-subjects", dest="cc0_only", action="store_false",
                    help="Process all subjects under raw_root (for non-CNeuroMod datasets)")
    ap.add_argument("--build-parcellation", action="store_true",
                    help="Build data/parcellate.npz at the start.")
    args = ap.parse_args()
    result = _project_all(
        args.raw_root, args.out_root, args.tr,
        cc0_only=args.cc0_only,
        build_parcellation=args.build_parcellation,
    )
    print(f"Done. {result}")


# ============================================================
# Modal wrapper — keeps fMRI prep on the same volume as the data.
# Architectural rule: no raw BOLD on a laptop. Volume-mounted on Modal.
# ============================================================

import modal  # noqa: E402

APP_NAME = "sapient-1-prepare-fmri"

_modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "nilearn>=0.10.4",
        "nibabel>=5.2",
        "numpy>=1.26,<3",
        "scipy>=1.13",
        "scikit-learn>=1.4",
    )
)

_modal_volume = modal.Volume.from_name("sapient-data", create_if_missing=True)

_app = modal.App(APP_NAME)


@_app.function(
    image=_modal_image,
    cpu=4.0,
    memory=16384,
    timeout=8 * 60 * 60,
    volumes={"/data": _modal_volume},
)
def prepare_remote(
    raw_root: str,
    out_root: str,
    tr: float,
    cc0_only: bool = True,
    build_parcellation: bool = False,
) -> dict:
    """Modal-resident wrapper. raw_root + out_root are paths inside /data/."""
    result = _project_all(
        Path(raw_root), Path(out_root), tr,
        cc0_only=cc0_only,
        build_parcellation=build_parcellation,
        parcellation_path=Path("/data/parcellate.npz"),
    )
    _modal_volume.commit()
    return result


@_app.local_entrypoint()
def modal_main(
    raw_root: str = "/data/raw/cneuromod",
    out_root: str = "/data/fmri/cneuromod",
    tr: float = 1.49,
    build_parcellation: bool = False,
) -> None:
    """`modal run data/prepare_fmri.py --raw-root /data/raw/cneuromod \\
                                       --out-root /data/fmri/cneuromod \\
                                       --tr 1.49 --build-parcellation`"""
    result = prepare_remote.remote(
        raw_root, out_root, tr,
        cc0_only=True,
        build_parcellation=build_parcellation,
    )
    print(f"Done. {result}")


if __name__ == "__main__":
    main()
