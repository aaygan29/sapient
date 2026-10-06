"""
sapienteval/modal/repair_nifti.py — Modal app to repair malformed ds004996 BOLD files.

Problem
-------
ds004996 .nii BOLD files have NIfTI headers that claim more time points
than the actual file content contains. nibabel/fmriprep raise:
    OSError: Expected N bytes, got M bytes from ... - could the file be damaged?

OpenNeuro's S3 hosts these files as-is (Content-Length matches the actual
truncated bytes), so this is a dataset-level issue, not a download issue.

Repair
------
For each affected .nii:
  1. Read header to determine voxel layout: dim1*dim2*dim3 * bytes_per_voxel
  2. Compute actual_data_bytes = file_size - header_size
  3. actual_volumes = actual_data_bytes // bytes_per_volume
  4. Patch header dim[4] (time) to actual_volumes
  5. Save back, in-place

Result: header now matches data. fmriprep can ingest cleanly. We lose the
trailing TRs that the original header lied about (but those bytes were never
on disk to begin with, so this only corrects the metadata).

Deploy: modal deploy sapienteval/modal/repair_nifti.py
Run:    modal run sapienteval/modal/repair_nifti.py
        modal run sapienteval/modal/repair_nifti.py --subjects sub-02,sub-04,sub-05
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("nibabel", "numpy")
)

data_vol = modal.Volume.from_name("sapienteval-data", create_if_missing=True)

app = modal.App("sapienteval-repair-nifti", image=image)


@app.function(
    cpu=2,
    memory=4096,
    timeout=30 * 60,
    volumes={"/data": data_vol},
)
def repair_subject(subject_id: str) -> dict:
    """Walk a subject's BOLD .nii files; patch any header/data mismatches in-place."""
    import nibabel as nib
    import numpy as np

    label = subject_id.replace("sub-", "")
    func_dir = Path(f"/data/ds004996/sub-{label}/func")
    if not func_dir.exists():
        return {"subject_id": subject_id, "status": "missing_func"}

    per_file = []
    t0 = time.time()
    for nii_path in sorted(func_dir.glob(f"sub-{label}_task-*_run-*_bold.nii")):
        try:
            img = nib.load(str(nii_path))
            header = img.header.copy()
            actual_bytes = nii_path.stat().st_size
            # Standard NIfTI-1 header size is 348 bytes; the data offset
            # lives in vox_offset.
            vox_offset = int(header["vox_offset"])
            data_bytes_available = actual_bytes - vox_offset

            dim = header["dim"]            # array of 8 ints: [n_dims, d1, d2, d3, d4, ...]
            n_dims = int(dim[0])
            d1, d2, d3, d4 = int(dim[1]), int(dim[2]), int(dim[3]), int(dim[4])
            bytes_per_voxel = int(header["bitpix"]) // 8
            bytes_per_volume = d1 * d2 * d3 * bytes_per_voxel
            expected_bytes = bytes_per_volume * d4

            if data_bytes_available >= expected_bytes:
                per_file.append({
                    "file": nii_path.name,
                    "status": "ok",
                    "actual_volumes": d4,
                })
                continue

            actual_volumes = data_bytes_available // bytes_per_volume
            if actual_volumes <= 0:
                per_file.append({
                    "file": nii_path.name,
                    "status": "unrepairable_too_small",
                    "data_bytes": data_bytes_available,
                    "bytes_per_volume": bytes_per_volume,
                })
                continue

            # Read only the volumes that are actually present, write back as a
            # new NIfTI with corrected header dim[4]. Use load + slicing to
            # avoid trying to read the missing volumes.
            new_header = img.header.copy()
            new_dim = new_header["dim"].copy()
            new_dim[4] = actual_volumes
            new_header["dim"] = new_dim

            # Load the data we DO have. Direct memmap up to actual_volumes
            # avoids the size check that nibabel would otherwise fail on.
            raw_dtype = img.get_data_dtype()
            with open(nii_path, "rb") as f:
                f.seek(vox_offset)
                raw = f.read(actual_volumes * bytes_per_volume)
            arr = np.frombuffer(raw, dtype=raw_dtype).reshape(
                (actual_volumes, d3, d2, d1)
            ).transpose(3, 2, 1, 0)  # nibabel order: (x,y,z,t)

            new_img = nib.Nifti1Image(arr, img.affine, new_header)
            # Write to a temp NIfTI then atomic rename (nibabel needs .nii or
            # .nii.gz suffix to know the format; use .repairing.nii so the
            # extension is preserved but the file is clearly transient).
            tmp_path = nii_path.parent / (nii_path.name + ".repairing.nii")
            nib.save(new_img, str(tmp_path))
            tmp_path.replace(nii_path)

            per_file.append({
                "file": nii_path.name,
                "status": "repaired",
                "claimed_volumes": d4,
                "actual_volumes": int(actual_volumes),
                "trimmed_volumes": d4 - int(actual_volumes),
                "new_size_bytes": nii_path.stat().st_size,
            })
        except Exception as e:
            per_file.append({
                "file": nii_path.name,
                "status": "error",
                "error": str(e)[:300],
            })

    data_vol.commit()
    return {
        "subject_id": subject_id,
        "n_files": len(per_file),
        "n_repaired": sum(1 for f in per_file if f["status"] == "repaired"),
        "n_ok": sum(1 for f in per_file if f["status"] == "ok"),
        "duration_s": round(time.time() - t0, 1),
        "files": per_file,
    }


@app.local_entrypoint()
def main(subjects: str = ""):
    """Repair NIfTI headers for the given (or all pilot) subjects."""
    if subjects:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
    else:
        splits_path = Path("sapienteval/splits.json")
        if splits_path.exists():
            subject_list = json.loads(splits_path.read_text())["pilot"]["subjects"]
        else:
            subject_list = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]

    print(f"Repairing NIfTI headers for: {subject_list}")
    results = list(repair_subject.map(subject_list))
    print(json.dumps(results, indent=2))
