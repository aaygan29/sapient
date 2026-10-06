"""Salvage truncated-at-source ds004996 BOLD files (sub-02/03/04).

Problem (verified corrupt at source): the conversation-task BOLD NIfTIs for
sub-02, sub-03, sub-04 are truncated. Their headers claim dim[4]=560 volumes
but the on-disk body holds *fewer* — and a non-integer number under the wrong
geometry assumption, which makes nibabel / fMRIPrep choke.

Proven geometry for these runs:
  - dims:        84 x 84 x 54
  - dtype:       int16  (2 bytes/voxel)
  - bytes/vol:   84*84*54*2 = 762,048
  - header:      352 bytes (standard NIfTI-1 single-file vox_offset)

Salvage = make the file internally consistent so it loads cleanly:
  n = (filesize - 352) // 762048                # whole volumes actually present
  truncate file to exactly 352 + n*762048 bytes  # drop the partial tail volume
  rewrite header dim[4] = n                       # and pixdim/cal stay valid

We then verify with nibabel: img.shape == (84,84,54,n) and the data array
loads without error. sub-01 is untouched (it was never corrupt).

Runs whose `n` is too short to yield >=105 one-Hz TRs are simply left as-is on
disk with their (now-correct) smaller n; the manifest stage drops them.

ARCHITECTURAL RULE: everything happens on the `sapient-data` Modal volume.
Nothing is copied to a laptop.

Usage:
  modal run data/salvage_bold.py
  modal run data/salvage_bold.py --subjects "02 03 04"
"""

from __future__ import annotations

import modal

APP_NAME = "sapient-2-salvage-bold"

# Light image: just nibabel/numpy to verify after the raw byte surgery.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("nibabel==5.2.1", "numpy<2.0")
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
app = modal.App(APP_NAME)

# --- proven geometry constants ---
NX, NY, NZ = 84, 84, 54
BYTES_PER_VOXEL = 2          # int16
HEADER_BYTES = 352           # NIfTI-1 single-file standard vox_offset
BYTES_PER_VOL = NX * NY * NZ * BYTES_PER_VOXEL  # 762,048


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=30 * 60,
    volumes={"/data": volume},
)
def salvage(subjects: str = "02 03 04") -> list[dict]:
    """Salvage every conversation BOLD run for the given subjects."""
    import struct
    from pathlib import Path

    import nibabel as nib

    assert BYTES_PER_VOL == 762048, f"geometry mismatch: {BYTES_PER_VOL}"

    raw_root = Path("/data/raw/ds004996")
    results: list[dict] = []

    sub_labels = subjects.split()
    for lab in sub_labels:
        func_dir = raw_root / f"sub-{lab}" / "func"
        bolds = sorted(func_dir.glob("*_task-conversation_run-*_bold.nii"))
        if not bolds:
            results.append({"subject": lab, "error": "no BOLD runs found"})
            continue

        for path in bolds:
            filesize = path.stat().st_size
            n = (filesize - HEADER_BYTES) // BYTES_PER_VOL
            target_size = HEADER_BYTES + n * BYTES_PER_VOL
            rec: dict = {
                "file": path.name,
                "subject": lab,
                "filesize_before": filesize,
                "n_volumes": int(n),
                "target_size": target_size,
            }

            if n <= 0:
                rec["error"] = "fewer than one whole volume present"
                results.append(rec)
                continue

            # --- 1. truncate the body to a whole number of volumes ---
            with open(path, "r+b") as f:
                f.truncate(target_size)

            # --- 2. rewrite the NIfTI-1 header dim[4] (volume count) ---
            # NIfTI-1 header: `dim` is a short[8] at byte offset 40.
            #   dim[0] = number of dims (4 here), dim[4] = #volumes (index 4).
            # We read all 8 shorts, set dim[0]>=4 and dim[4]=n, write them back.
            with open(path, "r+b") as f:
                f.seek(40)
                dim = list(struct.unpack("<8h", f.read(16)))
                rec["dim_before"] = dim[:]
                # endianness guard: dim[0] should be 1..7; if absurd, try big-endian
                endian = "<"
                if not (1 <= dim[0] <= 7):
                    f.seek(40)
                    dim = list(struct.unpack(">8h", f.read(16)))
                    endian = ">"
                    rec["dim_before"] = dim[:]
                if dim[0] < 4:
                    dim[0] = 4
                dim[4] = int(n)
                f.seek(40)
                f.write(struct.pack(f"{endian}8h", *dim))
                rec["dim_after"] = dim[:]
                rec["endian"] = endian

            # --- 3. verify with nibabel ---
            try:
                img = nib.load(str(path))
                shape = tuple(int(x) for x in img.shape)
                rec["nib_shape"] = shape
                ok_shape = (len(shape) == 4 and shape[3] == n
                            and shape[0] == NX and shape[1] == NY and shape[2] == NZ)
                # force a real read of the last volume to prove the body is intact
                arr = img.dataobj[..., n - 1]
                rec["last_vol_sum"] = float(arr.sum())
                rec["verified"] = bool(ok_shape)
            except Exception as e:  # noqa: BLE001
                rec["verified"] = False
                rec["nib_error"] = repr(e)

            rec["filesize_after"] = path.stat().st_size
            results.append(rec)
            print(f"[{path.name}] n={n} verified={rec.get('verified')} "
                  f"shape={rec.get('nib_shape')}", flush=True)

    volume.commit()
    return results


@app.local_entrypoint()
def main(subjects: str = "02 03 04"):
    results = salvage.remote(subjects)
    print("\n===== SALVAGE SUMMARY =====")
    for r in results:
        print(r)
    n_ok = sum(1 for r in results if r.get("verified"))
    n_tot = sum(1 for r in results if "n_volumes" in r)
    print(f"\nverified {n_ok}/{n_tot} BOLD runs")
