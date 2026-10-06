"""Read-only check: are CNeuroMod BOLD files materialized on the sapient-data volume,
or just DataLad annex pointers? Prints sizes of a few representative files + a stimuli probe.

Run: modal run mary/scripts/stat_volume.py
"""
import os
import modal

app = modal.App("mary-stat-volume")
vol = modal.Volume.from_name("sapient-data")


@app.function(volumes={"/data": vol}, timeout=300)
def stat():
    probes = [
        "/data/raw/cneuromod/fmriprep/movie10/sub-01/ses-001/func",
        "/data/raw/cneuromod/fmriprep/friends/sub-01",
    ]
    for d in probes:
        print(f"\n=== {d} ===")
        if not os.path.isdir(d):
            print("  (not a dir)")
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p)
                    islink = os.path.islink(p)
                    print(f"  {sz:>12,d}  {'LINK' if islink else 'file'}  {p}")
                except OSError as e:
                    print(f"  ERR {e}  {p}")
            break  # top level of each walked dir only
        # also descend one func dir for movie10
    # locate stimuli (movie media) anywhere under cneuromod
    print("\n=== search for stimulus media (.mkv/.mp4/.wav/.flac) under cneuromod ===")
    hits = 0
    for root, _, files in os.walk("/data/raw/cneuromod"):
        for f in files:
            if f.endswith((".mkv", ".mp4", ".wav", ".flac", ".m4a", ".avi")):
                p = os.path.join(root, f)
                try:
                    print(f"  {os.path.getsize(p):>12,d}  {p}")
                except OSError:
                    print(f"  ?            {p}")
                hits += 1
                if hits >= 25:
                    break
        if hits >= 25:
            break
    if hits == 0:
        print("  NONE FOUND — stimulus media not on volume (will need datalad get of stimuli subdataset)")


@app.local_entrypoint()
def main():
    stat.remote()
