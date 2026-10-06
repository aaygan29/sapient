"""One-off: scale + ISC-feasibility stats for the multi-dataset Mary manifest.

modal run scripts/_inspect_manifest.py
"""
from __future__ import annotations

import modal

app = modal.App("mary-inspect-manifest")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26,<3")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)


@app.function(image=image, cpu=1.0, timeout=10 * 60, volumes={"/data": vol})
def run() -> dict:
    import json
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    p = Path("/data/manifest_mary_multi.json")
    m = json.loads(p.read_text())
    entries = m.get("entries", [])

    # Group by (dataset, story/run) — same stimulus across subjects.
    def story_key(e):
        return (e.get("dataset"), e.get("story", e.get("run", e.get("run_key"))))

    groups = defaultdict(list)
    for e in entries:
        groups[story_key(e)].append(e)

    # How many stories are shared by >=2 subjects (ISC-eligible)?
    shareable = {k: v for k, v in groups.items() if len({e["subject_idx"] for e in v}) >= 2}
    by_ds_shared = Counter(k[0] for k in shareable)
    by_ds_total = Counter(k[0] for k in groups)

    # For a few shareable groups, check fMRI length agreement across subjects.
    len_samples = []
    for k, v in list(shareable.items())[:12]:
        lens = []
        for e in v[:6]:
            try:
                arr = np.load(e["fmri_path"], mmap_mode="r")
                lens.append(int(arr.shape[0]))
            except Exception as ex:
                lens.append(f"err:{ex}")
        len_samples.append({"group": f"{k[0]}:{k[1]}", "n_subj": len({e['subject_idx'] for e in v}),
                            "fmri_lens": lens})

    # Total subject-pairs available for ISC.
    isc_pairs = sum(len({e["subject_idx"] for e in v}) for v in shareable.values())

    return {
        "n_story_groups": len(groups),
        "n_shareable_groups(>=2 subj)": len(shareable),
        "shareable_by_dataset": dict(by_ds_shared),
        "total_groups_by_dataset": dict(by_ds_total),
        "isc_subject_runs_in_shareable": isc_pairs,
        "length_check_samples": len_samples,
    }


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=2))
