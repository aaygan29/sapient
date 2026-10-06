"""Multi-dataset manifest builder for Mary (the faithful, pipeline-working-together step).

Scans /data/fmri/mary/{dataset}/{subject}/{run}.npy across several datasets and pairs
each with that run's shared per-story features at
/data/features/mary/{dataset}/_stories/{run}/{stream}.npy (records only streams present).

Global subject_idx is keyed by "{dataset}:{subject}" (a subject in CNeuroMod != a subject in
HAD — distinct per-subject heads, which is correct). Adds per-dataset + per-entry sampling
weights so a big dataset doesn't drown small ones (dataset-balanced sampling).

Usage:
  modal run data/build_manifest_multi.py --datasets huth,lebel2023 --test-spec huth:forgot
  modal run data/build_manifest_multi.py --datasets huth,lebel2023,cneuromod,had,wen2017 \
      --test-spec huth:forgot --out /data/manifest_mary_multi.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import modal

ALL_STREAMS = ["slowfast", "qwen_vl", "beats", "whisper", "qwen_ctx", "got_ocr"]
FMRI_ROOT = "/data/fmri/mary"
FEAT_ROOT = "/data/features/mary"

app = modal.App("mary-manifest-multi")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26,<3")


# Datasets whose features are assembled PER (subject, run) rather than shared
# per-story. HAD is event-based: each run is an independent timeline of clips, so
# features live at {ds}/{subject}/{run}/{stream}.npy (see assemble_had_features.py),
# NOT at the shared {ds}/_stories/{story}/... used by the continuous-narrative
# datasets (huth, lebel2023, cneuromod).
PER_SUBJECT_FEATURE_DATASETS = {"had"}


def _feature_paths(ds: str, run: str, subject: str | None = None) -> dict:
    """Return {stream: path} for the features paired with this (ds, subject, run).

    Story datasets share features across subjects at {ds}/_stories/{run}/.
    Per-subject datasets (HAD) key features by (subject, run): {ds}/{subject}/{run}/.
    """
    if ds in PER_SUBJECT_FEATURE_DATASETS and subject is not None:
        sdir = Path(FEAT_ROOT) / ds / subject / run
    else:
        sdir = Path(FEAT_ROOT) / ds / "_stories" / run
    out = {}
    for s in ALL_STREAMS:
        p = sdir / f"{s}.npy"
        if p.exists():
            out[s] = str(p)
    return out


@app.function(image=image, cpu=2.0, timeout=30 * 60, volumes={"/data": vol})
def build(datasets: str, test_spec: str = "huth:forgot",
          out: str = "/data/manifest_mary_multi.json") -> dict:
    import numpy as np

    ds_list = [d for d in datasets.split(",") if d]
    # test_spec: comma list of "dataset:pattern" held out as OOD test. `pattern`
    # matches a run if it equals OR is a substring of the run name — so
    # "cneuromod:bourne" holds out ALL Bourne segments (movie10_bourne01..10),
    # giving a clean OOD that Mary never trains on (fair vs zero-shot TRIBE v2).
    holdout = []
    for tok in test_spec.split(","):
        if ":" in tok:
            holdout.append(tuple(tok.split(":", 1)))

    def _held(ds: str, run: str) -> bool:
        return any(hds == ds and (pat == run or pat in run) for hds, pat in holdout)

    pairs = []
    subj_ids = {}
    for ds in ds_list:
        froot = Path(FMRI_ROOT) / ds
        if not froot.exists():
            print(f"  [skip] no fMRI dir for {ds}")
            continue
        # Subject dir names vary by dataset: cneuromod/had/lebel2023/huth use
        # `sub-*`, Wen2017 uses `subject1`. Match ANY subject dir (each holds
        # per-run `.npy`), not just `sub-*`, so Wen2017 is included.
        for f in sorted(froot.glob("*/*.npy")):
            subject, run = f.parent.name, f.stem
            key = f"{ds}:{subject}"
            subj_ids.setdefault(key, len(subj_ids))
            try:
                n_trs = int(np.load(f, mmap_mode="r").shape[0])
            except Exception as e:
                print(f"  skip {f}: {e}"); continue
            feats = _feature_paths(ds, run, subject)
            if not feats:
                continue  # no features yet for this run → not trainable
            pairs.append({"dataset": ds, "subject": subject, "run": run,
                          "subject_idx": subj_ids[key], "n_trs": n_trs,
                          "feature_paths": feats, "fmri_path": str(f)})

    if not pairs:
        print("NO trainable pairs found."); return {"n_entries": 0}

    # splits: holdout -> test; rest deterministic 90/10 train/val
    rng = np.random.default_rng(13)
    nontest = [p for p in pairs if not _held(p["dataset"], p["run"])]
    order = rng.permutation(len(nontest))
    n_val = max(1, int(round(len(nontest) * 0.10)))
    val_pos = set(order[:n_val].tolist())

    # dataset-balanced weights: per-entry weight ∝ 1 / (#train entries in its dataset)
    ds_counts = defaultdict(int)
    for i, p in enumerate(nontest):
        if i not in val_pos:
            ds_counts[p["dataset"]] += 1

    entries = []
    counts = defaultdict(int)
    for p in pairs:
        if _held(p["dataset"], p["run"]):
            split = "test"; w = 0.0
        else:
            pos = nontest.index(p)
            split = "val" if pos in val_pos else "train"
            w = (1.0 / ds_counts[p["dataset"]]) if (split == "train" and ds_counts[p["dataset"]]) else 0.0
        entries.append({**p, "split": split, "sample_weight": w})
        counts[split] += 1

    manifest = {
        "datasets": ds_list, "n_subjects": len(subj_ids),
        "subject_index": subj_ids, "streams": ALL_STREAMS,
        "holdout_test": sorted(":".join(h) for h in holdout),
        "counts": dict(counts), "entries": entries,
    }
    Path(out).write_text(json.dumps(manifest, indent=2))
    vol.commit()
    # stream availability + per-dataset entry counts
    by_ds = defaultdict(int)
    for e in entries:
        by_ds[e["dataset"]] += 1
    print(f"Wrote {out}: n_subjects={len(subj_ids)} entries={len(entries)} "
          f"splits={dict(counts)} per_dataset={dict(by_ds)}")
    return {"n_subjects": len(subj_ids), "n_entries": len(entries),
            "counts": dict(counts), "per_dataset": dict(by_ds), "out": out}


@app.local_entrypoint()
def main(datasets: str, test_spec: str = "huth:forgot",
         out: str = "/data/manifest_mary_multi.json"):
    print(json.dumps(build.remote(datasets, test_spec, out), indent=2))
