"""Build data/manifest.json for sapient-2 (pooled ds004996 + ds001740 HRI corpus).

Per spec §5.1, the sapient-2 split is subject-and-stimulus-novel:
  - train: 80% subjects × 80% runs
  - val:   20% subjects × all runs       (subject-novel — generalize to new humans)
  - test:  all subjects × 20% runs       (stimulus-novel — generalize to new conversations)

Subject indexing is pooled across the two datasets: ds004996 subjects come
first, then ds001740 subjects, so subject_embed[i] resolves uniquely.

Usage:
  python -m data.manifest \
      --fmri-roots data/fmri/ds004996 data/fmri/ds001740 \
      --feat-root data/features \
      --out data/manifest.json \
      --val-frac 0.2 --test-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# numpy lives in the Modal container, not on the local machine that runs
# `modal run`. Import lazily so the LOCAL entrypoint parse doesn't need it.
try:
    import numpy as np
except ModuleNotFoundError:  # local parse only — the real run is in-container
    pass

BIDS_SUB = re.compile(r"sub-[A-Za-z0-9]+")
RUN_RE = re.compile(r"run-0*([0-9]+)")


def _run_number(name: str) -> str | None:
    """Extract the integer run number (as a normalized string) from a filename.

    fMRI stems look like ``sub-01_task-conversation_run-01`` while feature
    files look like ``sub-01_run-01`` (video) or
    ``sub-01_participant_denoised_run-01`` (audio). They share only the
    subject and the run *number*, so that's what we join on.
    """
    m = RUN_RE.search(name)
    return m.group(1) if m else None


def _index_features(feat_dir: Path, *, prefer_substr: str | None = None) -> dict[tuple[str, str], Path]:
    """Index a feature encoder dir by (subject, run_number).

    ``prefer_substr`` (e.g. ``"participant_denoised"`` for audio) disambiguates
    when several files share a subject+run (operator / raw / denoised wavs).
    """
    out: dict[tuple[str, str], Path] = {}
    if not feat_dir.exists():
        return out
    for npy in sorted(feat_dir.rglob("*.npy")):
        sub = next((p for p in npy.parts if p.startswith("sub-")), None)
        if sub is None:
            sub_m = BIDS_SUB.search(npy.stem)
            sub = sub_m.group(0) if sub_m else None
        run = _run_number(npy.stem)
        if sub is None or run is None:
            continue
        key = (sub, run)
        if prefer_substr is not None:
            # Prefer files whose name contains the substring; only fall back to
            # others if nothing preferred is found.
            is_pref = prefer_substr in npy.name
            if key in out and prefer_substr in out[key].name and not is_pref:
                continue  # already have a preferred match
            if key in out and prefer_substr not in out[key].name and not is_pref:
                continue  # keep first non-preferred; deterministic via sort
        if key not in out or (prefer_substr is not None and prefer_substr in npy.name):
            out[key] = npy
    return out


def build(fmri_roots: list[Path], feat_root: Path,
          val_frac: float, test_frac: float, seed: int,
          *, require_text: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    pooled_subjects: list[tuple[str, str]] = []  # (dataset, sub)
    all_runs: list[dict] = []

    # Discover (dataset, subject, run) tuples
    for root in fmri_roots:
        dataset = root.name
        # Pre-index features for this dataset by (subject, run_number).
        vid_idx = _index_features(feat_root / "vjepa2" / dataset)
        aud_idx = _index_features(feat_root / "w2vbert" / dataset,
                                  prefer_substr="participant_denoised")
        txt_idx = _index_features(feat_root / "llama" / dataset)

        for fmri in sorted(root.rglob("*.npy")):
            sub = next((p for p in fmri.parts if p.startswith("sub-")), None)
            if sub is None:
                continue
            run = _run_number(fmri.stem)
            if run is None:
                print(f"  skip {fmri.name}: no run number")
                continue
            if (dataset, sub) not in pooled_subjects:
                pooled_subjects.append((dataset, sub))
            stem = fmri.stem
            key = (sub, run)
            feats = {
                "video": vid_idx.get(key),
                "audio": aud_idx.get(key),
                "text":  txt_idx.get(key),
            }
            required = ["video", "audio"] + (["text"] if require_text else [])
            missing = [k for k in required if feats[k] is None]
            if missing:
                print(f"  skip {fmri.name}: missing features {missing}")
                continue
            try:
                n_trs = int(np.load(fmri, mmap_mode="r").shape[0])
            except Exception as e:
                print(f"  skip {fmri.name}: cannot read shape ({e})")
                continue
            all_runs.append({
                "dataset": dataset,
                "subject": sub,
                "fmri_path": str(fmri),
                "video_path": str(feats["video"]),
                "audio_path": str(feats["audio"]),
                # text may be None in AV-only mode; dataset.py feeds zeros.
                "text_path":  str(feats["text"]) if feats["text"] else None,
                "n_trs": n_trs,
                "stim": stem,
            })

    # Subject index lookup (pooled across datasets)
    sub_idx = {(ds, sub): i for i, (ds, sub) in enumerate(pooled_subjects)}
    for r in all_runs:
        r["subject_idx"] = sub_idx[(r["dataset"], r["subject"])]

    # Subject-novel val split: hold out val_frac of subjects entirely for val.
    all_subject_keys = list(sub_idx.keys())
    rng.shuffle(all_subject_keys)
    n_val_subs = max(1, int(round(val_frac * len(all_subject_keys))))
    val_subjects = set(all_subject_keys[:n_val_subs])
    train_subjects = set(all_subject_keys[n_val_subs:])

    # Stimulus-novel test split: within train_subjects, peel off test_frac of runs.
    by_train_sub: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_runs:
        key = (r["dataset"], r["subject"])
        if key in train_subjects:
            by_train_sub[key].append(r)

    entries: list[dict] = []
    for r in all_runs:
        key = (r["dataset"], r["subject"])
        if key in val_subjects:
            r["split"] = "val"
        # else handled below
        entries.append(r)

    # For each train subject, randomly mark test_frac of their runs as 'test'.
    for key, runs in by_train_sub.items():
        n_test = max(1, int(round(test_frac * len(runs))))
        idx = rng.choice(len(runs), size=n_test, replace=False)
        for i, r in enumerate(runs):
            r["split"] = "test" if i in idx else "train"

    counts = defaultdict(int)
    for e in entries:
        counts[e["split"]] += 1
    return {
        "subjects": [f"{ds}/{sub}" for ds, sub in pooled_subjects],
        "n_subjects": len(pooled_subjects),
        "datasets": sorted(set(ds for ds, _ in pooled_subjects)),
        "counts": dict(counts),
        "entries": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fmri-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--feat-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/manifest.json"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--av-only", action="store_true",
                    help="Build an audio+video-only manifest (skip text/llama).")
    args = ap.parse_args()
    _write_manifest(args.fmri_roots, args.feat_root, args.out,
                    args.val_frac, args.test_frac, args.seed,
                    require_text=not args.av_only)


def _write_manifest(fmri_roots: list[Path], feat_root: Path, out: Path,
                    val_frac: float, test_frac: float, seed: int,
                    *, require_text: bool = True) -> dict:
    manifest = build(fmri_roots, feat_root, val_frac, test_frac, seed,
                     require_text=require_text)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out}")
    print(f"  pooled subjects: {manifest['n_subjects']}")
    print(f"  datasets:        {manifest['datasets']}")
    print(f"  per split:       {manifest['counts']}")
    return manifest


# ============================================================
# Modal wrapper — pooled ds004996 + ds001740 manifest on the volume.
# ============================================================

import modal  # noqa: E402

APP_NAME = "sapient-2-manifest"

_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=1.26,<3",
)

_volume = modal.Volume.from_name("sapient-data", create_if_missing=True)

_app = modal.App(APP_NAME)


@_app.function(
    image=_image,
    cpu=2.0,
    memory=4096,
    timeout=30 * 60,
    volumes={"/data": _volume},
)
def build_remote(
    fmri_roots: list[str] | None = None,
    feat_root: str = "/data/features",
    out: str = "/data/manifest.json",
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
    av_only: bool = False,
) -> dict:
    if fmri_roots is None:
        fmri_roots = ["/data/fmri/ds004996", "/data/fmri/ds001740"]
    roots = [Path(p) for p in fmri_roots]
    manifest = _write_manifest(roots, Path(feat_root), Path(out),
                               val_frac, test_frac, seed,
                               require_text=not av_only)
    _volume.commit()
    return {
        "n_subjects": manifest["n_subjects"],
        "datasets": manifest["datasets"],
        "counts": manifest["counts"],
    }


@_app.local_entrypoint()
def modal_main(
    feat_root: str = "/data/features",
    out: str = "/data/manifest.json",
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
    av_only: bool = False,
    fmri_roots: str = "/data/fmri/ds004996 /data/fmri/ds001740",
) -> None:
    """`modal run data/manifest.py`

    AV-only ds004996 manifest:
      modal run data/manifest.py --av-only \
          --fmri-roots /data/fmri/ds004996 --out /data/manifest_ds004996_av.json
    """
    roots = fmri_roots.split()
    result = build_remote.remote(
        roots, feat_root, out, val_frac, test_frac, seed, av_only,
    )
    print(f"\nDone. {result}")


if __name__ == "__main__":
    main()
