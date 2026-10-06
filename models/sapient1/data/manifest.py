"""Build data/manifest.json describing every (subject, run, clip) tuple.

Per spec §5.1, holdout strategy for sapient-1 trained on CNeuroMod:
  - train: Friends S1–S6 episodes 1 through (last - 3), all 4 subjects;
           plus 3 of the 4 Movie10 movies.
  - val:   Friends S1–S6 last 3 episodes per subject (early stopping).
  - test:  1 held-out Movie10 movie (subject- and stimulus-novel).

Each manifest entry stores the resolved paths to every feature cache, the
fMRI cache, and metadata (subject index, TR count, total duration). The
loader (data/dataset.py) consumes this manifest verbatim — it never walks
the filesystem at train time.

Usage:
  python -m data.manifest \
      --fmri-root data/fmri/cneuromod \
      --feat-root data/features \
      --out data/manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# numpy is imported lazily inside build() so the local `modal run` parse
# (which has no numpy) doesn't fail at import time.

CNEUROMOD_CC0_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
# 3 of 4 Movie10 movies go to train; the 4th to test.
MOVIE10_TEST_HOLDOUT = "life"
VAL_EPISODES_PER_SEASON = 3   # last 3 episodes per season → val

# CNeuroMod friends episodes are scanned in segments (a/b/c). The BOLD run is
# per-segment (task-s01e01a) and so is the stimulus .mkv (friends_s01e01a.mkv),
# so we keep the segment suffix in the stim stem to pair them 1:1.
EPISODE_RE = re.compile(r"task-s(?P<season>\d{2})e(?P<ep>\d{2})(?P<seg>[a-z]?)",
                        re.IGNORECASE)
MOVIE_RE = re.compile(r"task-(?P<movie>[a-z]+)", re.IGNORECASE)


def subject_to_idx(sub: str) -> int:
    return CNEUROMOD_CC0_SUBJECTS.index(sub)


def classify_run(fmri_path: Path) -> tuple[str, dict]:
    """Return (split_kind, metadata) where split_kind is 'train'|'val'|'test'.

    split_kind is a *proposed* assignment; the final split is resolved across
    the dataset after we've seen every episode (we need the per-season episode
    counts to peel off the last 3 as val).
    """
    name = fmri_path.name
    sub = fmri_path.parent.name
    if "_task-" not in name:
        return "skip", {}

    if (m := EPISODE_RE.search(name)):
        season = int(m["season"])
        ep = int(m["ep"])
        seg = (m["seg"] or "").lower()
        return "friends_pending", {
            "subject": sub, "season": season, "episode": ep,
            "segment": seg, "task": "friends"
        }
    if (m := MOVIE_RE.search(name)):
        movie = m["movie"].lower()
        kind = "test" if movie == MOVIE10_TEST_HOLDOUT else "train"
        return kind, {"subject": sub, "movie": movie, "task": "movie10"}
    return "skip", {}


def feature_path(feat_root: Path, encoder: str, dataset: str, stim_stem: str) -> Path | None:
    """Resolve a feature cache path for the given encoder/stim. Returns None if missing."""
    cand = feat_root / encoder / dataset / f"{stim_stem}.npy"
    return cand if cand.exists() else None


def build(fmri_root: Path, feat_root: Path) -> dict:
    import numpy as np
    fmri_paths = sorted(fmri_root.rglob("*.npy"))
    print(f"Scanning {len(fmri_paths)} fMRI files under {fmri_root}")

    entries: list[dict] = []
    friends_runs: list[dict] = []      # provisional, resolved after we see all

    for fmri in fmri_paths:
        kind, meta = classify_run(fmri)
        if kind == "skip":
            continue

        # Pair each fMRI run with its stimulus features by task identifier.
        # The stimulus stem is the BIDS task suffix (e.g. "task-s01e01" or
        # "task-bourne"). Feature caches live under
        # data/features/<encoder>/<dataset>/<stim_stem>.npy.
        dataset = fmri_root.name      # "cneuromod"
        if meta["task"] == "friends":
            stim_stem = f"s{meta['season']:02d}e{meta['episode']:02d}{meta.get('segment', '')}"
        else:
            stim_stem = meta["movie"]

        feats = {
            "video": feature_path(feat_root, "vjepa2", dataset, stim_stem),
            "audio": feature_path(feat_root, "w2vbert", dataset, stim_stem),
            "text":  feature_path(feat_root, "llama",   dataset, stim_stem),
        }
        # Text is OPTIONAL (AV-only first-light): only video + audio are
        # required. A missing text feature → text_path=None and the dataset
        # zero-fills it (modality_dropout trains the model to tolerate this).
        missing = [k for k, v in feats.items() if v is None and k != "text"]
        if missing:
            print(f"  skip {fmri.name}: missing features {missing}")
            continue

        try:
            n_trs = int(np.load(fmri, mmap_mode="r").shape[0])
        except Exception as e:
            print(f"  skip {fmri.name}: cannot read shape ({e})")
            continue

        entry = {
            "subject": meta["subject"],
            "subject_idx": subject_to_idx(meta["subject"]),
            "task": meta["task"],
            "stim": stim_stem,
            "fmri_path": str(fmri),
            "video_path": str(feats["video"]),
            "audio_path": str(feats["audio"]),
            "text_path":  str(feats["text"]) if feats["text"] is not None else None,
            "n_trs": n_trs,
            "split": kind,
        }
        if kind == "friends_pending":
            friends_runs.append(entry)
        else:
            entries.append(entry)

    # Resolve Friends split: per (subject, season), the last 3 episodes → val,
    # the rest → train.
    by_subseason: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for e in friends_runs:
        season = int(e["stim"][1:3])    # "s01e02" → 1
        by_subseason[(e["subject"], season)].append(e)
    for key, runs in by_subseason.items():
        runs.sort(key=lambda r: r["stim"])
        for r in runs[:-VAL_EPISODES_PER_SEASON]:
            r["split"] = "train"
        for r in runs[-VAL_EPISODES_PER_SEASON:]:
            r["split"] = "val"
    entries.extend(friends_runs)

    counts = defaultdict(int)
    for e in entries:
        counts[e["split"]] += 1
    return {
        "subjects": CNEUROMOD_CC0_SUBJECTS,
        "n_subjects": len(CNEUROMOD_CC0_SUBJECTS),
        "counts": dict(counts),
        "entries": entries,
    }


def _write_manifest(fmri_root: Path, feat_root: Path, out: Path) -> dict:
    """Build + write. Reused by local CLI and Modal entry points."""
    manifest = build(fmri_root, feat_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out}")
    print(f"  total entries: {len(manifest['entries'])}")
    print(f"  per split:     {manifest['counts']}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fmri-root", type=Path, required=True)
    ap.add_argument("--feat-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/manifest.json"))
    args = ap.parse_args()
    _write_manifest(args.fmri_root, args.feat_root, args.out)


# ============================================================
# Modal wrapper — manifest reads /data/fmri/* and /data/features/* on the
# shared sapient-data volume, which only exists inside Modal. Local CLI
# still works against any path the local Python can see.
# ============================================================

import modal  # noqa: E402

APP_NAME = "sapient-1-manifest"

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
    fmri_root: str = "/data/fmri/cneuromod",
    feat_root: str = "/data/features",
    out: str = "/data/manifest.json",
) -> dict:
    """Modal wrapper. Reads + writes the sapient-data volume."""
    manifest = _write_manifest(Path(fmri_root), Path(feat_root), Path(out))
    _volume.commit()
    return {
        "n_entries": len(manifest["entries"]),
        "counts": manifest["counts"],
        "subjects": manifest["subjects"],
    }


@_app.local_entrypoint()
def modal_main(
    fmri_root: str = "/data/fmri/cneuromod",
    feat_root: str = "/data/features",
    out: str = "/data/manifest.json",
) -> None:
    """`modal run data/manifest.py`"""
    result = build_remote.remote(fmri_root, feat_root, out)
    print(f"\nDone. {result}")


if __name__ == "__main__":
    main()
