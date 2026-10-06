"""Build /data/manifest_mary.json for the ds002345 ("Narratives"/Huth) slice.

Schema = CONTRACTS.md §4, consumed verbatim by data/dataset.py.

Story-level feature sharing (CONTRACTS.md, WS coordination): audio/text features
are identical across subjects for a given story, so they live ONCE at
    /data/features/mary/huth/_stories/{story}/{stream}.npy
and every (subject, story) entry references those shared paths plus the
subject's own fMRI at
    /data/fmri/mary/huth/{subject}/{story}.npy

This builder is tolerant of missing stream files: it records only streams whose
`.npy` exists right now (the feature-extraction agent writes them separately,
possibly after this manifest is first built). Active streams this sprint:
whisper, beats, qwen_ctx; the video streams (slowfast, qwen_vl, got_ocr) have no
open stimulus media and will simply be absent → handled by Mary's missing-stream
+ modality-dropout logic.

Splits:
  - test: the ONE story present across the most subjects (the OOD held-out for
          Papers A/B noise-ceiling + per-subject-vs-average eval).
  - train/val: the remaining (subject, story) pairs split ~90/10, deterministic.

Usage (Modal):  modal run data/manifest.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import modal

# CONTRACTS.md §2/§4: canonical 6-stream order (manifest advertises all 6; each
# entry records only the streams whose .npy currently exists on the volume).
ALL_STREAMS = ["slowfast", "qwen_vl", "beats", "whisper", "qwen_ctx", "got_ocr"]

DATASET = "huth"
FMRI_ROOT_DEFAULT = "/data/fmri/mary/huth"
FEAT_STORIES_ROOT_DEFAULT = "/data/features/mary/huth/_stories"
OUT_DEFAULT = "/data/manifest_mary.json"

VAL_FRACTION = 0.10
SEED = 13                                  # CONTRACTS §5 seed


def shared_feature_paths(stories_root: Path, story: str) -> dict[str, str]:
    """Return {stream: path} for the shared per-story feature files that EXIST.

    Missing streams are simply omitted (the feature agent may not have written
    them yet, or the modality is inactive this sprint).
    """
    out: dict[str, str] = {}
    sdir = stories_root / story
    for stream in ALL_STREAMS:
        cand = sdir / f"{stream}.npy"
        if cand.exists():
            out[stream] = str(cand)
    return out


def build(fmri_root: Path, stories_root: Path,
          subjects_filter: set | None = None,
          force_test_story: str | None = None) -> dict:
    """Scan the per-subject fMRI arrays + shared story features → manifest dict.

    subjects_filter: if given, keep only these subjects (clean cohort design).
    force_test_story: if given, hold out this story as the OOD `test` split
        (instead of auto-picking the most-shared story).
    """
    import numpy as np

    # Discover every (subject, story) fMRI array we projected.
    fmri_files = sorted(fmri_root.glob("sub-*/*.npy"))
    print(f"Scanning {len(fmri_files)} fMRI arrays under {fmri_root}")

    # (subject, story) -> {fmri_path, n_trs}
    pairs: list[dict] = []
    story_subjects: dict[str, set] = defaultdict(set)
    subjects_seen: set[str] = set()
    for f in fmri_files:
        subject = f.parent.name           # "sub-001"
        story = f.stem                     # "pieman"
        if subjects_filter and subject not in subjects_filter:
            continue
        try:
            n_trs = int(np.load(f, mmap_mode="r").shape[0])
        except Exception as e:
            print(f"  skip {f}: cannot read shape ({e})")
            continue
        pairs.append({"subject": subject, "story": story,
                      "fmri_path": str(f), "n_trs": n_trs})
        story_subjects[story].add(subject)
        subjects_seen.add(subject)

    if not pairs:
        return {"n_subjects": 0, "streams": ALL_STREAMS, "entries": []}

    # Stable subject_idx assignment (sorted subject id → index).
    subjects_sorted = sorted(subjects_seen)
    subject_idx = {s: i for i, s in enumerate(subjects_sorted)}

    # OOD test story: forced, or the one present across the MOST subjects.
    if force_test_story and force_test_story in story_subjects:
        test_story = force_test_story
    else:
        test_story = max(story_subjects, key=lambda s: (len(story_subjects[s]), s))
    print(f"OOD test story = {test_story!r} "
          f"({len(story_subjects[test_story])} subjects)")

    # Deterministic ~90/10 train/val split over the non-test pairs.
    rng = np.random.default_rng(SEED)
    nontest = [p for p in pairs if p["story"] != test_story]
    order = rng.permutation(len(nontest))
    n_val = max(1, int(round(len(nontest) * VAL_FRACTION))) if nontest else 0
    val_positions = set(order[:n_val].tolist())

    entries: list[dict] = []
    counts = defaultdict(int)
    for p in pairs:
        if p["story"] == test_story:
            split = "test"
        else:
            pos = nontest.index(p)
            split = "val" if pos in val_positions else "train"
        feats = shared_feature_paths(stories_root, p["story"])
        entries.append({
            "dataset": DATASET,
            "subject": p["subject"],
            "run": p["story"],                  # one stimulus == one story/run
            "story": p["story"],
            "subject_idx": subject_idx[p["subject"]],
            "n_trs": p["n_trs"],
            "split": split,
            "feature_paths": feats,             # only streams that exist
            "fmri_path": p["fmri_path"],
        })
        counts[split] += 1

    manifest = {
        "dataset": DATASET,
        "n_subjects": len(subjects_sorted),
        "subjects": subjects_sorted,
        "subject_idx": subject_idx,
        "streams": ALL_STREAMS,
        "feature_stories_root": str(stories_root),
        "test_story": test_story,
        "counts": dict(counts),
        "entries": entries,
    }
    return manifest


def _write(fmri_root: Path, stories_root: Path, out: Path,
           subjects_filter: set | None = None,
           force_test_story: str | None = None) -> dict:
    manifest = build(fmri_root, stories_root, subjects_filter, force_test_story)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out}")
    print(f"  n_subjects: {manifest['n_subjects']}")
    print(f"  entries:    {len(manifest['entries'])}")
    print(f"  per split:  {manifest.get('counts')}")
    print(f"  test story: {manifest.get('test_story')}")
    # report stream availability
    n_with = defaultdict(int)
    for e in manifest["entries"]:
        for s in e["feature_paths"]:
            n_with[s] += 1
    print(f"  entries with each stream: {dict(n_with)}")
    return manifest


# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------
APP_NAME = "mary-manifest"

_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26,<3")
_volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
_app = modal.App(APP_NAME)


@_app.function(image=_image, cpu=2.0, memory=4096, timeout=30 * 60,
               volumes={"/data": _volume})
def build_remote(
    fmri_root: str = FMRI_ROOT_DEFAULT,
    stories_root: str = FEAT_STORIES_ROOT_DEFAULT,
    out: str = OUT_DEFAULT,
    subjects: str = "",
    test_story: str = "",
) -> dict:
    subj = {s for s in subjects.split(",") if s} or None
    manifest = _write(Path(fmri_root), Path(stories_root), Path(out),
                      subjects_filter=subj, force_test_story=test_story or None)
    _volume.commit()
    return {
        "n_subjects": manifest["n_subjects"],
        "n_entries": len(manifest["entries"]),
        "counts": manifest.get("counts"),
        "test_story": manifest.get("test_story"),
        "subjects": manifest.get("subjects"),
    }


@_app.local_entrypoint()
def main(
    fmri_root: str = FMRI_ROOT_DEFAULT,
    stories_root: str = FEAT_STORIES_ROOT_DEFAULT,
    out: str = OUT_DEFAULT,
    subjects: str = "",
    test_story: str = "",
) -> None:
    """`modal run data/manifest.py [--subjects a,b] [--test-story forgot]`"""
    res = build_remote.remote(fmri_root, stories_root, out, subjects, test_story)
    print("\n========== SUMMARY ==========")
    print(json.dumps(res, indent=2))


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fmri-root", default=FMRI_ROOT_DEFAULT)
    ap.add_argument("--stories-root", default=FEAT_STORIES_ROOT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    raise SystemExit(
        "This script runs on Modal; use `modal run data/manifest.py`. "
        f"(args={args})"
    )


if __name__ == "__main__":
    _cli()
