"""Generate synthetic fMRI + frozen-encoder features for end-to-end smoke tests.

Writes to `data/{fmri,features}/synthetic/...` and a top-level
`data/manifest.json`. Dimensions match production (20,484 vertices,
1280/1024/3072 feature dims, 2 Hz stimulus, 1 Hz fMRI), so train.py / eval.py
/ release.py exercise the exact same code paths they would against real data.

Layout produced:
  data/
    fmri/synthetic/sub-00/run-0.npy ... sub-01/run-3.npy
    features/vjepa2/synthetic/sub-XX_run-YY.npy
    features/w2vbert/synthetic/sub-XX_run-YY.npy
    features/llama/synthetic/sub-XX_run-YY.npy
    parcellate.npz                 (sparse 1000x20484 random partition)
    manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

N_SUBJECTS = 2
RUNS_PER_SUBJECT = 4
N_TRS = 150            # 150 s at 1 Hz; allows 100-TR window + 5-TR HRF offset
STIM_RATE_HZ = 2
N_VERTICES = 20484
D_VIDEO = 1280
D_AUDIO = 1024
D_TEXT  = 3072
N_PARCELS = 1000


def _signal_targets(video: np.ndarray, audio: np.ndarray,
                    text: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build fMRI targets correlated with the stimulus features.

    Real fMRI is a noisy nonlinear function of stimulus. We use a fixed
    random linear map per modality plus noise so train.py loss can actually
    decrease on this data — proving the gradient signal flows end-to-end.
    """
    # 2 Hz → 1 Hz mean-pool first.
    pooled_v = video.reshape(-1, 2, D_VIDEO).mean(axis=1)   # (N_TRS, D_VIDEO)
    pooled_a = audio.reshape(-1, 2, D_AUDIO).mean(axis=1)   # (N_TRS, D_AUDIO)
    pooled_t = text.reshape(-1, 2, D_TEXT).mean(axis=1)     # (N_TRS, D_TEXT)
    Wv = rng.standard_normal((D_VIDEO, N_VERTICES)) * 0.01
    Wa = rng.standard_normal((D_AUDIO, N_VERTICES)) * 0.01
    Wt = rng.standard_normal((D_TEXT,  N_VERTICES)) * 0.01
    signal = pooled_v @ Wv + pooled_a @ Wa + pooled_t @ Wt
    noise = rng.standard_normal((N_TRS, N_VERTICES)) * 0.5
    return (signal + noise).astype(np.float32)


def make(data_root: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fmri_root = data_root / "fmri" / "synthetic"
    feat_roots = {
        "video":   data_root / "features" / "vjepa2"  / "synthetic",
        "audio":   data_root / "features" / "w2vbert" / "synthetic",
        "text":    data_root / "features" / "llama"   / "synthetic",
    }
    for p in (fmri_root, *feat_roots.values()):
        p.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for sub_i in range(N_SUBJECTS):
        sub = f"sub-{sub_i:02d}"
        (fmri_root / sub).mkdir(exist_ok=True)
        for run_i in range(RUNS_PER_SUBJECT):
            stem = f"{sub}_run-{run_i:02d}"
            video = rng.standard_normal((N_TRS * STIM_RATE_HZ, D_VIDEO)).astype(np.float32)
            audio = rng.standard_normal((N_TRS * STIM_RATE_HZ, D_AUDIO)).astype(np.float32)
            text  = rng.standard_normal((N_TRS * STIM_RATE_HZ, D_TEXT)).astype(np.float32)
            fmri  = _signal_targets(video, audio, text, rng)

            np.save(fmri_root / sub / f"run-{run_i:02d}.npy", fmri)
            np.save(feat_roots["video"] / f"{stem}.npy", video)
            np.save(feat_roots["audio"] / f"{stem}.npy", audio)
            np.save(feat_roots["text"]  / f"{stem}.npy", text)

            # Manual split assignment: 5 train / 2 val / 1 test across 8 runs.
            global_idx = sub_i * RUNS_PER_SUBJECT + run_i
            if global_idx < 5:
                split = "train"
            elif global_idx < 7:
                split = "val"
            else:
                split = "test"

            entries.append({
                "subject": sub,
                "subject_idx": sub_i,
                "task": "synthetic",
                "stim": stem,
                "fmri_path":  str(fmri_root / sub / f"run-{run_i:02d}.npy"),
                "video_path": str(feat_roots["video"] / f"{stem}.npy"),
                "audio_path": str(feat_roots["audio"] / f"{stem}.npy"),
                "text_path":  str(feat_roots["text"]  / f"{stem}.npy"),
                "n_trs": N_TRS,
                "split": split,
            })

    # Random vertex→parcel matrix so eval.py's parcel_pearson + release.py's
    # parcellate.npz upload have real bytes to point at.
    assignment = rng.integers(0, N_PARCELS, size=N_VERTICES)
    rows, cols, data = [], [], []
    for p_id in range(N_PARCELS):
        idx = np.where(assignment == p_id)[0]
        if len(idx) == 0:
            continue
        w = 1.0 / len(idx)
        rows.extend([p_id] * len(idx))
        cols.extend(idx.tolist())
        data.extend([w] * len(idx))
    M = csr_matrix((data, (rows, cols)), shape=(N_PARCELS, N_VERTICES),
                   dtype=np.float32)
    save_npz(data_root / "parcellate.npz", M)

    manifest = {
        "subjects": [f"sub-{i:02d}" for i in range(N_SUBJECTS)],
        "n_subjects": N_SUBJECTS,
        "counts": {
            "train": sum(1 for e in entries if e["split"] == "train"),
            "val":   sum(1 for e in entries if e["split"] == "val"),
            "test":  sum(1 for e in entries if e["split"] == "test"),
        },
        "entries": entries,
    }
    with (data_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote synthetic dataset under {data_root}")
    print(f"  subjects: {N_SUBJECTS}  runs/subj: {RUNS_PER_SUBJECT}  N_TRS: {N_TRS}")
    print(f"  splits: {manifest['counts']}")
    print(f"  parcellate.npz: {M.shape}, nnz={M.nnz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make(args.data_root, seed=args.seed)
