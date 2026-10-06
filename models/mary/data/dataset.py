"""MaryDataset — multimodal windows aligned to fMRI, for the ds002345/Huth slice.

CONTRACTS.md §2 (tensor-shape spine), §4 (manifest schema), §6 (MaryModel API).

One __getitem__ returns a dict compatible with
``MaryModel.forward(features, subject_idx)``:

    features:    dict[stream -> (T_2Hz, D_m)]   # ONLY streams present for the entry
    fmri:        (T_TR, 20484)                  # z-scored fsaverage5 vertices
    subject_idx: long                           # per-subject head/embedding index
    mask:        (T_TR,)                         # 1 where fMRI is valid

Temporal convention (family + CONTRACTS §2): stream features at 2 Hz, fMRI at
1 Hz. A window is `sequence_length` TRs (default 100 → T_2Hz=200, T_TR=100).
HRF offset = 5 TRs (decision #8): the fMRI window starts 5 TR after the stim
window. Concretely for window start `s` (in TR units):
    stim slice = [s*2, (s+win)*2)            # 2 Hz feature grid
    fmri slice = [s+hrf, s+hrf+win)          # 1 Hz fMRI grid

Missing streams: an entry's `feature_paths` only lists streams whose `.npy`
exists; this loader returns exactly those (MaryModel zeros/masks the rest).

Manifest (`/data/manifest_mary.json`) is consumed verbatim — no filesystem
walks at train time. Arrays are read via numpy memmaps to keep the working set
tiny.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

HRF_OFFSET_TRS = 5      # CONTRACTS / decision #8
WINDOW_TRS = 100        # sequence_length default (TR window) → T_2Hz = 200
STIM_RATE_HZ = 2
FMRI_RATE_HZ = 1
N_VERTICES = 20484


class MaryDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: Literal["train", "val", "test"],
        sequence_length: int = WINDOW_TRS,
        hrf_offset_trs: int = HRF_OFFSET_TRS,
        seed: int | None = 13,
        oversample: int = 1,
    ):
        # train: draw `oversample` random windows per entry per epoch (each
        # (subject,story) run holds many distinct 100-TR windows). This turns a
        # small POC cohort into a meaningful number of optimization steps.
        self.oversample = max(1, int(oversample))
        with Path(manifest_path).open() as f:
            manifest = json.load(f)

        # `sequence_length` in the config is the TR window (CONTRACTS §5 default
        # 200 is the stim-token count; the family's TR window is 100). Accept
        # either: if > max plausible TR window, treat it as the stim length.
        win = int(sequence_length)
        if win > 200:
            win = win // STIM_RATE_HZ
        # If passed the stim-length (200), convert to TR window (100).
        if win >= 200:
            win = win // STIM_RATE_HZ

        self.entries = [e for e in manifest["entries"] if e["split"] == split]
        self.n_subjects = manifest["n_subjects"]
        self.streams = manifest.get(
            "streams",
            ["slowfast", "qwen_vl", "beats", "whisper", "qwen_ctx", "got_ocr"],
        )
        self.duration_trs = win
        self.hrf_offset_trs = int(hrf_offset_trs)
        self.split = split
        self._rng = np.random.default_rng(seed)
        self._mmap_cache: dict[str, np.ndarray] = {}

        if split != "train":
            self._eval_windows = self._build_eval_windows()

    # ----- memmap helpers -------------------------------------------------
    def _mmap(self, path: str) -> np.ndarray:
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.load(path, mmap_mode="r")
        return self._mmap_cache[path]

    def _max_start(self, entry: dict) -> int:
        """Largest valid window start (in TR units) for an entry."""
        return entry["n_trs"] - self.duration_trs - self.hrf_offset_trs

    def _build_eval_windows(self) -> list[tuple[int, int]]:
        """Deterministic non-overlapping windows: (entry_idx, start_tr)."""
        out: list[tuple[int, int]] = []
        for i, e in enumerate(self.entries):
            usable = self._max_start(e)
            if usable < 0:
                continue
            for start in range(0, usable + 1, self.duration_trs):
                out.append((i, start))
        return out

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.entries) * self.oversample
        return len(self._eval_windows)

    def sample_weights(self):
        """Per-sample weights for a WeightedRandomSampler (dataset-balanced sampling).
        Uses each entry's manifest `sample_weight` (∝ 1/#train-entries-in-its-dataset),
        repeated across the oversampled index space. Train split only."""
        n = len(self.entries)
        if n == 0:
            return []
        w = [float(e.get("sample_weight", 1.0)) or 1.0 for e in self.entries]
        return [w[i % n] for i in range(len(self))]

    # ----- item -----------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        if self.split == "train":
            entry = self.entries[idx % len(self.entries)]
            max_start = self._max_start(entry)
            start_tr = 0 if max_start < 0 else int(self._rng.integers(0, max_start + 1))
        else:
            entry_i, start_tr = self._eval_windows[idx]
            entry = self.entries[entry_i]

        win = self.duration_trs
        stim_s = start_tr * STIM_RATE_HZ
        stim_e = (start_tr + win) * STIM_RATE_HZ

        # Present streams only (CONTRACTS: skip missing).
        features: dict[str, torch.Tensor] = {}
        for stream, path in entry.get("feature_paths", {}).items():
            arr = self._mmap(path)
            chunk = np.asarray(arr[stim_s:stim_e])
            # Pad along time if the feature array is shorter than the window.
            if chunk.shape[0] < (stim_e - stim_s):
                pad = (stim_e - stim_s) - chunk.shape[0]
                chunk = np.concatenate(
                    [chunk, np.zeros((pad, chunk.shape[1]), dtype=chunk.dtype)],
                    axis=0,
                )
            features[stream] = torch.from_numpy(
                np.ascontiguousarray(chunk)
            ).float()

        # fMRI window with HRF offset.
        fmri_s = start_tr + self.hrf_offset_trs
        fmri_e = fmri_s + win
        fmri_arr = self._mmap(entry["fmri_path"])
        fmri = np.asarray(fmri_arr[fmri_s:fmri_e])
        mask = np.ones(win, dtype=np.float32)
        if fmri.shape[0] < win:
            pad = win - fmri.shape[0]
            fmri = np.concatenate(
                [fmri, np.zeros((pad, fmri.shape[1]), dtype=fmri.dtype)], axis=0
            )
            mask[win - pad:] = 0.0

        return {
            "features": features,
            "fmri": torch.from_numpy(np.ascontiguousarray(fmri)).float(),
            "subject_idx": torch.tensor(entry["subject_idx"], dtype=torch.long),
            "mask": torch.from_numpy(mask),
        }


def collate_mary(batch: list[dict]) -> dict:
    """Collate variable-stream items into a batch for MaryModel.forward.

    Streams present in ANY sample are stacked; samples missing a stream get a
    zero tensor for it (matching MaryModel's absent-stream handling, which
    additionally tracks presence via the modality masks during training).
    """
    streams: set[str] = set()
    for b in batch:
        streams.update(b["features"].keys())

    # Reference shapes from the first sample that has each stream.
    feat_batch: dict[str, torch.Tensor] = {}
    for s in sorted(streams):
        ref = next(b["features"][s] for b in batch if s in b["features"])
        T, D = ref.shape
        stacked = torch.zeros(len(batch), T, D, dtype=torch.float32)
        for i, b in enumerate(batch):
            if s in b["features"]:
                stacked[i] = b["features"][s]
        feat_batch[s] = stacked

    return {
        "features": feat_batch,
        "fmri": torch.stack([b["fmri"] for b in batch]),
        "subject_idx": torch.stack([b["subject_idx"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }
