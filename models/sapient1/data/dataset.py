"""SapientDataset — 100-second multimodal windows aligned to fMRI labels.

Per spec §2.4. One __getitem__ returns:

    video:   (200, 1280)    # 100 s × 2 Hz, V-JEPA features
    audio:   (200, 1024)    # 100 s × 2 Hz, W2V-BERT features
    text:    (200, 3072)    # 100 s × 2 Hz, Llama-3.2-3B features
    fmri:    (100, 20484)   # 100 s × 1 Hz, fsaverage5 vertices
    subject: long           # subject index for the embedding lookup
    mask:    (100,)         # 1 where fMRI is valid, 0 where padded

HRF offset (decision #8): the fMRI window starts 5 TR after the stimulus
window. Concretely: stim slice = [start_tr*2, (start_tr+100)*2),
fmri slice = [start_tr+5, start_tr+105).

Manifest is consumed verbatim from data/manifest.json — no filesystem walks
at training time. All feature/fMRI arrays are read via numpy memmaps to keep
the working set tiny.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


HRF_OFFSET_TRS = 5    # decision #8
WINDOW_TRS = 100      # spec §2.4 / decision #5
STIM_RATE_HZ = 2
FMRI_RATE_HZ = 1


class SapientDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: Literal["train", "val", "test"],
        duration_trs: int = WINDOW_TRS,
        hrf_offset_trs: int = HRF_OFFSET_TRS,
        seed: int | None = None,
        text_dim: int = 3072,
    ):
        with Path(manifest_path).open() as f:
            manifest = json.load(f)
        self.entries = [e for e in manifest["entries"] if e["split"] == split]
        self.n_subjects = manifest["n_subjects"]
        self.duration_trs = duration_trs
        self.hrf_offset_trs = hrf_offset_trs
        self.split = split
        # AV-only manifests may have text_path=None. We feed a zero tensor of
        # width text_dim so proj_text contributes only its bias (modality_dropout
        # already trains the model to tolerate a missing modality).
        self.text_dim = text_dim

        # Train: random window per __getitem__. Val/test: deterministic stride
        # to cover the full run with non-overlapping windows.
        self._rng = np.random.default_rng(seed)

        # Lazy memmaps cached by path.
        self._mmap_cache: dict[str, np.ndarray] = {}

        if split != "train":
            self._eval_windows = self._build_eval_windows()

    def _mmap(self, path: str) -> np.ndarray:
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.load(path, mmap_mode="r")
        return self._mmap_cache[path]

    def _build_eval_windows(self) -> list[tuple[int, int]]:
        """Non-overlapping windows: (entry_idx, start_tr). Last one truncated."""
        out: list[tuple[int, int]] = []
        for i, e in enumerate(self.entries):
            usable = e["n_trs"] - self.duration_trs - self.hrf_offset_trs
            if usable < 0:
                continue
            for start in range(0, usable + 1, self.duration_trs):
                out.append((i, start))
        return out

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.entries)
        return len(self._eval_windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.split == "train":
            entry = self.entries[idx]
            n_trs = entry["n_trs"]
            max_start = n_trs - self.duration_trs - self.hrf_offset_trs
            if max_start < 0:
                # Runs shorter than 105 TRs should never appear in manifests.
                start_tr = 0
            else:
                start_tr = int(self._rng.integers(0, max_start + 1))
        else:
            entry_i, start_tr = self._eval_windows[idx]
            entry = self.entries[entry_i]

        stim_s = start_tr * STIM_RATE_HZ
        stim_e = (start_tr + self.duration_trs) * STIM_RATE_HZ

        video = self._mmap(entry["video_path"])[stim_s:stim_e]
        audio = self._mmap(entry["audio_path"])[stim_s:stim_e]
        if entry.get("text_path"):
            text = self._mmap(entry["text_path"])[stim_s:stim_e]
        else:
            text = np.zeros((stim_e - stim_s, self.text_dim), dtype=np.float32)

        fmri_s = start_tr + self.hrf_offset_trs
        fmri_e = fmri_s + self.duration_trs
        fmri = self._mmap(entry["fmri_path"])[fmri_s:fmri_e]

        # Pad/truncate each stream to EXACTLY its window length so every sample
        # is the same shape — otherwise an edge window can be 1 TR short
        # ([199] vs [200]) and the DataLoader can't stack the batch. .copy()
        # also materializes a WRITABLE array (the source is a read-only mmap)
        # so torch.from_numpy doesn't crash with "resize storage not resizable".
        stim_len = self.duration_trs * STIM_RATE_HZ

        def _fit(a, target):
            n = a.shape[0]
            if n == target:
                return a
            if n > target:
                return a[:target]
            pad = np.zeros((target - n,) + a.shape[1:], dtype=a.dtype)
            return np.concatenate([a, pad], axis=0)

        return {
            "video":   torch.from_numpy(_fit(np.ascontiguousarray(video).copy(), stim_len)).float(),
            "audio":   torch.from_numpy(_fit(np.ascontiguousarray(audio).copy(), stim_len)).float(),
            "text":    torch.from_numpy(_fit(np.ascontiguousarray(text).copy(), stim_len)).float(),
            "fmri":    torch.from_numpy(_fit(np.ascontiguousarray(fmri).copy(), self.duration_trs)).float(),
            "subject": torch.tensor(entry["subject_idx"], dtype=torch.long),
            "mask":    torch.ones(self.duration_trs, dtype=torch.float32),
        }
