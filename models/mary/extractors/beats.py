"""Live `beats` extractor — BEATs audio-event features at 2 Hz.

Pure refactor of data/extract_beats.py's inner featurize loop. Loads the EXACT
official BEATs_iter3_plus_AS2M checkpoint via the vendored microsoft/unilm source
(the serving image must place that source on sys.path at /opt/beats, same as the
batch image). Model loaded once; `featurize(audio)` → `(T_2Hz, 768)` float32.
"""
from __future__ import annotations

import numpy as np

HIDDEN_DIM = 768
TARGET_RATE_HZ = 2.0
TARGET_SR = 16000
CHUNK_SECONDS = 30.0
MIN_AUDIO_SECONDS = 0.5
MIN_AUDIO_SAMPLES = int(MIN_AUDIO_SECONDS * TARGET_SR)

BEATS_REPO = "Bencr/beats-checkpoints"
BEATS_CKPT = "BEATs_iter3_plus_AS2M.pt"


class BeatsExtractor:
    """Frozen official BEATs, loaded once and reused across requests."""

    def __init__(self, device: str = "cuda", beats_src: str = "/opt/beats"):
        import sys
        import torch
        from huggingface_hub import hf_hub_download

        if beats_src not in sys.path:
            sys.path.insert(0, beats_src)
        from BEATs import BEATs, BEATsConfig  # vendored official source

        self.device = device
        ckpt_path = hf_hub_download(
            repo_id=BEATS_REPO, filename=BEATS_CKPT, repo_type="dataset"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(ckpt["model"])
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _run(self, wav_1d):  # (n,) float tensor @16k → (T_native, 768) np
        import torch

        if wav_1d.shape[0] < MIN_AUDIO_SAMPLES:
            pad_n = MIN_AUDIO_SAMPLES - wav_1d.shape[0]
            wav_1d = torch.cat([wav_1d, torch.zeros(pad_n, dtype=wav_1d.dtype)])
        x = wav_1d.unsqueeze(0).to(self.device)
        pad = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        with torch.no_grad():
            feats, _ = self.model.extract_features(x, padding_mask=pad)
        return feats.squeeze(0).float().cpu().numpy()

    def featurize(self, audio: np.ndarray) -> np.ndarray:
        """audio: 16 kHz mono float32 1-D → (T_2Hz, 768) float32."""
        import torch

        duration_s = len(audio) / TARGET_SR
        n_steps = int(duration_s * TARGET_RATE_HZ)
        if n_steps <= 0 or len(audio) < MIN_AUDIO_SAMPLES:
            raise ValueError(f"audio too short ({duration_s:.3f}s) for beats")

        native_chunks: list[np.ndarray] = []
        chunk_samples = int(CHUNK_SECONDS * TARGET_SR)
        n_padded_samples = 0
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            if len(chunk) < MIN_AUDIO_SAMPLES:
                n_padded_samples += MIN_AUDIO_SAMPLES - len(chunk)
            native_chunks.append(self._run(torch.from_numpy(chunk).float()))
        native = np.concatenate(native_chunks, axis=0)  # (T_native, 768)

        effective_duration_s = duration_s + n_padded_samples / TARGET_SR
        native_rate = native.shape[0] / effective_duration_s
        bin_size = max(1, int(round(native_rate / TARGET_RATE_HZ)))
        feats = np.empty((n_steps, HIDDEN_DIM), dtype=np.float32)
        for i in range(n_steps):
            s = i * bin_size
            e = min(s + bin_size, native.shape[0])
            seg = native[s:e] if e > s else native[-1:]
            feats[i] = seg.mean(axis=0)
        return feats
