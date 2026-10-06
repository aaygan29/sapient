"""Live `whisper` extractor — Whisper-large-v3-turbo ENCODER features at 2 Hz.

Pure refactor of data/extract_whisper.py's inner featurize loop. Model loaded
once; `featurize(audio)` returns `(T_2Hz, 1280)` float32. Contract: D_m = 1280.
"""
from __future__ import annotations

import numpy as np

MODEL_ID = "openai/whisper-large-v3-turbo"
HIDDEN_DIM = 1280
TARGET_RATE_HZ = 2.0
TARGET_SR = 16000
WINDOW_SECONDS = 30.0  # Whisper's fixed mel context window


class WhisperExtractor:
    """Frozen Whisper encoder, loaded once and reused across requests."""

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoFeatureExtractor, WhisperForConditionalGeneration

        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        full = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        )
        self.encoder = full.get_encoder().eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def featurize(self, audio: np.ndarray) -> np.ndarray:
        """audio: 16 kHz mono float32 1-D → (T_2Hz, 1280) float32."""
        import torch

        duration_s = len(audio) / TARGET_SR
        n_steps = int(duration_s * TARGET_RATE_HZ)
        if n_steps <= 0:
            raise ValueError(f"audio too short ({duration_s:.3f}s) for whisper")

        native_chunks: list[np.ndarray] = []
        win = int(WINDOW_SECONDS * TARGET_SR)
        for start in range(0, len(audio), win):
            chunk = audio[start:start + win]
            chunk_secs = len(chunk) / TARGET_SR
            inputs = self.fe(
                chunk, sampling_rate=TARGET_SR, return_tensors="pt"
            ).to(self.device, torch.float16)
            with torch.no_grad():
                hs = self.encoder(inputs.input_features).last_hidden_state.squeeze(0)
            n_native = hs.shape[0]
            n_keep = max(1, int(round(n_native * (chunk_secs / WINDOW_SECONDS))))
            native_chunks.append(hs[:n_keep].float().cpu().numpy())

        native = np.concatenate(native_chunks, axis=0)  # (T_native, 1280)
        native_rate = native.shape[0] / duration_s
        bin_size = max(1, int(round(native_rate / TARGET_RATE_HZ)))

        feats = np.empty((n_steps, HIDDEN_DIM), dtype=np.float32)
        for i in range(n_steps):
            s = i * bin_size
            e = min(s + bin_size, native.shape[0])
            seg = native[s:e] if e > s else native[-1:]
            feats[i] = seg.mean(axis=0)
        return feats
