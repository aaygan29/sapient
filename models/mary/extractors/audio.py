"""Shared audio loading for the live extractors — 16 kHz mono float32.

Lifted verbatim from data/extract_whisper.py::_load_audio_16k_mono so the live
path decodes audio identically to training-time feature extraction.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

TARGET_SR = 16000


def load_audio_16k_mono(path: str) -> np.ndarray:
    """Load audio at 16 kHz mono as float32. `.wav/.flac/.ogg` via librosa;
    video containers (`.mp4/.mkv/...`) decode the audio track via ffmpeg.
    Returns a possibly-empty 1-D array (clips with no audio track → empty)."""
    ext = Path(path).suffix.lower()
    if ext in (".wav", ".flac", ".ogg"):
        import librosa

        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        return audio.astype(np.float32)
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
