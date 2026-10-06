"""Shared Modal definitions for Mary feature extractors (WS-F).

Every `data/extract_*.py` app mounts the `sapient-data` volume at `/data`,
authenticates HF downloads with the `hf-token` secret, and caches HF model
downloads on a dedicated `mary-hf-cache` volume mounted at `/cache` (with
`HF_HOME=/cache/huggingface`, `TORCH_HOME=/cache/torch`) so the (large) Qwen /
Whisper / BEATs checkpoints are downloaded ONCE and reused across the 3 active
apps + re-runs. NOTE: the Modal base image ships a non-empty `/root/.cache`, so
we deliberately mount at `/cache` rather than `/root/.cache`.

Conventions locked by CONTRACTS.md:
- Modal profile `robert-16572`, volume `sapient-data` at `/data`, secret `hf-token`.
- Stream features cached at 2 Hz as `(T_2Hz, D_m)` float16.
- Audio/text features are IDENTICAL across subjects for a given story, so we
  write ONCE per story to the shared path (see `shared_feature_path`).
"""

from __future__ import annotations

from pathlib import Path

import modal

# --- Volumes & secrets (shared by all extractors) ---------------------------
DATA_VOLUME = modal.Volume.from_name("sapient-data", create_if_missing=True)
HF_CACHE_VOLUME = modal.Volume.from_name("mary-hf-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("hf-token")

# Mount points inside the container.
DATA_MOUNT = "/data"
HF_CACHE_MOUNT = "/cache"   # NOT /root/.cache (base image already populates it)
HF_CACHE_ENV = {"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"}

VOLUMES = {DATA_MOUNT: DATA_VOLUME, HF_CACHE_MOUNT: HF_CACHE_VOLUME}

# --- Locked contract constants ----------------------------------------------
TARGET_RATE_HZ = 2.0   # canonical 2 Hz feature grid (CONTRACTS §2)
TARGET_SR = 16000      # 16 kHz mono PCM for all audio backbones

# ds002345 (Huth Narratives) — the sprint dataset (CONTRACTS §0).
DS002345_STIM_ROOT = "/data/raw/ds002345/stimuli"

# Story-level shared feature destination (CONTRACTS §3, audio/text streams are
# identical across subjects → written once per story).
SHARED_STORIES_ROOT = "/data/features/mary/huth/_stories"

# Stream dims (CONTRACTS §2, LOCKED).
STREAM_DIMS = {
    "slowfast": 2304,
    "qwen_vl": 3584,
    "beats": 768,
    "whisper": 1280,
    "qwen_ctx": 4096,
    "got_ocr": 768,
}


def story_name_from_wav(wav_path: str | Path) -> str:
    """`/data/raw/ds002345/stimuli/pieman_audio.wav` -> `pieman`."""
    stem = Path(wav_path).stem            # pieman_audio
    return stem[:-6] if stem.endswith("_audio") else stem


def shared_feature_path(story: str, stream: str) -> str:
    """Shared story-level cache path: .../_stories/{story}/{stream}.npy."""
    return f"{SHARED_STORIES_ROOT}/{story}/{stream}.npy"


def list_story_wavs(stim_root: str) -> list[Path]:
    """All `*_audio.wav` story stimuli under stim_root, deterministic order."""
    root = Path(stim_root)
    return sorted(root.glob("*_audio.wav"))


# --- Reusable image builders -------------------------------------------------
def audio_image(extra_pip: list[str] | None = None) -> modal.Image:
    """Base image for audio backbones (ffmpeg + torch + transformers)."""
    pkgs = [
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "transformers==4.46.0",
        "soundfile==0.12.1",
        "librosa==0.10.2",
        "numpy>=1.26,<3",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
    ]
    if extra_pip:
        pkgs.extend(extra_pip)
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg")
        .pip_install(*pkgs)
    )


def video_image(extra_pip: list[str] | None = None) -> modal.Image:
    """Base image for video backbones (ffmpeg + decord + torch + transformers).

    Authored for the INACTIVE video streams; ready to use if open video
    stimuli appear.
    """
    pkgs = [
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.46.0",
        "decord==0.6.0",
        "numpy>=1.26,<3",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
    ]
    if extra_pip:
        pkgs.extend(extra_pip)
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg")
        .pip_install(*pkgs)
    )
