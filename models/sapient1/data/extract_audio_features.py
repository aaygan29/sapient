"""Extract Wav2Vec-BERT 2.0 features at 2 Hz, frozen.

Per spec §2.3 and §13. W2V-BERT 2.0 is the PRIMARY audio encoder (decision
#9); Whisper-large-v3 is kept as an ablation only.

- Encoder: facebook/w2v-bert-2.0 (MIT). Native rate ≈ 50 Hz, hidden 1024.
- We downsample to 2 Hz by averaging 50-Hz frames inside each 0.5-second
  window. Output: (N_STEPS, 1024) float16 per stimulus.

Modal app: sapient-1-features-w2vbert (A100-40GB per spec §12).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "sapient-1-features-w2vbert"
MODEL_ID = "facebook/w2v-bert-2.0"
HIDDEN_DIM = 1024
TARGET_RATE_HZ = 2.0
TARGET_SR = 16000     # W2V-BERT accepts 16-kHz mono PCM
CHUNK_SECONDS = 30.0  # process audio in 30-s chunks to stay in memory


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "transformers==4.46.0",
        "soundfile==0.12.1",
        "librosa==0.10.2",
        "numpy>=1.26,<3",
        "huggingface_hub>=0.25,<2",
    )
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def extract(stim_root: str, out_root: str) -> None:
    import os
    import numpy as np
    import torch
    import librosa
    from transformers import AutoFeatureExtractor, AutoModel

    stim_dir = Path(stim_root)
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = (
        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        .eval()
        .cuda()
    )

    audio_paths = sorted(
        p for p in stim_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".mkv"}
    )
    print(f"Found {len(audio_paths)} audio sources under {stim_dir}.")

    for src in audio_paths:
        rel = src.relative_to(stim_dir).with_suffix(".npy")
        out = out_dir / rel
        if out.exists():
            print(f"  skip {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        print(f"  extract {src} → {out}")
        try:
            audio, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"    librosa failed: {e}; skipping")
            continue
        duration_s = len(audio) / TARGET_SR
        n_steps = int(duration_s * TARGET_RATE_HZ)
        if n_steps <= 0:
            continue

        # Native W2V-BERT output ≈ 50 Hz; we'll average to 2 Hz at the end.
        native_features: list[np.ndarray] = []
        chunk_samples = int(CHUNK_SECONDS * TARGET_SR)
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            inputs = fe(chunk, sampling_rate=TARGET_SR, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out_h = model(**inputs).last_hidden_state.squeeze(0)  # (T_native, 1024)
            native_features.append(out_h.to("cpu").float().numpy())
        native = np.concatenate(native_features, axis=0).astype(np.float32)

        # Native rate ≈ len(audio)/320 frames per chunk for 16-kHz W2V-BERT.
        native_rate = native.shape[0] / duration_s
        bin_size = int(round(native_rate / TARGET_RATE_HZ))
        # Average non-overlapping bins of size bin_size to land at 2 Hz.
        feats = np.empty((n_steps, HIDDEN_DIM), dtype=np.float16)
        for i in range(n_steps):
            s = i * bin_size
            e = s + bin_size
            feats[i] = native[s:e].mean(axis=0).astype(np.float16)

        np.save(out, feats)
        volume.commit()
    print("Done.")


@app.local_entrypoint()
def main(
    stim_root: str = "/data/raw/cneuromod/fmriprep/friends/sourcedata/friends/stimuli",
    out_root: str = "/data/features/w2vbert/cneuromod",
) -> None:
    extract.remote(stim_root, out_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    main(args.stim_root, args.out_root)
