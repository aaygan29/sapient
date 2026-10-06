"""Extract Wav2Vec-BERT 2.0 features at 2 Hz, frozen.

Per spec §2.3 and §13. W2V-BERT 2.0 is the PRIMARY audio encoder (decision
#9); Whisper-large-v3 is kept as an ablation only.

- Encoder: facebook/w2v-bert-2.0 (MIT). Native rate ≈ 50 Hz, hidden 1024.
- We downsample to 2 Hz by averaging 50-Hz frames inside each 0.5-second
  window. Output: (N_STEPS, 1024) float16 per stimulus.

Modal app: sapient-2-features-w2vbert (A100-40GB per spec §12).

ROOT-CAUSE FIX (2026-06-07): the original run found all 591 audio sources
(stim-root was correct) but crashed on the FIRST file with
`RuntimeError: expected scalar type Float but found Half`. Cause: the model
was loaded in float16 while the feature extractor feeds float32 input, and
W2V-BERT's conv frontend does not accept the mixed dtype. Fix: load the model
in float32 (it's ~600M params — trivially fits A100-40GB) for numerically
stable, dtype-consistent extraction. Output is still cast to float16 on save.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "sapient-2-features-w2vbert"
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
def extract(stim_root: str, out_root: str, subjects: str = "") -> None:
    import os
    import numpy as np
    import torch
    import librosa
    from transformers import AutoFeatureExtractor, AutoModel

    stim_dir = Path(stim_root)
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Optional CSV subject allowlist (e.g. "sub-01,sub-02,sub-03,sub-04") so we
    # only process the subjects fMRIPrep actually produced BOLD for.
    sub_filter = {s.strip() for s in subjects.split(",") if s.strip()}

    print(f"Loading {MODEL_ID} ...")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    # float32: the FE emits float32 features and W2V-BERT's conv frontend
    # rejects a float16 model fed float32 input ("expected Float but found Half").
    model = (
        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        .eval()
        .cuda()
    )

    audio_paths = sorted(
        p for p in stim_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".mkv"}
        and (not sub_filter or any(s in p.parts for s in sub_filter))
    )
    print(f"Found {len(audio_paths)} audio sources under {stim_dir}"
          f"{' (filtered to ' + subjects + ')' if sub_filter else ''}.")

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
        # The seamless_m4t fbank extractor computes a spectrogram with a 400-
        # sample (25 ms) window; a tail chunk shorter than that yields negative
        # frame counts ("negative dimensions are not allowed"). Pad short tail
        # chunks up to a safe minimum (0.5 s) so the last sliver is still
        # encoded rather than crashing the whole file.
        MIN_SAMPLES = TARGET_SR // 2  # 0.5 s
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            if len(chunk) < MIN_SAMPLES:
                if len(chunk) == 0:
                    continue
                chunk = np.pad(chunk, (0, MIN_SAMPLES - len(chunk)))
            inputs = fe(chunk, sampling_rate=TARGET_SR, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out_h = model(**inputs).last_hidden_state.squeeze(0)  # (T_native, 1024)
            native_features.append(out_h.to("cpu").float().numpy())
        native = np.concatenate(native_features, axis=0).astype(np.float32)

        # Native rate ≈ len(audio)/320 frames per chunk for 16-kHz W2V-BERT.
        native_rate = native.shape[0] / duration_s
        bin_size = max(1, int(round(native_rate / TARGET_RATE_HZ)))
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
    stim_root: str = "/data/raw/ds004996/sourcedata",
    out_root: str = "/data/features/w2vbert/ds004996",
    subjects: str = "sub-01,sub-02,sub-03,sub-04",
) -> None:
    extract.remote(stim_root, out_root, subjects)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--subjects", default="sub-01,sub-02,sub-03,sub-04")
    args = ap.parse_args()
    main(args.stim_root, args.out_root, args.subjects)
