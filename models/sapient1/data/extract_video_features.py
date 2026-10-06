"""Extract V-JEPA 2 Gigantic features at 2 Hz, frozen.

Per spec §2.3 and §13 (HF + Modal naming lock).
- Encoder: facebook/vjepa2-vitg-fpc64-256 (MIT, ~1B params, ViT-Gigantic).
- Output: (N_STEPS, 1280) float16 per stimulus, where N_STEPS = duration_s * 2.
- For each 2-Hz step at time t, we feed V-JEPA the 64 frames uniformly sampled
  from the preceding 4 seconds (the model's native clip size).
- Encoder is frozen — no gradients, no weight updates.

Modal app: sapient-1-features-vjepa2 (A100-80GB per spec §12).

Usage:
  modal run data/extract_video_features.py \
      --stim-root /data/raw/cneuromod/.../stimuli \
      --out-root /data/features/vjepa2/cneuromod
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import modal

APP_NAME = "sapient-1-features-vjepa2"
MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
HIDDEN_DIM = 1280
CLIP_SECONDS = 4.0
FRAMES_PER_CLIP = 64
TARGET_RATE_HZ = 2.0


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.46.0",
        "decord==0.6.0",      # fast video frame loader
        "numpy>=1.26,<3",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
    )
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def extract(stim_root: str, out_root: str) -> None:
    import os
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from transformers import AutoImageProcessor, AutoModel

    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN",
                          os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

    stim_dir = Path(stim_root)
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = (
        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        .eval()
        .cuda()
    )

    video_paths = sorted(
        p for p in stim_dir.rglob("*")
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    )
    print(f"Found {len(video_paths)} videos under {stim_dir}.")

    for vid in video_paths:
        rel = vid.relative_to(stim_dir).with_suffix(".npy")
        out = out_dir / rel
        if out.exists():
            print(f"  skip {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        print(f"  extract {vid} → {out}")
        vr = VideoReader(str(vid), ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        n_frames = len(vr)
        duration_s = n_frames / fps
        n_steps = int(math.floor(duration_s * TARGET_RATE_HZ))
        if n_steps <= 0:
            print(f"    too short ({duration_s:.2f}s), skipping")
            continue

        feats = np.empty((n_steps, HIDDEN_DIM), dtype=np.float16)
        for step in range(n_steps):
            t_end = (step + 1) / TARGET_RATE_HZ        # seconds
            t_start = max(0.0, t_end - CLIP_SECONDS)
            # Uniformly sample FRAMES_PER_CLIP frames over [t_start, t_end].
            idx = np.linspace(
                t_start * fps,
                min(n_frames - 1, t_end * fps),
                FRAMES_PER_CLIP,
            ).astype(np.int64)
            frames = vr.get_batch(idx).asnumpy()       # (64, H, W, 3) uint8
            inputs = processor(list(frames), return_tensors="pt").to("cuda")
            with torch.no_grad():
                out_h = model(**inputs).last_hidden_state   # (1, n_tokens, 1280)
            feats[step] = out_h.mean(dim=1).squeeze(0).to("cpu").numpy()

        np.save(out, feats)
        volume.commit()
    print("Done.")


@app.local_entrypoint()
def main(
    stim_root: str = "/data/raw/cneuromod/fmriprep/friends/sourcedata/friends/stimuli",
    out_root: str = "/data/features/vjepa2/cneuromod",
) -> None:
    extract.remote(stim_root, out_root)


if __name__ == "__main__":
    # Local-mode argparse so `python extract_video_features.py --help` works.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    main(args.stim_root, args.out_root)
