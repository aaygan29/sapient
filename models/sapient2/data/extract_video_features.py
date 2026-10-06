"""Extract V-JEPA 2 Gigantic features at 2 Hz, frozen.

Per spec §2.3 and §13 (HF + Modal naming lock).
- Encoder: facebook/vjepa2-vitg-fpc64-256 (MIT, ~1B params, ViT-Gigantic).
- Output: (N_STEPS, 1408) float16 per stimulus, where N_STEPS = duration_s * 2.
  (ViT-Gigantic hidden size is 1408; the original "1280" in the spec was wrong.)
- For each 2-Hz step at time t, we feed V-JEPA the 64 frames uniformly sampled
  from the preceding 4 seconds (the model's native clip size).
- Encoder is frozen — no gradients, no weight updates.

Modal app: sapient-2-features-vjepa2 (A100-80GB per spec §12).

ROOT-CAUSE FIX (2026-06-07): the original run crashed at load time with
`404 ... preprocessor_config.json` because it used `AutoImageProcessor` on
transformers==4.46.0 — a version that predates V-JEPA 2 entirely. V-JEPA 2
ships a `VJEPA2VideoProcessor` (loaded via `AutoVideoProcessor`) and was added
to transformers in 4.52. Fixes: (1) transformers>=4.52 + torch>=2.5, (2) use
`AutoVideoProcessor`, (3) call `model.get_vision_features(...)` instead of
`model(**inputs).last_hidden_state`.

Usage (ds004996 stim layout is per-subject under sourcedata/sub-XX):
  modal run --detach data/extract_video_features.py \
      --stim-root /data/raw/ds004996/sourcedata \
      --out-root /data/features/vjepa2/ds004996
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import modal

APP_NAME = "sapient-2-features-vjepa2"
MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
# V-JEPA 2 ViT-Gigantic hidden size is 1408 (NOT 1280 — that was a spec error;
# get_vision_features returns (1, n_tokens, 1408)). We allocate the feature
# buffer from the model's real hidden size at runtime to stay robust.
HIDDEN_DIM = 1408
CLIP_SECONDS = 4.0
FRAMES_PER_CLIP = 64
TARGET_RATE_HZ = 2.0
# How many 2-Hz clips to push through the encoder per forward pass. 8 clips ×
# 64 frames × 256px fits comfortably in A100-80GB fp16; raise/lower for VRAM.
BATCH_STEPS = 8


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        # V-JEPA 2 needs transformers>=4.52 (AutoVideoProcessor + VJEPA2Model).
        "transformers==4.53.0",
        "decord==0.6.0",      # fast video frame loader
        "numpy>=1.26,<3",
        "huggingface_hub>=0.30,<2",
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
def extract(stim_root: str, out_root: str, subjects: str = "") -> None:
    import os
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from transformers import AutoVideoProcessor, AutoModel

    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN",
                          os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

    stim_dir = Path(stim_root)
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_filter = {s.strip() for s in subjects.split(",") if s.strip()}

    print(f"Loading {MODEL_ID} ...")
    # V-JEPA 2 uses a VIDEO processor, loaded via AutoVideoProcessor.
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = (
        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        .eval()
        .cuda()
    )
    # Derive the true hidden size from the model so the feature buffer never
    # mismatches the encoder output (ViT-g is 1408, not the 1280 in the spec).
    hidden_dim = int(getattr(model.config, "hidden_size", HIDDEN_DIM))
    print(f"  encoder hidden_size = {hidden_dim}")

    video_paths = sorted(
        p for p in stim_dir.rglob("*")
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        and (not sub_filter or any(s in p.parts for s in sub_filter))
    )
    print(f"Found {len(video_paths)} videos under {stim_dir}"
          f"{' (filtered to ' + subjects + ')' if sub_filter else ''}.")

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

        feats = np.empty((n_steps, hidden_dim), dtype=np.float16)
        # Batch BATCH_STEPS clips per GPU forward pass — one ViT-g call per step
        # is dominated by Python/dispatch overhead and leaves the A100
        # under-utilized; batching cuts wall-time and cost ~linearly in batch.
        import time
        t0 = time.time()
        for base in range(0, n_steps, BATCH_STEPS):
            steps = range(base, min(base + BATCH_STEPS, n_steps))
            clips = []
            for step in steps:
                t_end = (step + 1) / TARGET_RATE_HZ        # seconds
                t_start = max(0.0, t_end - CLIP_SECONDS)
                idx = np.linspace(
                    t_start * fps,
                    min(n_frames - 1, t_end * fps),
                    FRAMES_PER_CLIP,
                ).astype(np.int64)
                frames = vr.get_batch(idx).asnumpy()       # (64, H, W, 3) uint8
                # AutoVideoProcessor wants each video as (T, C, H, W).
                clips.append(torch.from_numpy(frames).permute(0, 3, 1, 2))
            # processor accepts a list of videos → batched (B, T, C, H, W).
            inputs = processor(clips, return_tensors="pt").to("cuda")
            with torch.no_grad():
                # (B, n_tokens, hidden_dim); mean-pool tokens → (B, hidden_dim).
                out_h = model.get_vision_features(**inputs)
            pooled = out_h.float().mean(dim=1).to("cpu").numpy()
            feats[base:base + pooled.shape[0]] = pooled.astype(np.float16)
            if base % (BATCH_STEPS * 20) == 0:
                rate = (base + len(steps)) / max(1e-6, time.time() - t0)
                print(f"    {base + len(steps)}/{n_steps} steps "
                      f"({rate:.1f} steps/s)", flush=True)

        np.save(out, feats)
        volume.commit()
    print("Done.")


@app.local_entrypoint()
def main(
    stim_root: str = "/data/raw/ds004996/sourcedata",
    out_root: str = "/data/features/vjepa2/ds004996",
    subjects: str = "sub-01,sub-02,sub-03,sub-04",
) -> None:
    extract.remote(stim_root, out_root, subjects)


if __name__ == "__main__":
    # Local-mode argparse so `python extract_video_features.py --help` works.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--subjects", default="sub-01,sub-02,sub-03,sub-04")
    args = ap.parse_args()
    main(args.stim_root, args.out_root, args.subjects)
