"""Mary stream `qwen_vl` — Qwen3-VL-8B-Instruct visual features at 2 Hz, frozen. [INACTIVE]

CONTRACTS §2: D_m = 3584, grid = 2 Hz, output `(T_2Hz, 3584)` float16.
Purpose (01-§2): scene semantics, visual reasoning.

STATUS THIS SPRINT: INACTIVE. ds002345 is audio-only — NO open video media.
The `local_entrypoint` detects the absence of video stimuli and exits cleanly.
The full extraction logic is authored and drop-in for when video appears.

Backbone: `Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0). We use the vision tower +
language model's hidden states: for each 2 Hz step we sample the frame at that
timestamp, run the VL model on a fixed visual-description prompt, and mean-pool
the last-layer hidden states (hidden_size=3584) over the visual+text tokens.

Modal app: mary-features-qwen_vl (GPU A100-80GB). Frozen, eval mode, no grad.

Usage:
  modal run data/extract_qwen_vl.py --dataset cneuromod --only movie10_bourne05  # one clip
  modal run --detach data/extract_qwen_vl.py --dataset cneuromod --only movie10   # all movie10
  modal run data/extract_qwen_vl.py --stim-root /data/raw/<ds>/stimuli --out-root ...  # generic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "mary-features-qwen_vl"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
STREAM = "qwen_vl"
HIDDEN_DIM = 4096   # Qwen3-VL-8B real hidden size (paper's 3584 = Qwen2-VL-7B). Matches STREAM_DIMS.
TARGET_RATE_HZ = 2.0
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
PROMPT = "Describe the scene."

# Named video datasets (CONTRACTS §3). `--dataset cneuromod` selects the
# Algonauts-2025 movie tree + canonical story keys aligned to fMRI `task-` labels.
DATASETS = {
    "cneuromod": {
        "stim_root": "/data/raw/algonauts2025/stimuli/movies",
        "out_root": "/data/features/mary/cneuromod/_stories",
        "canonical_keys": True,
    },
    # HAD (ds004488): per-clip features keyed `{Category}__{clip}`, restricted to
    # clips shown to sub-01/sub-02 (their events.tsv).
    "had": {
        "stim_root": "/data/raw/ds004488/stimuli",
        "out_root": "/data/features/mary/had/_stories",
        "had_subjects": "sub-01,sub-02",
    },
}
_CN_PRUNE = {".git", ".datalad", ".github", "__pycache__"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        # Qwen3-VL (model_type `qwen3_vl`) needs transformers>=4.57; the older
        # 4.51.3 pin raised "model type qwen3_vl ... not recognized". Bumped here
        # with a matching torch 2.6.0 so the HAD video stream can run.
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers>=4.57,<5",
        "accelerate>=0.34",
        "qwen-vl-utils",
        "decord==0.6.0",
        "pillow>=10",
        "numpy>=1.26,<3",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"})
)

data_volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("mary-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App(APP_NAME)


def _find_videos(stim_root: str) -> list[Path]:
    root = Path(stim_root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in VIDEO_EXTS
                  and not any(part in _CN_PRUNE for part in p.parts))


def _cn_norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _cn_story_key(vid: Path, root: Path) -> str:
    """Canonical CNeuroMod story key aligned to fMRI `task-` labels."""
    import re
    rel = vid.relative_to(root)
    group = rel.parts[0] if rel.parts else ""
    stem = vid.stem
    if group == "friends":
        m = re.search(r"(s\d{2}e\d{2}[a-d]?)", stem.lower())
        return f"friends_{m.group(1) if m else _cn_norm(stem)}"
    return f"{group}_{_cn_norm(stem)}"


def _rel_story(vid: Path, root: Path) -> str:
    """Per-clip key `{Category}__{clip}` (matches slowfast/beats/whisper for HAD)."""
    return str(vid.relative_to(root).with_suffix("")).replace("/", "__")


# HAD (ds004488): restrict to clips sub-01/sub-02 actually saw (events.tsv).
_HAD_SES = "ses-action01"


def _had_referenced_videos(stim_root: str, subjects: list[str]) -> list[Path]:
    import csv
    root = Path(stim_root)
    raw = root.parent
    rels: set[str] = set()
    for sub in subjects:
        func = raw / sub / _HAD_SES / "func"
        if not func.exists():
            continue
        for ev in sorted(func.glob(f"{sub}_{_HAD_SES}_task-action_run-*_events.tsv")):
            with ev.open() as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    stim = (r.get("stim_file") or "").strip()
                    if stim and stim not in ("n/a", "nan"):
                        rels.add(stim)
    return sorted(root / rel for rel in rels if (root / rel).exists())


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/cache": hf_cache_volume},
    secrets=[hf_secret],
)
def extract(stim_root: str, out_root: str, overwrite: bool = False,
            canonical: bool = False, only: str | None = None,
            limit: int = 0, had_subjects: str | None = None) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from decord import VideoReader, cpu
    from transformers import AutoModelForImageTextToText, AutoProcessor

    root = Path(stim_root)
    is_had = bool(had_subjects)
    if is_had:
        subs = [s.strip() for s in had_subjects.split(",") if s.strip()]
        videos = _had_referenced_videos(stim_root, subs)
        print(f"[{STREAM}] HAD: {len(videos)} clips referenced by {subs}")
    else:
        videos = _find_videos(stim_root)
    if canonical and only:
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        videos = [v for v in videos
                  if (_cn_story_key(v, root) in wanted
                      or _cn_story_key(v, root).split("_", 1)[0] in wanted
                      or any(w in _cn_story_key(v, root) for w in wanted))]
    if limit > 0:
        videos = videos[:limit]
    if not videos:
        print(f"[{STREAM}] INACTIVE: no video stimuli under {stim_root}; nothing to do.")
        return {"stream": STREAM, "active": False, "n": 0}

    print(f"[{STREAM}] loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, output_hidden_states=True
        )
        .eval()
        .cuda()
    )
    for p in model.parameters():
        p.requires_grad_(False)

    out_dir = Path(out_root)
    n_done = 0
    for vid in videos:
        story = (_rel_story(vid, root) if is_had
                 else _cn_story_key(vid, root) if canonical else vid.stem)
        out = out_dir / story / f"{STREAM}.npy"
        if out.exists() and not overwrite:
            print(f"  skip {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        vr = VideoReader(str(vid), ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        n_frames = len(vr)
        duration_s = n_frames / fps
        n_steps = int(duration_s * TARGET_RATE_HZ)
        if n_steps <= 0:
            continue

        feats = np.empty((n_steps, HIDDEN_DIM), dtype=np.float16)
        for step in range(n_steps):
            t = (step + 0.5) / TARGET_RATE_HZ
            fidx = min(n_frames - 1, int(round(t * fps)))
            frame = Image.fromarray(vr[fidx].asnumpy())
            messages = [{"role": "user", "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": PROMPT},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[frame], return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                out_h = model(**inputs)
            last = out_h.hidden_states[-1].squeeze(0).float()   # (T_tok, 4096)
            feats[step] = last.mean(dim=0).cpu().numpy()

        np.save(out, feats)
        data_volume.commit()
        n_done += 1
        print(f"  ✓ {story}: {feats.shape} → {out}")

    print(f"[{STREAM}] done. {n_done} videos.")
    return {"stream": STREAM, "active": True, "n": n_done}


@app.local_entrypoint()
def main(
    stim_root: str = "/data/raw/ds002345/stimuli",
    out_root: str = "/data/features/mary/huth/_stories",
    overwrite: bool = False,
    dataset: str = "",
    only: str = "",
    limit: int = 0,
    had_subjects: str = "",
) -> None:
    canonical = False
    if dataset:
        cfg = DATASETS[dataset]
        stim_root = cfg["stim_root"]
        out_root = cfg["out_root"]
        canonical = cfg.get("canonical_keys", False)
        had_subjects = had_subjects or cfg.get("had_subjects", "")
    res = extract.remote(stim_root, out_root, overwrite=overwrite,
                         canonical=canonical, only=only or None, limit=limit,
                         had_subjects=had_subjects or None)
    if not res.get("active"):
        print(f"[{STREAM}] INACTIVE — no video stimuli under {stim_root}. "
              f"Re-run with --dataset cneuromod (or --stim-root) once stimuli land.")
    print(f"\nResult: {res}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim-root", default="/data/raw/ds002345/stimuli")
    ap.add_argument("--out-root", default="/data/features/mary/huth/_stories")
    ap.add_argument("--dataset", default="", choices=[""] + list(DATASETS))
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--had-subjects", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args.stim_root, args.out_root, args.overwrite,
         args.dataset, args.only, args.limit, args.had_subjects)
