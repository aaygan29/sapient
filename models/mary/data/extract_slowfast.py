"""Mary stream `slowfast` — SlowFast R101 video features at 2 Hz, frozen. [ACTIVE]

CONTRACTS §2: D_m = 2304, grid = 2 Hz, output `(T_2Hz, 2304)` float16.
Purpose (01-§2): motion & spatiotemporal patterns (drives visual cortex).

BACKBONE / SUBSTITUTION NOTE
----------------------------
ORCLE spec: "SlowFast R101". There is no HF-hosted `transformers` SlowFast, so we
load the EXACT official backbone via torchhub: `facebookresearch/pytorchvideo ::
slowfast_r101` (Kinetics-400, ~60M params, the canonical PySlowFast R101 weights).
This is the real backbone, not a re-implementation. The two pathways concatenate
to slow 2048 + fast 256 = **2304-d** at the head's global-pooled features — we tap
those pre-classifier pooled features with a forward hook (no projection needed,
the native dim already equals the contract dim).

Decode: ffmpeg/decord. For each 2 Hz step we feed the SlowFast clip ending at that
timestamp (32-frame fast pathway over `CLIP_SECONDS`, 8-frame slow subsample), so
output rate is exactly 2 Hz → `(T_2Hz, 2304)`.

Generalized for multi-dataset: pass `--stim-root` (a dir tree of video files; any
of .mp4/.mkv/.webm/.mov/.avi) and `--out-root`. Writes one `.npy` per video to
`{out_root}/{relpath_no_ext}/slowfast.npy` (CONTRACTS §3 `_stories/{clip}` layout).

Modal app: mary-features-slowfast (GPU A10G/A100). Frozen, eval mode, no grad.

Usage:
  # validate one HAD clip
  modal run data/extract_slowfast.py --stim-root /data/raw/ds004488/stimuli \
      --out-root /data/features/mary/had/_stories --limit 1
  # full HAD action clips
  modal run --detach data/extract_slowfast.py \
      --stim-root /data/raw/ds004488/stimuli \
      --out-root /data/features/mary/had/_stories
  # CNeuroMod movies (canonical story keys aligned to fMRI task labels)
  modal run data/extract_slowfast.py --dataset cneuromod --only movie10_bourne05  # one clip
  modal run --detach data/extract_slowfast.py --dataset cneuromod --only movie10  # all movie10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "mary-features-slowfast"
STREAM = "slowfast"
HIDDEN_DIM = 2304            # slow 2048 + fast 256
TARGET_RATE_HZ = 2.0
CLIP_SECONDS = 2.0           # temporal receptive field per 2 Hz step
SLOW_FRAMES = 8
FAST_FRAMES = 32
SLOWFAST_ALPHA = 4           # fast:slow temporal sampling ratio
SIDE = 256                   # spatial size fed to the network
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

# Named video datasets (CONTRACTS §3). `--dataset cneuromod` selects the
# Algonauts-2025 movie tree and uses canonical story keys aligned to the fMRI
# `task-` labels (so a later manifest pairs feature↔fMRI). Other video datasets
# (e.g. HAD) keep using bare --stim-root/--out-root with stem-based keys.
DATASETS = {
    "cneuromod": {
        "stim_root": "/data/raw/algonauts2025/stimuli/movies",
        "out_root": "/data/features/mary/cneuromod/_stories",
        "canonical_keys": True,
    },
    # HAD (ds004488): per-clip features keyed `{Category}__{clip}` (== _rel_story),
    # restricted to clips shown to sub-01/sub-02 (their events.tsv).
    "had": {
        "stim_root": "/data/raw/ds004488/stimuli",
        "out_root": "/data/features/mary/had/_stories",
        "had_subjects": "sub-01,sub-02",
    },
}
_CN_PRUNE = {".git", ".datalad", ".github", "__pycache__"}

# pytorchvideo 0.1.5 has import breakage on torch>=2.1 (functorch move); torch
# 2.0.1 + torchvision 0.15.2 + pytorchvideo 0.1.5 is the known-good triple.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        "pytorchvideo==0.1.5",
        "fvcore",
        "decord==0.6.0",
        "numpy>=1.23,<2",
        "huggingface_hub>=0.25,<1",
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


def _rel_story(vid: Path, root: Path) -> str:
    """Stable per-clip key from path relative to stim root (no extension),
    slashes → '__' so it is a single dir level under out_root."""
    rel = vid.relative_to(root).with_suffix("")
    return str(rel).replace("/", "__")


# HAD (ds004488): 21,600 clips on disk, but sub-01/sub-02 see only ~720 each.
# When extracting HAD, pass --had-subjects sub-01,sub-02 to restrict to the
# clips actually shown (their events.tsv) instead of all 21,600.
_HAD_SES = "ses-action01"


def _had_referenced_videos(stim_root: str, subjects: list[str]) -> list[Path]:
    import csv
    root = Path(stim_root)
    raw = root.parent  # /data/raw/ds004488
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


def _cn_norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _cn_story_key(vid: Path, root: Path) -> str:
    """Canonical CNeuroMod story key aligned to fMRI `task-` labels:
    movie10/bourne/bourne05.mkv -> 'movie10_bourne05';
    friends/s1/friends_s01e24a.mkv -> 'friends_s01e24a'."""
    import re
    rel = vid.relative_to(root)
    group = rel.parts[0] if rel.parts else ""
    stem = vid.stem
    if group == "friends":
        m = re.search(r"(s\d{2}e\d{2}[a-d]?)", stem.lower())
        return f"friends_{m.group(1) if m else _cn_norm(stem)}"
    return f"{group}_{_cn_norm(stem)}"


@app.function(
    image=image,
    gpu="A10G",
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/cache": hf_cache_volume},
    secrets=[hf_secret],
)
def extract(stim_root: str, out_root: str, overwrite: bool = False,
            limit: int = 0, canonical: bool = False,
            only: str | None = None, had_subjects: str | None = None) -> dict:
    import time
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    root = Path(stim_root)
    if had_subjects:
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
        print(f"[{STREAM}] no video stimuli under {stim_root}; nothing to do.")
        return {"stream": STREAM, "active": False, "n": 0}
    print(f"[{STREAM}] {len(videos)} video(s) under {stim_root}")

    print(f"[{STREAM}] loading SlowFast R101 (Kinetics-400) via torchhub ...")
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", "slowfast_r101", pretrained=True
    )
    model = model.eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)

    # Tap the head's pooled features (pre-classifier). The SlowFast head is the
    # last block; its `pool` produces (B, 2304, 1,1,1). We hook its dropout (or
    # pool) output rather than the projected logits.
    captured = {}
    head = model.blocks[-1]

    def hook(_m, _inp, out):
        captured["feat"] = out

    # Prefer hooking the dropout (post-pool, pre-proj). Fall back to pool.
    target = getattr(head, "dropout", None) or getattr(head, "pool", None)
    assert target is not None, "could not find SlowFast head pool/dropout"
    h = target.register_forward_hook(hook)

    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1).cuda()
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1).cuda()

    def pack_pathways(frames: torch.Tensor) -> list[torch.Tensor]:
        # frames: (C, T_fast, H, W) on cuda. Fast = all; Slow = every alpha-th.
        fast = frames
        idx = torch.linspace(
            0, frames.shape[1] - 1, SLOW_FRAMES, device=frames.device
        ).long()
        slow = frames.index_select(1, idx)
        return [slow.unsqueeze(0), fast.unsqueeze(0)]

    def featurize(clip_thwc: np.ndarray) -> np.ndarray:
        # clip_thwc: (T_fast, H, W, 3) uint8
        clip = torch.from_numpy(clip_thwc).to("cuda")
        clip = clip.permute(3, 0, 1, 2).float() / 255.0          # (3,T,H,W)
        # center-crop-ish resize to SIDE x SIDE
        clip = torch.nn.functional.interpolate(
            clip, size=(SIDE, SIDE), mode="bilinear", align_corners=False
        )
        clip = (clip - mean) / std
        captured.clear()
        with torch.no_grad():
            _ = model(pack_pathways(clip))
        t = captured["feat"].float()                 # e.g. (1, 2304, 1, h, w)
        # Find the channel axis equal to HIDDEN_DIM and global-avg-pool the rest.
        ch_axis = next(
            (i for i, s in enumerate(t.shape) if s == HIDDEN_DIM), None)
        assert ch_axis is not None, (
            f"no axis == {HIDDEN_DIM} in slowfast head feat shape {tuple(t.shape)}")
        reduce_axes = [i for i in range(t.dim()) if i != ch_axis]
        if reduce_axes:
            t = t.mean(dim=reduce_axes)
        feat = t.reshape(-1).cpu().numpy()
        assert feat.shape[0] == HIDDEN_DIM, (
            f"slowfast pooled dim {feat.shape[0]} != {HIDDEN_DIM} "
            f"(raw {tuple(captured['feat'].shape)})")
        return feat

    out_dir = Path(out_root)
    timings: dict[str, float] = {}
    n_done = 0
    for vid in videos:
        story = _cn_story_key(vid, root) if canonical else _rel_story(vid, root)
        out = out_dir / story / f"{STREAM}.npy"
        if out.exists() and not overwrite:
            print(f"  skip {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            vr = VideoReader(str(vid), ctx=cpu(0))
        except Exception as e:
            print(f"  !! decode fail {vid}: {e}")
            continue
        fps = float(vr.get_avg_fps()) or 30.0
        n_frames = len(vr)
        duration_s = n_frames / fps
        n_steps = int(duration_s * TARGET_RATE_HZ)
        if n_steps <= 0:
            print(f"  {story}: too short ({duration_s:.2f}s), skipping")
            continue

        feats = np.empty((n_steps, HIDDEN_DIM), dtype=np.float16)
        for step in range(n_steps):
            t_end = (step + 1) / TARGET_RATE_HZ
            t_start = max(0.0, t_end - CLIP_SECONDS)
            fidx = np.linspace(
                t_start * fps, min(n_frames - 1, t_end * fps - 1e-3), FAST_FRAMES
            )
            fidx = np.clip(np.round(fidx), 0, n_frames - 1).astype(np.int64)
            clip = vr.get_batch(list(fidx)).asnumpy()        # (T,H,W,3) uint8
            feats[step] = featurize(clip).astype(np.float16)

        np.save(out, feats)
        data_volume.commit()
        dt = time.time() - t0
        timings[story] = dt
        n_done += 1
        print(f"  ✓ {story}: {feats.shape} ({duration_s:.1f}s @ {fps:.1f}fps) "
              f"in {dt:.1f}s → {out}", flush=True)

    h.remove()
    print(f"[{STREAM}] done. {n_done} videos.")
    return {"stream": STREAM, "active": True, "n": n_done, "timings": timings}


@app.local_entrypoint()
def main(
    stim_root: str = "/data/raw/ds004488/stimuli",
    out_root: str = "/data/features/mary/had/_stories",
    overwrite: bool = False,
    limit: int = 0,
    dataset: str = "",
    only: str = "",
    had_subjects: str = "",
) -> None:
    # `--dataset cneuromod` overrides stim/out roots + enables canonical keys.
    # `--dataset had` restricts to the clips sub-01/sub-02 actually saw.
    canonical = False
    if dataset:
        cfg = DATASETS[dataset]
        stim_root = cfg["stim_root"]
        out_root = cfg["out_root"]
        canonical = cfg.get("canonical_keys", False)
        had_subjects = had_subjects or cfg.get("had_subjects", "")
    res = extract.remote(stim_root, out_root, overwrite=overwrite, limit=limit,
                         canonical=canonical, only=only or None,
                         had_subjects=had_subjects or None)
    print(f"\nResult: {res}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim-root", default="/data/raw/ds004488/stimuli")
    ap.add_argument("--out-root", default="/data/features/mary/had/_stories")
    ap.add_argument("--dataset", default="", choices=[""] + list(DATASETS))
    ap.add_argument("--only", default="")
    ap.add_argument("--had-subjects", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    main(args.stim_root, args.out_root, args.overwrite, args.limit,
         args.dataset, args.only, args.had_subjects)
