"""Mary stream `beats` — BEATs audio-event features at 2 Hz, frozen.

CONTRACTS §2: D_m = 768, grid = 2 Hz, output `(T_2Hz, 768)` float16.
Purpose (01-§2): environmental audio, music, sound events.

BACKBONE / SUBSTITUTION NOTE
----------------------------
The ORCLE spec names "BEATs" (microsoft/unilm BEATs). Microsoft does NOT publish
BEATs as a `transformers` AutoModel, and there is no `beats-trainer` PyPI package
(checked: "No matching distribution found"). The canonical artifact is the
checkpoint `BEATs_iter3_plus_AS2M.pt` (MIT, 768-d, 90M params, AudioSet-2M iter3+)
plus the official `BEATs.py` model code.

We therefore load the EXACT official backbone: the image vendors the official
microsoft/unilm BEATs source (`BEATs.py`, `backbone.py`, `modules.py`,
downloaded at build time into `/opt/beats`) and the EXACT official checkpoint
`BEATs_iter3_plus_AS2M.pt`, pulled from the HF dataset `Bencr/beats-checkpoints`.
This is the *real* BEATs backbone (not a re-implementation); only the packaging
differs from a bare AutoModel call.
HF id used for weights: `Bencr/beats-checkpoints :: BEATs_iter3_plus_AS2M.pt`.

BEATs emits patch-level hidden states at ~49.96 Hz (16 kHz audio, 320-sample
stride) of dim 768. We average non-overlapping bins down to 2 Hz → `(T_2Hz, 768)`.

Audio/text features are identical across subjects for a story → written ONCE
per story to `/data/features/mary/huth/_stories/{story}/beats.npy` (CONTRACTS §3).

Modal app: mary-features-beats (GPU A100-40GB). Frozen, eval mode, no grad.

Usage:
  modal run data/extract_beats.py                                  # all ds002345 (huth) stories
  modal run data/extract_beats.py --only pieman                    # validate one huth story
  modal run data/extract_beats.py --dataset lebel2023              # all ds003020 stories
  modal run data/extract_beats.py --dataset lebel2023 --only buck  # validate one lebel story
  modal run data/extract_beats.py --dataset cneuromod --only movie10_bourne05   # one Bourne clip
  modal run --detach data/extract_beats.py --dataset cneuromod --only movie10   # all movie10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "mary-features-beats"
STREAM = "beats"
HIDDEN_DIM = 768
TARGET_RATE_HZ = 2.0
TARGET_SR = 16000
CHUNK_SECONDS = 30.0   # process in 30-s chunks to bound memory
# BEATs builds a log-mel fbank (128 mel bins, 10 ms hop) then a 16x16 patch
# embed. A clip shorter than ~16 fbank frames yields a spectrogram smaller than
# the 16x16 kernel and crashes ("Calculated padded input size ... Kernel size:
# (16 x 16)"). 16 frames @ 10 ms = 160 ms; we pad to a safe 0.5 s minimum so the
# patch embed always has a valid input.
MIN_AUDIO_SECONDS = 0.5
MIN_AUDIO_SAMPLES = int(MIN_AUDIO_SECONDS * TARGET_SR)

BEATS_REPO = "Bencr/beats-checkpoints"
BEATS_CKPT = "BEATs_iter3_plus_AS2M.pt"

# Per-dataset stimulus + output layout (CONTRACTS §3). Audio/text features are
# story-level and shared across subjects.
#   ds002345 (huth):  stimuli are `{story}_audio.wav`
#   ds003020 (lebel2023): stimuli are `{story}.wav`
DATASETS = {
    "huth": {
        "stim_root": "/data/raw/ds002345/stimuli",
        "stories_root": "/data/features/mary/huth/_stories",
        "glob": "*_audio.wav",
    },
    "lebel2023": {
        "stim_root": "/data/raw/ds003020/stimuli",
        "stories_root": "/data/features/mary/lebel2023/_stories",
        "glob": "*.wav",
    },
    # ds004996 (NeuroEngage): operator (partner) audio — human voice for human-group
    # subjects, robot voice for robot-group subjects. Demo subset, flat dir; story
    # key = filename stem (e.g. "sub-01_operator_run-01").
    "neuroengage": {
        "stim_root": "/data/raw/ds004996_demo/operator",
        "stories_root": "/data/features/mary/neuroengage/_stories",
        "glob": "*_operator_run-*.wav",
    },
    # CNeuroMod / Algonauts-2025: .mkv movies (audio decoded from video via
    # ffmpeg/librosa). Canonical story key matches the fMRI `task-` label.
    "cneuromod": {
        "stim_root": "/data/raw/algonauts2025/stimuli/movies",
        "stories_root": "/data/features/mary/cneuromod/_stories",
        "glob": "**/*.mkv",
        "is_video": True,
    },
    # ds004488 (HAD): audio+video action clips. Audio decoded from the .mp4.
    # Per-CLIP features keyed `{Category}__{clip}`; only clips shown to the target
    # subjects (events.tsv) are featurized. assemble_had_features.py then drops
    # each clip at its event onset on the 2 Hz run grid.
    "had": {
        "stim_root": "/data/raw/ds004488/stimuli",
        "stories_root": "/data/features/mary/had/_stories",
        "glob": "**/*.mp4",
        "is_video": True,
        "is_had": True,
    },
}
SKIP_STORIES = {"auditory_localizer"}

_UNILM_RAW = "https://raw.githubusercontent.com/microsoft/unilm/master/beats"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "curl")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "timm==0.4.5",        # official BEATs backbone depends on timm
        "soundfile==0.12.1",
        "librosa==0.10.2",
        "numpy>=1.26,<3",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
        "einops",
    )
    # Vendor the official microsoft/unilm BEATs source (inference subset).
    .run_commands(
        "mkdir -p /opt/beats",
        f"curl -fsSL {_UNILM_RAW}/BEATs.py    -o /opt/beats/BEATs.py",
        f"curl -fsSL {_UNILM_RAW}/backbone.py -o /opt/beats/backbone.py",
        f"curl -fsSL {_UNILM_RAW}/modules.py  -o /opt/beats/modules.py",
        "touch /opt/beats/__init__.py",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"})
)

data_volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("mary-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App(APP_NAME)


def _story_name(wav: Path) -> str:
    stem = wav.stem
    return stem[:-6] if stem.endswith("_audio") else stem


def _load_audio_16k_mono(path: str):
    """16 kHz mono float32. .wav via librosa; video (.mp4/.mkv) via ffmpeg
    (librosa's audioread backend isn't wired for mp4). Empty array if no audio."""
    import subprocess
    import numpy as np

    ext = Path(path).suffix.lower()
    if ext in (".wav", ".flac", ".ogg"):
        import librosa
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        return audio.astype(np.float32)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# --- CNeuroMod / Algonauts-2025: canonical story keys aligned to fMRI tasks ---
import re as _re

_CN_PRUNE = {".git", ".datalad", ".github", "__pycache__"}


def _cn_norm(s: str) -> str:
    return _re.sub(r"[^a-z0-9]", "", s.lower())


def _cn_story_key(path: Path, root: Path) -> str:
    """movie10/bourne/bourne05.mkv -> 'movie10_bourne05';
    friends/s1/friends_s01e24a.mkv -> 'friends_s01e24a'; aligned to fMRI task-."""
    rel = path.relative_to(root)
    group = rel.parts[0] if rel.parts else ""
    stem = path.stem
    if group == "friends":
        m = _re.search(r"(s\d{2}e\d{2}[a-d]?)", stem.lower())
        return f"friends_{m.group(1) if m else _cn_norm(stem)}"
    return f"{group}_{_cn_norm(stem)}"


def _cn_find_media(root: Path, only: set[str] | None) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for p in root.rglob("*.mkv"):
        if any(part in _CN_PRUNE for part in p.parts):
            continue
        key = _cn_story_key(p, root)
        if only and not (key in only or key.split("_", 1)[0] in only
                         or any(o in key for o in only)):
            continue
        out.append((p, key))
    return sorted(out, key=lambda t: t[1])


# --- HAD (ds004488): per-clip media restricted to clips the subjects saw -------
_HAD_SUBJECTS = ("sub-01", "sub-02")
_HAD_SES = "ses-action01"


def _had_find_media(stim_root: Path, only: set[str] | None) -> list[tuple[Path, str]]:
    """[(mp4_path, clip_key)] for unique clips referenced in sub-01/sub-02
    events.tsv. clip_key = `{Category}__{clip}` (== video extractors' _rel_story)."""
    import csv as _csv

    raw = stim_root.parent  # /data/raw/ds004488
    seen: dict[str, str] = {}
    for sub in _HAD_SUBJECTS:
        func = raw / sub / _HAD_SES / "func"
        if not func.exists():
            continue
        for ev in sorted(func.glob(f"{sub}_{_HAD_SES}_task-action_run-*_events.tsv")):
            with ev.open() as f:
                for r in _csv.DictReader(f, delimiter="\t"):
                    stim = (r.get("stim_file") or "").strip()
                    if not stim or stim in ("n/a", "nan"):
                        continue
                    key = (stim[:-4] if stim.lower().endswith(".mp4") else stim)
                    key = key.replace("/", "__")
                    seen.setdefault(key, stim)
    media: list[tuple[Path, str]] = []
    for key, stim in sorted(seen.items()):
        if only and not any(o in key for o in only):
            continue
        media.append((stim_root / stim, key))
    return media


def _load_beats():
    """Return (model, runner) where runner(audio_1d_tensor) -> (T_native, 768)
    hidden states on CPU as a float32 numpy array. Loads the EXACT official
    BEATs_iter3_plus_AS2M checkpoint.
    """
    import sys
    import torch
    from huggingface_hub import hf_hub_download

    if "/opt/beats" not in sys.path:
        sys.path.insert(0, "/opt/beats")
    from BEATs import BEATs, BEATsConfig  # vendored official source

    ckpt_path = hf_hub_download(
        repo_id=BEATS_REPO, filename=BEATS_CKPT, repo_type="dataset"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)

    def runner(wav_1d):  # (n_samples,) float tensor @16k
        # Zero-pad clips shorter than the patch-embed minimum so the BEATs
        # spectrogram is never smaller than the 16x16 kernel (avoids the
        # "Calculated padded input size per channel ... Kernel size: (16 x 16)"
        # RuntimeError on very short audio / short trailing chunks).
        if wav_1d.shape[0] < MIN_AUDIO_SAMPLES:
            pad_n = MIN_AUDIO_SAMPLES - wav_1d.shape[0]
            wav_1d = torch.cat([wav_1d, torch.zeros(pad_n, dtype=wav_1d.dtype)])
        x = wav_1d.unsqueeze(0).cuda()                 # (1, n)
        pad = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        with torch.no_grad():
            feats, _ = model.extract_features(x, padding_mask=pad)
        return feats.squeeze(0).float().cpu().numpy()  # (T_native, 768)

    print(f"[{STREAM}] loaded official BEATs ({BEATS_REPO}::{BEATS_CKPT})")
    return model, runner


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/cache": hf_cache_volume},
    secrets=[hf_secret],
)
def extract(dataset: str = "huth", only: str | None = None, overwrite: bool = False) -> dict:
    import time
    import numpy as np
    import torch
    import librosa

    cfg = DATASETS[dataset]
    stim_dir = Path(cfg["stim_root"])
    out_root = Path(cfg["stories_root"])

    if cfg.get("is_had"):
        only_set = {s.strip() for s in only.split(",")} if only else None
        media = _had_find_media(stim_dir, only_set)
    elif cfg.get("is_video"):
        only_set = {s.strip() for s in only.split(",")} if only else None
        media = _cn_find_media(stim_dir, only_set)
    else:
        wavs = sorted(w for w in stim_dir.glob(cfg["glob"])
                      if _story_name(w) not in SKIP_STORIES)
        if only:
            wavs = [w for w in wavs if _story_name(w) == only]
        media = [(w, _story_name(w)) for w in wavs]
    print(f"[{STREAM}] {len(media)} stimuli under {stim_dir}"
          + (f" (filtered to '{only}')" if only else ""))

    _model, runner = _load_beats()

    timings: dict[str, float] = {}
    for wav, story in media:
        out = out_root / story / f"{STREAM}.npy"
        if out.exists() and not overwrite:
            print(f"  skip {out} (exists)")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        audio = _load_audio_16k_mono(str(wav))
        duration_s = len(audio) / TARGET_SR
        n_steps = int(duration_s * TARGET_RATE_HZ)
        if n_steps <= 0:
            print(f"  {story}: too short ({duration_s:.3f}s, 0 output steps), skipping")
            continue

        # A whole story shorter than the patch-embed minimum would otherwise
        # require padding the *entire* clip, which inflates the native-rate
        # estimate. Skip-and-log these rather than emit features for a clip that
        # is effectively all padding.
        if len(audio) < MIN_AUDIO_SAMPLES:
            print(f"  {story}: too short ({duration_s:.3f}s < {MIN_AUDIO_SECONDS}s min), skipping")
            continue

        native_chunks: list[np.ndarray] = []
        chunk_samples = int(CHUNK_SECONDS * TARGET_SR)
        n_padded_samples = 0
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            if len(chunk) < MIN_AUDIO_SAMPLES:
                n_padded_samples += MIN_AUDIO_SAMPLES - len(chunk)
            wav_t = torch.from_numpy(chunk).float()
            native_chunks.append(runner(wav_t))
        native = np.concatenate(native_chunks, axis=0)  # (T_native, 768)

        # Derive native rate from the (possibly padded) effective duration so the
        # down-sampling bin size stays correct when a short trailing chunk was
        # zero-padded up to the minimum.
        effective_duration_s = duration_s + n_padded_samples / TARGET_SR
        native_rate = native.shape[0] / effective_duration_s
        bin_size = max(1, int(round(native_rate / TARGET_RATE_HZ)))
        feats = np.empty((n_steps, HIDDEN_DIM), dtype=np.float16)
        for i in range(n_steps):
            s = i * bin_size
            e = min(s + bin_size, native.shape[0])
            seg = native[s:e] if e > s else native[-1:]
            feats[i] = seg.mean(axis=0).astype(np.float16)

        np.save(out, feats)
        data_volume.commit()
        dt = time.time() - t0
        timings[story] = dt
        print(f"  ✓ {story}: {feats.shape} ({duration_s:.0f}s, "
              f"native ~{native_rate:.1f} Hz) in {dt:.1f}s → {out}")

    total = sum(timings.values())
    print(f"[{STREAM}] done. {len(timings)} stories, {total:.1f}s total.")
    return {"stream": STREAM, "n": len(timings), "total_s": total, "timings": timings}


@app.local_entrypoint()
def main(dataset: str = "huth", only: str = "", overwrite: bool = False) -> None:
    res = extract.remote(dataset=dataset, only=only or None, overwrite=overwrite)
    print(f"\nResult: {res}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="huth", choices=list(DATASETS))
    ap.add_argument("--only", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args.dataset, args.only, args.overwrite)
