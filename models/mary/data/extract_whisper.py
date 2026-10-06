"""Mary stream `whisper` — Whisper-large-v3-turbo ENCODER features at 2 Hz, frozen.

CONTRACTS §2: D_m = 1280, grid = 2 Hz, output `(T_2Hz, 1280)` float16.
Purpose (01-§2): speech, phonetics, prosody.

Backbone: `openai/whisper-large-v3-turbo` (Apache-2.0). We use ONLY the audio
ENCODER (`model.encoder` / `model.get_encoder()`), which yields hidden states
of dim 1280 at the encoder's native frame rate (50 Hz: 1500 frames per the
fixed 30-s mel window). We process audio in 30-s windows (Whisper's native
context), concatenate the encoder hidden states across windows, then average
non-overlapping bins down to a 2 Hz grid → `(T_2Hz, 1280)`.

Audio/text features are identical across subjects for a story → written ONCE
per story to `/data/features/mary/huth/_stories/{story}/whisper.npy`
(CONTRACTS §3).

Modal app: mary-features-whisper (GPU A100-40GB). Frozen, eval mode, no grad.
HF downloads cached on the `mary-hf-cache` volume to avoid re-downloads.

Usage:
  modal run data/extract_whisper.py                                  # all ds002345 (huth) stories
  modal run data/extract_whisper.py --only pieman                    # validate one huth story
  modal run data/extract_whisper.py --dataset lebel2023              # all ds003020 stories
  modal run data/extract_whisper.py --dataset lebel2023 --only buck  # validate one lebel story
  modal run data/extract_whisper.py --dataset cneuromod --only movie10_bourne05   # one Bourne clip
  modal run --detach data/extract_whisper.py --dataset cneuromod --only movie10   # all movie10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "mary-features-whisper"
MODEL_ID = "openai/whisper-large-v3-turbo"
STREAM = "whisper"
HIDDEN_DIM = 1280
TARGET_RATE_HZ = 2.0
TARGET_SR = 16000
WINDOW_SECONDS = 30.0   # Whisper's fixed mel context window

# Per-dataset stimulus + output layout. Audio/text features are story-level and
# shared across subjects, so they're keyed by dataset (CONTRACTS §3).
#   ds002345 (huth):  stimuli are `{story}_audio.wav`
#   ds003020 (lebel2023): stimuli are `{story}.wav`
#   cneuromod (Algonauts-2025): stimuli are `.mkv` MOVIES (audio decoded from the
#     video). librosa/ffmpeg reads .mkv directly. The canonical story key matches
#     the fMRI `task-` label so a later manifest can pair them:
#       movie10/bourne/bourne05.mkv      -> "movie10_bourne05"
#       friends/s1/friends_s01e24a.mkv   -> "friends_s01e24a"
#       ood/chaplin/chaplin1.mkv         -> "ood_chaplin1"
#     (rglob walks the movie tree; `.git`/`.datalad` dirs are pruned.)
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
    # ds004996 (NeuroEngage): the operator (partner) audio — a HUMAN voice for
    # human-group subjects, a ROBOT voice for robot-group subjects. This is what
    # the participant heard. Demo subset lives flat under ds004996_demo/operator;
    # story key = filename stem (e.g. "sub-01_operator_run-01"), unique per run.
    "neuroengage": {
        "stim_root": "/data/raw/ds004996_demo/operator",
        "stories_root": "/data/features/mary/neuroengage/_stories",
        "glob": "*_operator_run-*.wav",
    },
    "cneuromod": {
        "stim_root": "/data/raw/algonauts2025/stimuli/movies",
        "stories_root": "/data/features/mary/cneuromod/_stories",
        "glob": "**/*.mkv",
        "is_video": True,
    },
    # ds004488 (HAD): audio+video action clips. Audio decoded from the .mp4
    # (librosa/ffmpeg). Per-CLIP features, keyed `{Category}__{clip}` to match the
    # video extractors. We only featurize the clips actually shown to the target
    # subjects (events.tsv), then `data/assemble_had_features.py` drops each clip
    # at its event onset on the 2 Hz run grid.
    "had": {
        "stim_root": "/data/raw/ds004488/stimuli",
        "stories_root": "/data/features/mary/had/_stories",
        "glob": "**/*.mp4",
        "is_video": True,
        "is_had": True,
    },
}
# Stories that are not narrative stimuli (skip for lebel2023).
SKIP_STORIES = {"auditory_localizer"}

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
        "safetensors>=0.4",
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
    """Load audio at 16 kHz mono as float32 numpy. For .wav use librosa/soundfile;
    for video containers (.mp4/.mkv/...) decode the audio track via ffmpeg (librosa's
    audioread backend isn't wired for mp4). Returns a possibly-empty 1-D array
    (clips with no audio track -> empty)."""
    import subprocess
    import numpy as np

    ext = Path(path).suffix.lower()
    if ext in (".wav", ".flac", ".ogg"):
        import librosa
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        return audio.astype(np.float32)
    # ffmpeg → raw f32le mono @ TARGET_SR (works for mp4/mkv; empty if no audio)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# --- CNeuroMod / Algonauts-2025: canonical story keys aligned to fMRI tasks ---
import re as _re

_CN_GROUPS = ("movie10", "friends", "ood")
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
    """Walk the movie tree -> sorted [(mkv, story_key)], pruning git/datalad dirs."""
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
    """[(mp4_path, clip_key)] for the unique clips referenced in sub-01/sub-02
    events.tsv. clip_key = `{Category}__{clip}` (== video extractors' _rel_story).
    `only` filters by category or clip_key substring (for single-clip validation).
    """
    import csv as _csv

    raw = stim_root.parent  # /data/raw/ds004488
    seen: dict[str, str] = {}  # clip_key -> stim_file relpath
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
                    key = stim[:-4] if stim.lower().endswith(".mp4") else stim
                    key = key.replace("/", "__")
                    seen.setdefault(key, stim)
    media: list[tuple[Path, str]] = []
    for key, stim in sorted(seen.items()):
        if only and not any(o in key for o in only):
            continue
        media.append((stim_root / stim, key))
    return media


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
    from transformers import AutoFeatureExtractor, WhisperForConditionalGeneration

    cfg = DATASETS[dataset]
    stim_dir = Path(cfg["stim_root"])
    out_root = Path(cfg["stories_root"])

    # media = list of (path, story_name). For wav datasets the audio is the file;
    # for cneuromod the audio is decoded from the .mkv (ffmpeg/librosa handles it).
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

    print(f"[{STREAM}] loading ENCODER of {MODEL_ID} ...")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    full = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    )
    encoder = full.get_encoder().eval().cuda()
    for p in encoder.parameters():
        p.requires_grad_(False)

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
            print(f"  {story}: too short, skipping")
            continue

        # Encode each 30-s window; collect native (~50 Hz) hidden states.
        native_chunks: list[np.ndarray] = []
        win = int(WINDOW_SECONDS * TARGET_SR)
        for start in range(0, len(audio), win):
            chunk = audio[start:start + win]
            chunk_secs = len(chunk) / TARGET_SR
            inputs = fe(
                chunk, sampling_rate=TARGET_SR, return_tensors="pt"
            ).to("cuda", torch.float16)
            with torch.no_grad():
                hs = encoder(inputs.input_features).last_hidden_state.squeeze(0)
            # Encoder always emits 1500 frames for the padded 30-s window; keep
            # only the frames covering real audio (proportional to chunk_secs).
            n_native = hs.shape[0]
            n_keep = max(1, int(round(n_native * (chunk_secs / WINDOW_SECONDS))))
            native_chunks.append(hs[:n_keep].float().cpu().numpy())

        native = np.concatenate(native_chunks, axis=0)  # (T_native, 1280)
        native_rate = native.shape[0] / duration_s
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
        print(f"  ✓ {story}: {feats.shape} ({duration_s:.0f}s audio, "
              f"~{n_steps/duration_s:.2f} Hz) in {dt:.1f}s → {out}")

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
