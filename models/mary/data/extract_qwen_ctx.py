"""Mary stream `qwen_ctx` — Qwen3-8B narrative-context word features at 2 Hz, frozen.

CONTRACTS §2: D_m = 4096, grid = 2 Hz, output `(T_2Hz, 4096)` float16.
Purpose (01-§2): narrative comprehension, long-range memory.
Text-encoding convention (02-§2 / TRIBE): features are forced-aligned WORDS —
each word's embedding is placed at its onset and resampled to the 2 Hz grid.

Pipeline (per story `.wav`):
  1. FORCED ALIGNMENT: run whisper-timestamped over the `.wav` to get the
     transcript with per-word {word, onset, offset}. (WhisperX-equivalent;
     we reuse the proven sapient1 forced-align approach. If a sibling
     `<story>_audio.words.json` already exists on the volume, we use it.)
  2. CONTEXT ENCODING: run `Qwen/Qwen3-8B` over the running transcript with a
     sliding window of the preceding CONTEXT_WORDS words ending at the current
     word; take the LAST-layer hidden states of the tokens belonging to the
     current word, mean-pool → one 4096-d vector per word.
  3. RESAMPLE: place each word vector at its onset->offset midpoint and average
     into 2 Hz bins → `(T_2Hz, 4096)`.

MODEL NOTE: `Qwen/Qwen3-8B` (hidden_size=4096, Apache-2.0) is on HF and used
directly — no substitution needed. Requires transformers>=4.51 for the
`qwen3` architecture. If it were unavailable, the documented fallback is the
nearest ~7-8B Qwen base (`Qwen/Qwen2.5-7B`, hidden_size=3584 — would change D_m,
so flagged loudly). Qwen3-8B was available at build time → no fallback taken.

Audio/text features are identical across subjects for a story → written ONCE
per story to `/data/features/mary/huth/_stories/{story}/qwen_ctx.npy` (CONTRACTS §3).

Modal app: mary-features-qwen_ctx (GPU A100-80GB). Frozen, eval mode, no grad.

Usage:
  modal run data/extract_qwen_ctx.py                                  # all ds002345 (huth) stories
  modal run data/extract_qwen_ctx.py --only pieman                    # validate one huth story
  modal run data/extract_qwen_ctx.py --dataset lebel2023              # all ds003020 stories
  modal run data/extract_qwen_ctx.py --dataset lebel2023 --only buck  # validate one lebel story
  modal run data/extract_qwen_ctx.py --dataset cneuromod --only movie10_bourne05  # one Bourne clip (uses .tsv)
  modal run --detach data/extract_qwen_ctx.py --dataset cneuromod --only movie10  # all movie10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "mary-features-qwen_ctx"
MODEL_ID = "Qwen/Qwen3-8B"
STREAM = "qwen_ctx"
HIDDEN_DIM = 4096
CONTEXT_WORDS = 512    # sliding narrative context ending at the current word
TARGET_RATE_HZ = 2.0
TARGET_SR = 16000
ALIGN_MODEL = "large-v3"

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
    # CNeuroMod / Algonauts-2025: .mkv movies WITH shipped .tsv transcripts.
    # No forced alignment needed — we read word/onset/offset straight from the
    # transcript (much faster + exact). Canonical story key matches the fMRI
    # `task-` label so a later manifest pairs (feature, fMRI).
    "cneuromod": {
        "stim_root": "/data/raw/algonauts2025/stimuli/movies",
        "transcripts_root": "/data/raw/algonauts2025/stimuli/transcripts",
        "stories_root": "/data/features/mary/cneuromod/_stories",
        "glob": "**/*.mkv",
        "is_video": True,
        "use_transcript": True,
    },
}
SKIP_STORIES = {"auditory_localizer"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    # Layer 1: build backbone first so pkg_resources/setuptools exist for the
    # legacy openai-whisper sdist build (which imports pkg_resources).
    .pip_install(
        "setuptools<81",                 # provides pkg_resources for legacy builds
        "wheel",
        "numpy>=1.26,<3",
        "torch==2.4.1",
        "torchaudio==2.4.1",
    )
    # Layer 2: install whisper deps WITHOUT build isolation so their setup.py
    # sees the setuptools/torch installed above (avoids ModuleNotFoundError:
    # pkg_resources during isolated build).
    .pip_install(
        "openai-whisper==20240930",
        "whisper-timestamped==1.15.4",
        extra_options="--no-build-isolation",
    )
    # Layer 3: the rest.
    .pip_install(
        "transformers==4.51.3",          # >=4.51 required for qwen3
        "accelerate>=0.34",
        "sentencepiece==0.2.0",
        "tokenizers>=0.21",
        "librosa==0.10.2",
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


# --- CNeuroMod / Algonauts-2025: story keys + transcript (.tsv) parsing -------
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


def _cn_find_transcript(mkv: Path, movies_root: Path, tx_root: Path) -> Path | None:
    """Mirror the movie's relative dir under transcripts/ and match by clip token
    (handles flat names like movie10_bourne05.tsv / friends_s01e24a.tsv)."""
    rel_dir = mkv.parent.relative_to(movies_root)
    want = _cn_norm(mkv.stem)
    search_dirs = [tx_root / rel_dir]
    if rel_dir.parts:
        search_dirs.append(tx_root / rel_dir.parts[0])  # group fallback
    search_dirs.append(tx_root)
    for d in search_dirs:
        if not d.exists():
            continue
        cands = [t for t in d.rglob("*.tsv") if _cn_norm(t.stem).endswith(want)]
        if not cands:
            cands = [t for t in d.rglob("*.tsv") if want in _cn_norm(t.stem)]
        if cands:
            return sorted(cands, key=lambda t: len(t.name))[0]
    return None


def _parse_cneuromod_tsv(tsv: Path) -> list[dict]:
    """Parse a CNeuroMod/Algonauts transcript TSV → [{word, onset, offset}].

    Robust to the two shipped layouts:
      (a) word-level rows: columns among {word|text} + {onset|start} +
          {offset|end} or {duration}.
      (b) per-TR rows: a `text_per_tr` (or `words_per_tr`) column holding the
          words spoken in each 1.49 s TR, optionally with `onsets_per_tr`
          (per-word onsets as a stringified list). When per-word onsets are
          absent we spread the TR's words uniformly across the TR window using
          the row's `onset`/TR index (×TR_SECONDS).
    """
    import ast
    import csv

    TR = 1.49  # CNeuroMod movie TR (s)
    rows = list(csv.DictReader(tsv.open(newline=""), delimiter="\t"))
    if not rows:
        return []
    cols = {c.lower(): c for c in rows[0].keys()}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    def _flt(v, default=None):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _as_list(v):
        if v is None or v == "":
            return []
        try:
            x = ast.literal_eval(v)
            return list(x) if isinstance(x, (list, tuple)) else [x]
        except (ValueError, SyntaxError):
            return v.split()

    word_c = col("word")
    on_c = col("onset", "start", "onsets")
    off_c = col("offset", "end")
    dur_c = col("duration")
    words = []

    # (a) genuine word-level rows
    if word_c and on_c:
        for r in rows:
            w = (r.get(word_c) or "").strip()
            if not w:
                continue
            onset = _flt(r.get(on_c))
            if onset is None:
                continue
            offset = _flt(r.get(off_c)) if off_c else None
            if offset is None:
                offset = onset + (_flt(r.get(dur_c), 0.3) if dur_c else 0.3)
            words.append({"word": w, "onset": onset, "offset": offset})
        if words:
            return words

    # (b) per-TR text rows
    txt_c = col("text_per_tr", "words_per_tr", "text", "words")
    onsets_c = col("onsets_per_tr", "word_onsets")
    if txt_c:
        for i, r in enumerate(rows):
            cell = r.get(txt_c)
            toks = _as_list(cell)
            toks = [str(t).strip() for t in toks if str(t).strip()]
            if not toks:
                continue
            # TR window start: explicit onset col, else row index × TR.
            tr_start = _flt(r.get(on_c)) if on_c else None
            if tr_start is None:
                tr_start = i * TR
            per_on = _as_list(r.get(onsets_c)) if onsets_c else []
            per_on = [_flt(x) for x in per_on]
            per_on = [x for x in per_on if x is not None]
            n = len(toks)
            step = TR / max(1, n)
            for j, w in enumerate(toks):
                onset = per_on[j] if j < len(per_on) else tr_start + j * step
                offset = (per_on[j + 1] if j + 1 < len(per_on)
                          else onset + step)
                words.append({"word": w, "onset": float(onset),
                              "offset": float(offset)})
    return words


def _forced_align(wav: Path, align_model) -> list[dict]:
    """Return [{word, onset, offset}, ...] for a story `.wav`.

    Uses a cached sibling `<stem>.words.json` if present (idempotent); else runs
    whisper-timestamped and caches it for the other text streams / re-runs.
    """
    import json
    import whisper_timestamped as whisperts

    words_path = wav.with_suffix(".words.json")
    if words_path.exists():
        with words_path.open() as f:
            return json.load(f)

    result = whisperts.transcribe(align_model, str(wav), language="en")
    words_out: list[dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            txt = w.get("text", "").strip()
            if not txt:
                continue
            words_out.append({
                "word": txt,
                "onset": float(w["start"]),
                "offset": float(w["end"]),
            })
    if words_out:
        try:
            with words_path.open("w") as f:
                json.dump(words_out, f)
            data_volume.commit()
        except Exception as e:  # non-fatal: still return timings
            print(f"    (could not cache words.json: {e!r})")
    return words_out


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/cache": hf_cache_volume},
    secrets=[hf_secret],
)
def extract(dataset: str = "huth", only: str | None = None, overwrite: bool = False) -> dict:
    import os
    import time
    import numpy as np
    import torch
    import whisper_timestamped as whisperts
    from transformers import AutoModel, AutoTokenizer

    cfg = DATASETS[dataset]
    stim_dir = Path(cfg["stim_root"])
    out_root = Path(cfg["stories_root"])
    use_transcript = cfg.get("use_transcript", False)
    tx_root = Path(cfg["transcripts_root"]) if cfg.get("transcripts_root") else None

    # media = [(stimulus_path, story_name)]. For cneuromod the "stimulus" is the
    # .mkv and words come from the shipped .tsv (no forced alignment).
    if cfg.get("is_video"):
        only_set = ({s.strip() for s in only.split(",") if s.strip()}
                    if only else None)
        media = _cn_find_media(stim_dir, only_set)
    else:
        wavs = sorted(w for w in stim_dir.glob(cfg["glob"])
                      if _story_name(w) not in SKIP_STORIES)
        if only:
            wanted = {s.strip() for s in only.split(",") if s.strip()}
            wavs = [w for w in wavs if _story_name(w) in wanted]
        media = [(w, _story_name(w)) for w in wavs]
    print(f"[{STREAM}] {len(media)} stimuli under {stim_dir}"
          + (f" (filtered to '{only}')" if only else ""))

    # The whisper aligner is only needed when we forced-align audio (wav
    # datasets). Transcript-driven datasets (cneuromod) skip it entirely.
    align_model = None
    if not use_transcript:
        print(f"[{STREAM}] loading aligner whisper-{ALIGN_MODEL} ...")
        # Cache the aligner weights on the persistent /cache volume so re-runs
        # don't re-download the ~3 GB checkpoint.
        _whisper_cache = "/cache/whisper"
        os.makedirs(_whisper_cache, exist_ok=True)
        align_model = whisperts.load_model(
            ALIGN_MODEL, device="cuda", download_root=_whisper_cache
        )
        # whisper-timestamped 1.15.4 reads cross-attention weights via a forward
        # hook, but openai-whisper 20240930 uses scaled_dot_product_attention
        # which returns None for those weights → AttributeError. Force the eager
        # attention path so the hook sees real attention tensors.
        import whisper.model as _wm
        if hasattr(_wm.MultiHeadAttention, "use_sdpa"):
            _wm.MultiHeadAttention.use_sdpa = False
        for _m in align_model.modules():
            if isinstance(_m, _wm.MultiHeadAttention):
                _m.use_sdpa = False

    print(f"[{STREAM}] loading {MODEL_ID} (hidden={HIDDEN_DIM}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        .eval()
        .cuda()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    assert model.config.hidden_size == HIDDEN_DIM, (
        f"hidden_size {model.config.hidden_size} != contract {HIDDEN_DIM}"
    )

    timings: dict[str, float] = {}
    for stim, story in media:
        out = out_root / story / f"{STREAM}.npy"
        if out.exists() and not overwrite:
            print(f"  skip {out} (exists)")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        if use_transcript:
            tpath = _cn_find_transcript(stim, stim_dir, tx_root)
            if tpath is None:
                print(f"  {story}: no transcript .tsv found, skipping")
                continue
            words = _parse_cneuromod_tsv(tpath)
            if not words:
                print(f"  {story}: transcript {tpath.name} parsed 0 words, skipping")
                continue
        else:
            words = _forced_align(stim, align_model)
            if not words:
                print(f"  {story}: no words from alignment, skipping")
                continue

        duration_s = float(words[-1]["offset"])
        n_steps = int(duration_s * TARGET_RATE_HZ)
        if n_steps <= 0:
            print(f"  {story}: zero-length, skipping")
            continue

        per_word = np.empty((len(words), HIDDEN_DIM), dtype=np.float32)
        for i, w in enumerate(words):
            ctx_start = max(0, i - CONTEXT_WORDS + 1)
            ctx_words = words[ctx_start : i + 1]
            text = " ".join(x["word"] for x in ctx_words)
            enc = tok(text, return_tensors="pt", add_special_tokens=True)
            input_ids = enc.input_ids.cuda()
            with torch.no_grad():
                hidden = model(input_ids).last_hidden_state.squeeze(0)  # (T,4096)

            prefix = " ".join(x["word"] for x in ctx_words[:-1])
            n_prefix = tok(
                prefix, return_tensors="pt", add_special_tokens=True
            ).input_ids.shape[1] if prefix else hidden.shape[0] - 1
            cur = hidden[n_prefix:].float()
            if cur.shape[0] == 0:
                cur = hidden[-1:].float()
            per_word[i] = cur.mean(dim=0).cpu().numpy()

        # Word-level (irregular) → 2 Hz grid by onset/offset midpoint bin-average.
        feats = np.zeros((n_steps, HIDDEN_DIM), dtype=np.float32)
        counts = np.zeros(n_steps, dtype=np.int64)
        midpoints = np.array([0.5 * (w["onset"] + w["offset"]) for w in words])
        bin_idx = np.clip(
            (midpoints * TARGET_RATE_HZ).astype(np.int64), 0, n_steps - 1
        )
        for emb, b in zip(per_word, bin_idx):
            feats[b] += emb
            counts[b] += 1
        nz = counts > 0
        feats[nz] = feats[nz] / counts[nz, None]
        feats = feats.astype(np.float16)

        np.save(out, feats)
        data_volume.commit()
        dt = time.time() - t0
        timings[story] = dt
        print(f"  ✓ {story}: {feats.shape} ({len(words)} words, "
              f"{duration_s:.0f}s) in {dt:.1f}s → {out}")

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
