"""Forced word-level alignment for stimulus transcripts.

`extract_text_features.py` (spec §2.3) expects sibling `<stem>.words.json`
files of the form:
    [{"word": "the", "onset": 0.10, "offset": 0.28}, ...]
CNeuroMod ships stimulus media + sometimes subtitles, but NOT word-level
alignment. This script closes that gap by running whisper-timestamped over
each audio source and writing the JSON.

Modal app name: sapient-1-forced-align (A100-40GB; whisper-large-v3 fits
comfortably). Output is written next to each input audio file in the
shared `sapient-data` volume.

Usage:
  modal run data/forced_align.py \
      --stim-root /data/raw/cneuromod/fmriprep/friends/sourcedata/friends/stimuli
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

APP_NAME = "sapient-1-forced-align"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "openai-whisper==20240930",
        "whisper-timestamped==1.15.4",
        "librosa==0.10.2",
        "numpy>=1.26,<3",
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
def align(stim_root: str, model_size: str = "large-v3") -> dict:
    """For each audio source under stim_root, write a sibling .words.json.

    Already-aligned stimuli are skipped (idempotent). Returns a count summary.
    """
    import json
    import whisper_timestamped as whisperts

    stim_dir = Path(stim_root)
    audio_paths = sorted(
        p for p in stim_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".mkv", ".webm", ".mov"}
    )
    print(f"Found {len(audio_paths)} audio sources under {stim_dir}.")

    print(f"Loading whisper-{model_size}...")
    model = whisperts.load_model(model_size, device="cuda")

    n_done = n_skip = n_fail = 0
    for src in audio_paths:
        words_path = src.with_suffix(".words.json")
        if words_path.exists():
            n_skip += 1
            continue
        try:
            result = whisperts.transcribe(model, str(src), language=None)
            words_out: list[dict] = []
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    # whisper-timestamped yields {text, start, end, confidence}
                    txt = w.get("text", "").strip()
                    if not txt:
                        continue
                    words_out.append({
                        "word":   txt,
                        "onset":  float(w["start"]),
                        "offset": float(w["end"]),
                    })
            if not words_out:
                print(f"  ⚠️  {src.name}: no word timings produced")
                n_fail += 1
                continue
            with words_path.open("w") as f:
                json.dump(words_out, f, indent=None)
            volume.commit()
            print(f"  ✓ {src.name}: {len(words_out)} words → {words_path.name}")
            n_done += 1
        except Exception as e:
            print(f"  ✗ {src.name}: {e!r}")
            n_fail += 1

    return {"done": int(n_done), "skipped": int(n_skip), "failed": int(n_fail)}


@app.local_entrypoint()
def main(
    stim_root: str = "/data/raw/cneuromod/fmriprep/friends/sourcedata/friends/stimuli",
    model_size: str = "large-v3",
) -> None:
    """`modal run data/forced_align.py --stim-root /data/raw/...`"""
    result = align.remote(stim_root, model_size)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim-root", required=True)
    ap.add_argument("--model-size", default="large-v3")
    args = ap.parse_args()
    main(args.stim_root, args.model_size)
