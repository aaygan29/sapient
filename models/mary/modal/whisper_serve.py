"""whisper_serve.py — self-hosted faster-whisper transcription endpoint.

Serves the TRANSCRIPT stream of the Sapient app's grounded-evidence pipeline
(`api/_evidence.ts`). The app down-mixes any media to 16 kHz mono wav, base64s
it, and POSTs JSON to `MARY_WHISPER_URL`. This endpoint runs faster-whisper and
returns timestamped segments with the raw Whisper signals the app uses to derive
per-segment confidence (avg_logprob / no_speech_prob).

Contract (locked to api/_evidence.ts `transcribe()`):
  REQUEST  (POST application/json):
    {
      "auth_token":      str,   # must equal MARY_PIPELINE_SECRET (same secret
                                #   the live Mary `submit` endpoint uses)
      "audio_base64":    str,   # base64 of a 16 kHz mono wav
      "word_timestamps": bool   # app sends false; honored if true
    }
  RESPONSE (200 application/json):
    {
      "segments": [
        { "start": float, "end": float, "text": str,
          "avg_logprob": float, "no_speech_prob": float,
          "confidence": float, "words": [...]? }
      ],
      "language": str,
      "language_probability": float,
      "duration": float,
      "model": str
    }
  AUTH:  `auth_token` field in the JSON body == env MARY_PIPELINE_SECRET.
         401 on mismatch/missing (mirrors serve.py `submit`).

The app parses each segment's start/end/text and derives confidence from
`confidence` (preferred) else avg_logprob discounted by no_speech_prob. We
provide all three so the app's `segmentConfidence()` has its best signal.

Cost: CPU-only (cheap), small model, scaledown_window=60s so it idles to zero.
"""

from __future__ import annotations

import base64
import math
import os
import tempfile

import modal

# `large-v3` for best transcription accuracy — proper nouns, brand names + prices
# (which drive the Buy Signal read) and word-level timings. Ads/creatives are short
# clips, so large-v3 int8 on CPU stays within the (bumped) timeout, and the grounded
# report is computed once on run-completion + cached, so the user never waits on it.
# Override with WHISPER_MODEL env (e.g. "distil-large-v3" for ~6x speed) — no code change.
MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")

# Cache the (small) CTranslate2 weights on a volume so cold starts don't re-download.
MODEL_CACHE = modal.Volume.from_name("mary-whisper-cache", create_if_missing=True)
MODEL_CACHE_MOUNT = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # libsndfile/decoder support for arbitrary wav variants
    .pip_install("faster-whisper==1.0.3", "fastapi[standard]", "requests")
    .env({"HF_HOME": "/models/hf", "WHISPER_MODEL": MODEL_NAME})
)

app = modal.App("mary-whisper")

# Reuse the SAME shared secret the live Mary endpoint authenticates with; it
# exposes env var MARY_PIPELINE_SECRET, which the app sends as `auth_token`.
SECRETS = [modal.Secret.from_name("mary-pipeline-secret")]


def _avg_logprob_to_conf(avg_logprob: float, no_speech_prob: float) -> float:
    """Map Whisper's signals to a 0-1 confidence.

    faster-whisper's avg_logprob is a per-token mean log-prob (≈ -1.5..0). We map
    it to [0,1] the same way the app's fallback does (1 + lp/1.5), then discount
    by no_speech_prob so likely-silence/garble segments score low. Providing an
    explicit `confidence` means the app uses this directly (its preferred path).
    """
    if avg_logprob is None or math.isnan(avg_logprob):
        c = 0.5
    else:
        c = 1.0 + (avg_logprob / 1.5)
    if no_speech_prob is not None and not math.isnan(no_speech_prob):
        c *= 1.0 - max(0.0, min(1.0, no_speech_prob))
    return max(0.0, min(1.0, c))


@app.cls(
    image=image,
    secrets=SECRETS,
    volumes={MODEL_CACHE_MOUNT: MODEL_CACHE},
    cpu=4.0,               # more threads → large-v3 int8 stays quick on short clips
    memory=8192,           # headroom for the large-v3 weights + audio buffers
    scaledown_window=60,   # idle to zero quickly → tiny budget
    timeout=600,           # large-v3 is slower; ample margin for longer creatives
)
class Whisper:
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel

        # int8 on CPU is cost-efficient; large-v3 gives the best word/segment
        # timings + proper-noun accuracy. Cached on the volume after first load.
        self.model_name = os.environ.get("WHISPER_MODEL", MODEL_NAME)
        self.model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            download_root="/models/faster-whisper",
        )

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, payload: dict) -> dict:
        from fastapi import HTTPException

        # --- Auth: identical shape to serve.py `submit` -------------------
        expected = os.environ.get("MARY_PIPELINE_SECRET")
        if not expected or payload.get("auth_token") != expected:
            raise HTTPException(status_code=401, detail="bad auth_token")

        audio_b64 = payload.get("audio_base64")
        if not audio_b64:
            raise HTTPException(status_code=400, detail="audio_base64 required")
        want_words = bool(payload.get("word_timestamps", False))

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="audio_base64 not valid base64")

        # faster-whisper takes a path or file-like; write the wav to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            # Transcribe, forcing the lazy generator to evaluate INSIDE the try so
            # we catch faster-whisper's "max() arg is an empty sequence" — which it
            # raises when VAD removes ALL audio (music-only / speechless clips).
            # Recovery ladder: VAD on → VAD off → honest empty transcript (never
            # 500 the whole grounded report just because a clip has no speech).
            def _run(vad: bool):
                seg_iter, info = self.model.transcribe(
                    tmp.name, word_timestamps=want_words, vad_filter=vad,
                )
                return list(seg_iter), info

            info = None
            segs = []
            try:
                segs, info = _run(True)
            except Exception:
                try:
                    segs, info = _run(False)  # VAD nuked everything → retry raw
                except Exception:
                    segs, info = [], None  # genuinely nothing transcribable

            if info is None:
                return {
                    "segments": [], "language": None, "language_probability": 0.0,
                    "duration": 0.0, "model": self.model_name,
                }

            out_segments = []
            for s in segs:
                avg_lp = getattr(s, "avg_logprob", None)
                nsp = getattr(s, "no_speech_prob", None)
                seg = {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": (s.text or "").strip(),
                    "avg_logprob": (None if avg_lp is None else float(avg_lp)),
                    "no_speech_prob": (None if nsp is None else float(nsp)),
                    "confidence": _avg_logprob_to_conf(
                        avg_lp if avg_lp is not None else float("nan"),
                        nsp if nsp is not None else float("nan"),
                    ),
                }
                if want_words and getattr(s, "words", None):
                    seg["words"] = [
                        {
                            "start": float(w.start),
                            "end": float(w.end),
                            "word": w.word,
                            "probability": float(getattr(w, "probability", 0.0) or 0.0),
                        }
                        for w in s.words
                    ]
                out_segments.append(seg)

        return {
            "segments": out_segments,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration": float(info.duration),
            "model": self.model_name,
        }


@app.local_entrypoint()
def smoke():
    """Generate a tiny WAV, call the deployed endpoint, print the parsed shape.

    Run AFTER `modal deploy whisper_serve.py`. Requires MARY_PIPELINE_SECRET in
    the local env (same value as the Modal secret) and the deployed URL in
    WHISPER_URL, else it prints how to set them.
    """
    import json
    import struct
    import urllib.request
    import wave

    url = os.environ.get("WHISPER_URL", "")
    secret = os.environ.get("MARY_PIPELINE_SECRET", "")
    if not url or not secret:
        print("set WHISPER_URL (deployed endpoint) and MARY_PIPELINE_SECRET to smoke-test")
        return

    # 1.5s 16 kHz mono sine tone (no speech — exercises the pipeline + signals).
    sr, dur, freq = 16000, 1.5, 220.0
    frames = bytearray()
    for i in range(int(sr * dur)):
        v = int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr))
        frames += struct.pack("<h", v)
    buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(buf.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    with open(buf.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    body = json.dumps(
        {"auth_token": secret, "audio_base64": b64, "word_timestamps": False}
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        print("HTTP", resp.status)
        print(json.dumps(json.loads(resp.read()), indent=2))
