"""Live `qwen_ctx` extractor — Qwen3-8B narrative-context features at 2 Hz.

Pure refactor of data/extract_qwen_ctx.py. Two entry points for live input:
  - `words_from_text(text)`     — raw typed text → synthetic-timed word list
    (no audio, so words are laid on a uniform speaking-rate grid).
  - `forced_align(wav_path)`    — audio → real word timings (whisper-timestamped);
    only needed for the audio path, imports its deps lazily.
Then `QwenCtxExtractor.encode_words(words)` → `(T_2Hz, 4096)` float32, mirroring
the training-time context-encoding + onset-midpoint binning. Contract: D_m = 4096.
"""
from __future__ import annotations

import numpy as np

MODEL_ID = "Qwen/Qwen3-8B"
HIDDEN_DIM = 4096
CONTEXT_WORDS = 512
TARGET_RATE_HZ = 2.0
DEFAULT_WPM = 150.0  # synthetic speaking rate for typed text (≈2.5 words/s)


def words_from_text(text: str, wpm: float = DEFAULT_WPM) -> list[dict]:
    """Typed text → [{word, onset, offset}] on a uniform speaking-rate grid.

    There is no audio for typed input, so we lay words at a natural cadence so
    the downstream 2 Hz binning produces a sensible temporal grid."""
    toks = [w for w in str(text).split() if w.strip()]
    rate = max(0.5, wpm / 60.0)  # words per second
    step = 1.0 / rate
    return [
        {"word": w, "onset": i * step, "offset": (i + 1) * step}
        for i, w in enumerate(toks)
    ]


def _words_from_whisper_native(wav_path: str, align_model_name: str,
                               device: str, download_root: str) -> list[dict]:
    """Per-word timings via openai-whisper's NATIVE word_timestamps=True.

    This is the robust default: it derives word boundaries from the model's own
    DTW on cross-attention internally and does NOT depend on whisper-timestamped's
    external forward-hook (which can raise "NoneType is not iterable" under
    openai-whisper 20240930). Returns [{word, onset, offset}]."""
    import os
    import whisper

    os.makedirs(download_root, exist_ok=True)
    model = whisper.load_model(align_model_name, device=device, download_root=download_root)
    result = whisper.transcribe(
        model, str(wav_path), language="en",
        word_timestamps=True, verbose=False, fp16=(device != "cpu"),
    )
    words: list[dict] = []
    for seg in result.get("segments", []):
        for w in (seg.get("words") or []):
            txt = (w.get("word") or "").strip()
            if not txt:
                continue
            words.append({"word": txt, "onset": float(w["start"]), "offset": float(w["end"])})
    return words


def forced_align(wav_path: str, align_model_name: str = "large-v3",
                 device: str = "cuda", download_root: str = "/cache/whisper") -> list[dict]:
    """Audio → [{word, onset, offset}].

    Primary path: openai-whisper native word_timestamps (robust, no external
    attention hook). Fallback: whisper-timestamped with the eager-attention
    monkeypatch (kept for parity with the training-time aligner). Lazy imports so
    the text-only serving image needn't carry the aligner deps."""
    try:
        words = _words_from_whisper_native(wav_path, align_model_name, device, download_root)
        if words:
            return words
    except Exception as e:
        print(f"[qwen_ctx] native whisper word_timestamps failed ({e!r}); "
              f"falling back to whisper-timestamped")

    import os
    import whisper_timestamped as whisperts

    os.makedirs(download_root, exist_ok=True)
    model = whisperts.load_model(align_model_name, device=device, download_root=download_root)
    # Force eager attention so whisper-timestamped's cross-attn hook sees tensors.
    import whisper.model as _wm
    if hasattr(_wm.MultiHeadAttention, "use_sdpa"):
        _wm.MultiHeadAttention.use_sdpa = False
    for _m in model.modules():
        if isinstance(_m, _wm.MultiHeadAttention):
            _m.use_sdpa = False

    result = whisperts.transcribe(model, str(wav_path), language="en")
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            txt = w.get("text", "").strip()
            if not txt:
                continue
            words.append({"word": txt, "onset": float(w["start"]), "offset": float(w["end"])})
    return words


class QwenCtxExtractor:
    """Frozen Qwen3-8B context encoder, loaded once and reused across requests."""

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = (
            AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval().to(device)
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        assert self.model.config.hidden_size == HIDDEN_DIM, (
            f"hidden_size {self.model.config.hidden_size} != contract {HIDDEN_DIM}"
        )

    def encode_words(self, words: list[dict]) -> np.ndarray:
        """[{word, onset, offset}] → (T_2Hz, 4096) float32."""
        import torch

        if not words:
            raise ValueError("no words to encode")
        duration_s = float(words[-1]["offset"])
        n_steps = int(duration_s * TARGET_RATE_HZ)
        if n_steps <= 0:
            raise ValueError("zero-length transcript")

        per_word = np.empty((len(words), HIDDEN_DIM), dtype=np.float32)
        for i, w in enumerate(words):
            ctx_start = max(0, i - CONTEXT_WORDS + 1)
            ctx_words = words[ctx_start : i + 1]
            text = " ".join(x["word"] for x in ctx_words)
            input_ids = self.tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
            with torch.no_grad():
                hidden = self.model(input_ids).last_hidden_state.squeeze(0)  # (T, 4096)
            prefix = " ".join(x["word"] for x in ctx_words[:-1])
            n_prefix = (
                self.tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids.shape[1]
                if prefix else hidden.shape[0] - 1
            )
            cur = hidden[n_prefix:].float()
            if cur.shape[0] == 0:
                cur = hidden[-1:].float()
            per_word[i] = cur.mean(dim=0).cpu().numpy()

        # Word-level (irregular) → 2 Hz grid by onset/offset midpoint bin-average.
        feats = np.zeros((n_steps, HIDDEN_DIM), dtype=np.float32)
        counts = np.zeros(n_steps, dtype=np.int64)
        midpoints = np.array([0.5 * (w["onset"] + w["offset"]) for w in words])
        bin_idx = np.clip((midpoints * TARGET_RATE_HZ).astype(np.int64), 0, n_steps - 1)
        for emb, b in zip(per_word, bin_idx):
            feats[b] += emb
            counts[b] += 1
        nz = counts > 0
        feats[nz] = feats[nz] / counts[nz, None]
        return feats
