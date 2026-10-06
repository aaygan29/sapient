"""mary.extractors — single-file, callable feature extractors for live serving.

The `data/extract_*.py` apps are Modal BATCH jobs (dataset globs, volume writes,
caching). For the live Mary serving path we need the SAME frozen-backbone
featurize logic as plain, reusable functions that take one in-memory input and
return a `(T_2Hz, D_m)` float32 array — model loaded ONCE and reused warm.

Each extractor mirrors its `data/extract_*.py` counterpart exactly (same model
id, same 2 Hz binning) so live features match the training-time contract.

Active streams (CONTRACTS §0): whisper, beats, qwen_ctx. Video streams are
intentionally absent (untrained in the current checkpoint).
"""

from .audio import load_audio_16k_mono
from .whisper import WhisperExtractor
from .beats import BeatsExtractor
from .qwen_ctx import QwenCtxExtractor, words_from_text

__all__ = [
    "load_audio_16k_mono",
    "WhisperExtractor",
    "BeatsExtractor",
    "QwenCtxExtractor",
    "words_from_text",
]
