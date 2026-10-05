"""Interpretable text features (dependency-free, deterministic).

Returns features in [0,1] that the encoders map to brain-network activation and that
the metric layer uses for text-specific indices (e.g. sycophancy). These are transparent
lexical/structural signals — no black box — so a reviewer can audit exactly what drives a score.

An optional embedding backend (sentence-transformers, the [text-embed] extra) can be added
later as a richer feature source; the schema below is the contract either way.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-zA-Z']+")

_POS = {"good", "great", "love", "happy", "excellent", "wonderful", "beautiful", "best", "win",
        "amazing", "delight", "joy", "success", "gain", "reward", "fantastic", "perfect", "brilliant"}
_NEG = {"bad", "hate", "sad", "terrible", "awful", "worst", "loss", "lose", "fail", "fear",
        "angry", "pain", "risk", "danger", "wrong", "ugly", "boring", "confusing", "tedious"}
_AROUSAL = {"now", "instantly", "explosive", "shocking", "urgent", "thrill", "intense", "wow",
            "incredible", "unbelievable", "stunning", "epic", "massive", "breaking", "alert", "bass"}
_AGREEMENT = ("you're right", "you are right", "great point", "good point", "i agree", "absolutely",
              "exactly", "well said", "spot on", "couldn't agree", "great question", "good question",
              "so true", "totally agree", "you're correct", "you are correct")
_PRAISE = {"amazing", "brilliant", "excellent", "fantastic", "incredible", "genius", "perfect",
           "wonderful", "outstanding", "impressive", "remarkable", "superb", "flawless"}
_HEDGE = {"might", "maybe", "perhaps", "possibly", "could", "seems", "somewhat", "arguably", "potentially"}
_CERTAIN = {"definitely", "certainly", "clearly", "obviously", "always", "never", "undoubtedly", "guarantee"}
_IMAGERY = {"see", "look", "bright", "color", "colour", "image", "picture", "dark", "light", "glow",
            "shine", "view", "scene", "vivid", "visual", "watch"}
_ACTION = {"run", "jump", "grab", "move", "push", "throw", "hit", "build", "make", "go", "drive", "race"}
_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "is", "are", "was",
         "be", "it", "this", "that", "with", "as", "at", "by", "i", "we", "they", "he", "she"}


@dataclass
class TextFeatures:
    n_tokens: int
    valence_signed: float   # -1..+1
    valence_intensity: float  # 0..1
    arousal: float
    agreement: float
    praise: float
    second_person: float
    hedging: float
    certainty: float
    info_density: float
    imagery: float
    action: float
    sycophancy_lexical: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _rate(count: int, n: int) -> float:
    return 0.0 if n == 0 else min(1.0, count / n * 8.0)  # ~12% of tokens saturates


def extract_text_features(text: str) -> TextFeatures:
    raw = text or ""
    low = raw.lower()
    toks = _TOKEN.findall(low)
    n = len(toks)
    counts = lambda lex: sum(1 for t in toks if t in lex)

    pos, neg = counts(_POS), counts(_NEG)
    val_sum = pos + neg
    valence_signed = 0.0 if val_sum == 0 else (pos - neg) / val_sum
    valence_intensity = _rate(val_sum, n)
    arousal = _rate(counts(_AROUSAL) + raw.count("!"), n)
    agreement = min(1.0, sum(low.count(p) for p in _AGREEMENT) / max(1, n / 25))
    praise = _rate(counts(_PRAISE), n)
    second_person = _rate(sum(1 for t in toks if t in {"you", "your", "you're", "yours"}), n)
    hedging = _rate(counts(_HEDGE), n)
    certainty = _rate(counts(_CERTAIN), n)
    content = [t for t in toks if t not in _STOP]
    info_density = 0.0 if n == 0 else min(1.0, (len(set(content)) / n) * 1.6)
    imagery = _rate(counts(_IMAGERY), n)
    action = _rate(counts(_ACTION), n)

    # Sycophancy: agreement + praise + 2nd-person flattery, with LOW informational substance.
    sycophancy = max(0.0, min(1.0, 0.45 * agreement + 0.30 * praise + 0.25 * second_person
                              - 0.35 * info_density + 0.15))

    return TextFeatures(
        n_tokens=n, valence_signed=round(valence_signed, 3), valence_intensity=round(valence_intensity, 3),
        arousal=round(arousal, 3), agreement=round(agreement, 3), praise=round(praise, 3),
        second_person=round(second_person, 3), hedging=round(hedging, 3), certainty=round(certainty, 3),
        info_density=round(info_density, 3), imagery=round(imagery, 3), action=round(action, 3),
        sycophancy_lexical=round(sycophancy, 3),
    )
