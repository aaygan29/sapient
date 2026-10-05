"""Top-level multimodal entry point.

    from neurosignal import analyze
    a = analyze(text="You're absolutely right, what a brilliant question!")
    print(a.metrics)        # valence, arousal, engagement, manipulation, sycophancy, buy/sell

Multimodal: pass any of text / audio / video. text is extracted from a string; audio/video
accept either a precomputed feature dict (works today) or a file path (needs the [audio]/[video]
extras). The encoder maps features -> Yeo-7 networks; the metric layer scores them.
"""
from __future__ import annotations

from typing import Union

from .encoders import get_encoder
from .features.text import extract_text_features
from .metrics import compute_metrics
from .types import Analysis


def _audio_features(audio: Union[str, dict]) -> dict:
    if isinstance(audio, dict):
        return audio
    from .features.audio import extract_audio_features
    return extract_audio_features(audio)


def _video_features(video: Union[str, dict]) -> dict:
    if isinstance(video, dict):
        return video
    from .features.video import extract_video_features
    return extract_video_features(video)


def analyze(*, text: str | None = None, audio: Union[str, dict, None] = None,
            video: Union[str, dict, None] = None, encoder: str = "reference",
            brain_file=None) -> Analysis:
    """Analyze media for brain-activation constructs + neuro-behavioral metrics.

    brain_file: optional personal digital brain (a `neurosignal.enrollment.BrainFile`
    or a path stem). When supplied, the base encoder is wrapped so the read-out is
    PERSONALIZED to that enrolled subject instead of the average-brain read-out, and
    the result is stamped (`personalized=True`, `provenance=...`). When omitted,
    behavior is identical to before.
    """
    features: dict[str, object] = {}
    modalities: list[str] = []
    if text is not None:
        features["text"] = extract_text_features(text)
        modalities.append("text")
    if audio is not None:
        features["audio"] = _audio_features(audio)
        modalities.append("audio")
    if video is not None:
        features["video"] = _video_features(video)
        modalities.append("video")
    if not modalities:
        raise ValueError("provide at least one of: text, audio, video")

    enc = get_encoder(encoder)
    bf = None
    if brain_file is not None:
        from .enrollment import BrainFile, EnrolledEncoder
        bf = brain_file if isinstance(brain_file, BrainFile) else BrainFile.load(brain_file)
        enc = EnrolledEncoder(enc, bf)

    networks = enc.encode(features)
    if not networks:
        raise ValueError("encoder produced no network activation from the given inputs")

    analysis = compute_metrics(
        networks, text_features=features.get("text"),
        modalities=modalities, source=f"{enc.name} <- {'+'.join(modalities)}",
    )
    if bf is not None:
        analysis.personalized = True
        analysis.provenance = dict(bf.provenance or {}, subject=bf.subject,
                                   brain_file_fingerprint=bf.fingerprint)
        analysis.notes.append(
            f"Personalized read-out for enrolled subject '{bf.subject}' "
            f"(brain_file {bf.fingerprint or 'unsigned'}); scores are individual, not average-brain.")
    return analysis
