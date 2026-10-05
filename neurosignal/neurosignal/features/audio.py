"""Audio feature extraction (interface + schema).

The reference encoder consumes these [0,1] features; raw-audio extraction needs the
[audio] extra (librosa). Implement extract_audio_features to populate this schema, or pass
a precomputed dict to analyze(audio=...).

Schema: {energy, loudness_var, tempo, pitch_var, voicing, onset_rate, brightness}
  energy/loudness_var -> arousal (limbic); tempo/onset_rate -> attention (dorsal);
  pitch_var/voicing -> speech salience (somatomotor/ventral attention).
"""
from __future__ import annotations

AUDIO_FEATURE_SCHEMA = ("energy", "loudness_var", "tempo", "pitch_var", "voicing", "onset_rate", "brightness")


def extract_audio_features(path: str) -> dict:  # pragma: no cover - needs [audio]
    raise NotImplementedError(
        "Raw-audio extraction needs the [audio] extra (librosa). Implement here to fill "
        f"AUDIO_FEATURE_SCHEMA={AUDIO_FEATURE_SCHEMA}, or pass a precomputed feature dict to analyze(audio=...)."
    )
