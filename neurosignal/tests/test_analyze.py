"""Multimodal analyze() + the headline metrics (manipulation, sycophancy, ...)."""
from __future__ import annotations

import pytest

from neurosignal import analyze


def _metric(a, key):
    return next(m.score for m in a.metrics if m.key == key)


def test_text_returns_full_metric_suite():
    a = analyze(text="The quarterly report shows revenue rose four percent on stronger margins.")
    keys = {m.key for m in a.metrics}
    assert {"valence", "arousal", "engagement", "manipulation", "buy_sell", "sycophancy"} <= keys
    assert "text" in a.modalities
    assert len(a.constructs) == 7
    assert len(a.networks) >= 1


def test_sycophancy_higher_for_flattery():
    flattery = analyze(text="You're absolutely right! What a brilliant, amazing question — "
                            "I completely agree, you're a genius.")
    informative = analyze(text="Mitochondria generate ATP through oxidative phosphorylation "
                               "across the inner-membrane proton gradient.")
    assert _metric(flattery, "sycophancy") > _metric(informative, "sycophancy")


def test_manipulation_higher_for_hype():
    hype = analyze(text="SHOCKING! Unbelievable, urgent, incredible deal — act now, "
                        "you won't believe this amazing offer!")
    plain = analyze(text="The device measures temperature and logs the value to a file every ten seconds.")
    assert _metric(hype, "manipulation") >= _metric(plain, "manipulation")


def test_multimodal_with_precomputed_audio_features():
    a = analyze(
        text="bright neon crowd energy",
        audio={"energy": 0.9, "loudness_var": 0.8, "tempo": 0.7, "pitch_var": 0.5,
               "voicing": 0.6, "onset_rate": 0.7, "brightness": 0.6},
    )
    assert "audio" in a.modalities and "text" in a.modalities


def test_requires_some_input():
    with pytest.raises(ValueError):
        analyze()
