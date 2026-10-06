"""Verify the REAL Digital Brain engine loads a trained model and produces valid signals.

Skips automatically unless torch + the digital-brain repo + a geo_*.pkl are present
(so CI / fresh checkouts don't fail). CLIP is NOT exercised here — we feed synthetic
1024-d features straight into the real model's readout, which proves the weights run.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pytest

_DB_REPO = "/Users/aayushgandhi/Desktop/Research/Neuro-AI/digital-brain"
_MODELS = sorted(glob.glob(f"{_DB_REPO}/results/algonauts2023/models/geo_*.pkl"))

pytest.importorskip("torch")
pytest.importorskip("sklearn")
if not _MODELS:
    pytest.skip("digital-brain model not present", allow_module_level=True)


@pytest.fixture(scope="module")
def db_engine():
    os.environ["SAPIENT_DB_REPO"] = _DB_REPO
    os.environ["SAPIENT_DB_MODEL"] = _MODELS[0]
    if _DB_REPO not in sys.path:
        sys.path.insert(0, _DB_REPO)
    from sapient_serving.engine.digital_brain import DigitalBrainEngine
    return DigitalBrainEngine()


def test_real_model_loads_and_predicts(db_engine):
    feats = np.random.default_rng(0).standard_normal((1, 1024)).astype("float32")
    pred = db_engine.predict_from_features(feats)
    assert pred.signals.recommendation in ("Buy", "Hold", "Sell")
    assert 0 <= pred.signals.purchase_intent <= 100
    keys = {c.key for c in pred.signals.constructs}
    assert {"face", "scene", "body", "visual", "text", "higher"} == keys
    assert pred.image_data_uri.startswith("data:image/png;base64,")
