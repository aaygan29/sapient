"""Per-subject head routing test (ORCLE-Nano §3 step 8).

Different subject_idx must route through different per-subject head weights → different
output for identical features. Runnable with pytest OR directly.
"""

from __future__ import annotations

import torch

from mary.adapter import STREAM_DIMS
from mary.model import MaryConfig, MaryModel


def _model():
    torch.manual_seed(0)
    return MaryModel(MaryConfig(n_subjects=4)).eval()


def test_different_subjects_differ():
    model = _model()
    feats = {s: torch.randn(1, 200, STREAM_DIMS[s]) for s in STREAM_DIMS}
    with torch.no_grad():
        out0 = model(feats, torch.tensor([0]))
        out1 = model(feats, torch.tensor([1]))
    diff = (out0 - out1).abs().max().item()
    assert diff > 1e-5, f"subject routing produced identical output (max diff {diff})"


def test_same_subject_deterministic():
    model = _model()
    feats = {s: torch.randn(1, 200, STREAM_DIMS[s]) for s in STREAM_DIMS}
    with torch.no_grad():
        a = model(feats, torch.tensor([2]))
        b = model(feats, torch.tensor([2]))
    assert torch.allclose(a, b), "eval-mode forward should be deterministic"


def test_batched_mixed_subjects():
    model = _model()
    feats = {s: torch.randn(2, 200, STREAM_DIMS[s]) for s in STREAM_DIMS}
    # Same features for both rows, different subjects -> rows must differ.
    for s in feats:
        feats[s][1] = feats[s][0]
    with torch.no_grad():
        out = model(feats, torch.tensor([0, 3]))
    diff = (out[0] - out[1]).abs().max().item()
    assert diff > 1e-5, f"mixed-subject batch rows identical (max diff {diff})"


if __name__ == "__main__":
    test_different_subjects_differ()
    test_same_subject_deterministic()
    test_batched_mixed_subjects()
    print("test_subject_head: ALL PASS")
