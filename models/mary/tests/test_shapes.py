"""Shape contract tests for MaryModel (CONTRACTS.md §2/§6).

Runnable with pytest OR directly: `python tests/test_shapes.py`.
"""

from __future__ import annotations

import torch

from mary.adapter import STREAM_DIMS
from mary.model import MaryConfig, MaryModel


def _small_cfg() -> MaryConfig:
    # Smaller subject count for fast head; everything else faithful.
    return MaryConfig(n_subjects=4)


def _make_features(B: int, T_2Hz: int, streams) -> dict[str, torch.Tensor]:
    return {s: torch.randn(B, T_2Hz, STREAM_DIMS[s]) for s in streams}


def test_forward_all_six_streams():
    cfg = _small_cfg()
    model = MaryModel(cfg).eval()
    B, T_2Hz = 2, 200
    feats = _make_features(B, T_2Hz, list(STREAM_DIMS.keys()))
    subj = torch.tensor([0, 1])
    with torch.no_grad():
        out = model(feats, subj)
    assert out.shape == (B, 100, 20484), out.shape


def test_forward_subset_streams():
    cfg = _small_cfg()
    model = MaryModel(cfg).eval()
    B, T_2Hz = 2, 200
    feats = _make_features(B, T_2Hz, ["slowfast", "whisper"])
    subj = torch.tensor([0, 1])
    with torch.no_grad():
        out = model(feats, subj)
    assert out.shape == (B, 100, 20484), out.shape


def test_forward_single_stream():
    cfg = _small_cfg()
    model = MaryModel(cfg).eval()
    feats = _make_features(1, 200, ["qwen_ctx"])
    with torch.no_grad():
        out = model(feats, torch.tensor([0]))
    assert out.shape == (1, 100, 20484), out.shape


def test_shorter_window():
    # T_2Hz=100 -> T_TR=50 (min(max_fmri_len, T_2Hz//2)).
    cfg = _small_cfg()
    model = MaryModel(cfg).eval()
    feats = _make_features(2, 100, list(STREAM_DIMS.keys()))
    with torch.no_grad():
        out = model(feats, torch.tensor([0, 1]))
    assert out.shape == (2, 50, 20484), out.shape


def test_demographic_slot_optional():
    cfg = MaryConfig(n_subjects=4, use_demographic_slot=False)
    model = MaryModel(cfg).eval()
    feats = _make_features(1, 200, ["beats"])
    with torch.no_grad():
        out = model(feats, torch.tensor([0]))
    assert out.shape == (1, 100, 20484), out.shape


def test_param_count():
    # Faithful 9-subject config — report the real trainable param count.
    cfg = MaryConfig(n_subjects=9)
    model = MaryModel(cfg)
    n = model.n_trainable_params()
    print(f"n_trainable_params (9 subjects) = {n:,} ({n/1e6:.2f}M)")
    assert n > 0


if __name__ == "__main__":
    test_forward_all_six_streams()
    test_forward_subset_streams()
    test_forward_single_stream()
    test_shorter_window()
    test_demographic_slot_optional()
    test_param_count()
    print("test_shapes: ALL PASS")
