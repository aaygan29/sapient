"""Shape + param-count sanity. Per spec §3.4.

Forward pass with (B, 200, *) inputs must produce (B, 100, 20484) output.
Trainable param count expected ~180M at SapientConfig defaults.
"""

from __future__ import annotations

import pytest
import torch

from sapient2 import SapientConfig, SapientModel


@pytest.fixture(scope="module")
def model() -> SapientModel:
    cfg = SapientConfig()
    return SapientModel(cfg)


def test_forward_shape(model: SapientModel) -> None:
    B = 4
    video = torch.randn(B, 200, 1280)
    audio = torch.randn(B, 200, 1024)
    text  = torch.randn(B, 200, 3072)
    subj  = torch.randint(0, model.cfg.n_subjects, (B,))
    model.eval()
    with torch.no_grad():
        out = model(video, audio, text, subj)
    assert out.shape == (B, 100, 20484), f"got {tuple(out.shape)}"


def test_param_count_ballpark(model: SapientModel) -> None:
    n = model.n_trainable_params()
    # Spec §3.1 table sums to ~180M. Tolerate ±10% for layer-norm gammas etc.
    assert 160_000_000 < n < 200_000_000, f"got {n:,} (expected ~180M)"


def test_train_eval_modes(model: SapientModel) -> None:
    """Confirm modality + subject dropout fire in train but not in eval."""
    B = 2
    video = torch.zeros(B, 200, 1280)
    audio = torch.zeros(B, 200, 1024)
    text  = torch.zeros(B, 200, 3072)
    subj  = torch.zeros(B, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out_eval = model(video, audio, text, subj)
    model.train()
    out_train = model(video, audio, text, subj)
    # Outputs should differ between modes when input is identical, because
    # dropout (regular + modality + subject) is active only in training.
    assert not torch.allclose(out_eval, out_train), \
        "train and eval outputs identical — dropout layers are not firing"
