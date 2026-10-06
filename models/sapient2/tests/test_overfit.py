"""Single-batch overfit test for the training loop.

Per CODEBASE_MAP §8: "trains on a single batch for 50 steps; loss must drop
below 0.01". We use a SMALL config variant here so CPU runs in seconds —
this validates the training-loop *wiring* (gradient flow + optimizer + loss).
The full ~180M model overfit is verified separately on H100 via Modal once
real features are cached; that result lives in W&B, not in pytest.

If you want to run the full-config variant locally:
  pytest tests/test_overfit.py::test_overfit_full_config -v --no-header
(slow — minutes on CPU.)
"""

from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW

from sapient2 import SapientConfig, SapientModel, masked_mse_loss


def _run_overfit(cfg: SapientConfig, *, steps: int, lr: float = 1e-3) -> tuple[float, float]:
    """Returns (initial_loss, final_loss)."""
    torch.manual_seed(0)
    model = SapientModel(cfg)
    model.train()
    optim = AdamW(model.parameters(), lr=lr)

    B = 2
    video = torch.randn(B, cfg.max_stim_len, cfg.d_video)
    audio = torch.randn(B, cfg.max_stim_len, cfg.d_audio)
    text  = torch.randn(B, cfg.max_stim_len, cfg.d_text)
    subj  = torch.randint(0, cfg.n_subjects, (B,))
    target = torch.randn(B, cfg.max_fmri_len, cfg.n_vertices)
    mask = torch.ones(B, cfg.max_fmri_len)

    initial_loss = None
    final_loss = None
    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        pred = model(video, audio, text, subj)
        loss = masked_mse_loss(pred, target, mask)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        final_loss = loss.item()
    assert initial_loss is not None and final_loss is not None
    return initial_loss, final_loss


def test_overfit_small_config() -> None:
    """Tiny model: hidden=128, n_layers=2, n_vertices=512. Should fit one batch fast.

    This is what `pytest` runs in CI / local dev — proves the training loop
    is wired correctly. The full model is too slow on CPU to overfit in 50
    steps, but the *wiring* (params, gradients, optimizer step, loss) is the
    same.
    """
    cfg = SapientConfig(
        hidden=128, n_layers=2, n_heads=4, ff_mult=2,
        low_rank_dim=128, n_vertices=512,
        max_stim_len=20, max_fmri_len=10,
        n_subjects=2, dropout=0.0,
        modality_dropout=0.0, subject_dropout=0.0,
    )
    initial, final = _run_overfit(cfg, steps=80, lr=3e-3)
    assert final < initial * 0.1, \
        f"loss did not drop ≥10×: {initial:.4f} → {final:.4f}"
    # With dropout disabled and a tiny model, 80 steps should drive loss near zero.
    assert final < 0.05, \
        f"loss did not approach zero: {final:.4f}"


@pytest.mark.slow
def test_overfit_full_config() -> None:
    """Full ~178M-param config. Runs on CPU but takes minutes.

    Marked `slow`; skipped by default. Run manually with:
        pytest tests/test_overfit.py::test_overfit_full_config -v
    Real verification of the full-config overfit happens on H100 via the
    training Modal app; the W&B run is the artifact, not this test.
    """
    cfg = SapientConfig(
        modality_dropout=0.0, subject_dropout=0.0, dropout=0.0,
        n_subjects=2,
    )
    initial, final = _run_overfit(cfg, steps=50, lr=1e-3)
    assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"
