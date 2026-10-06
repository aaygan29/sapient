"""Composite-loss tests (ORCLE-Nano §5, CONTRACTS.md §6).

Verifies: finite scalar + 3-term dict; gradients flow to all trainable params; backward
works end-to-end through the model. Runnable with pytest OR directly.
"""

from __future__ import annotations

import torch

from mary.adapter import STREAM_DIMS
from mary.losses import composite_loss
from mary.model import MaryConfig, MaryModel


def test_composite_loss_finite_and_terms():
    torch.manual_seed(0)
    B, T, V = 2, 100, 20484
    pred = torch.randn(B, T, V, requires_grad=True)
    target = torch.randn(B, T, V)
    loss, terms = composite_loss(pred, target)
    assert torch.isfinite(loss), loss
    assert set(terms.keys()) == {"mse", "negcorr", "infonce"}, terms.keys()
    for k, v in terms.items():
        assert isinstance(v, float)
        assert v == v, f"{k} is NaN"


def test_composite_loss_backward_to_pred():
    B, T, V = 2, 50, 20484
    pred = torch.randn(B, T, V, requires_grad=True)
    target = torch.randn(B, T, V)
    loss, _ = composite_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_gradients_flow_to_all_params():
    torch.manual_seed(0)
    cfg = MaryConfig(n_subjects=4)
    model = MaryModel(cfg).train()
    feats = {s: torch.randn(2, 200, STREAM_DIMS[s]) for s in STREAM_DIMS}
    subj = torch.tensor([0, 1])
    target = torch.randn(2, 100, cfg.n_vertices)
    pred = model(feats, subj)
    loss, terms = composite_loss(pred, target)
    loss.backward()

    # Every trainable param that participates should get a finite grad.
    # Per-subject head rows for unused subjects (2,3) legitimately get no grad,
    # so we check that the vast majority of params have grads and none are NaN.
    missing = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
        else:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    # Only the per-subject head's unused-subject slices may be missing.
    for name in missing:
        assert "subject_weight" in name or "subject_bias" in name, (
            f"unexpected param with no gradient: {name}"
        )
    print(f"params with grad checked; only-subject-head-unused missing = {len(missing)}")


def test_mask_partial():
    B, T, V = 2, 100, 20484
    pred = torch.randn(B, T, V, requires_grad=True)
    target = torch.randn(B, T, V)
    mask = torch.ones(B, T)
    mask[:, 80:] = 0  # pad last 20 TRs
    loss, terms = composite_loss(pred, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


if __name__ == "__main__":
    test_composite_loss_finite_and_terms()
    test_composite_loss_backward_to_pred()
    test_gradients_flow_to_all_params()
    test_mask_partial()
    print("test_losses: ALL PASS")
