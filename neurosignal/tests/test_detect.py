"""Detection logic: whole-brain -> Buy/Sell, partial coverage honesty, validation."""
from __future__ import annotations

import numpy as np
import pytest

from neurosignal.constructs import YEO7_NETWORKS
from neurosignal.detect import detect_from_networks
from neurosignal.inputs.parcels import detect_from_parcels


def test_high_reward_low_conflict_is_buy():
    a = {n: 0.5 for n in YEO7_NETWORKS}
    a["Limbic"] = 1.0      # reward/value + emotion
    a["Default"] = 1.0     # reward/value + memory
    a["VentralAttention"] = 0.0   # conflict low
    a["Frontoparietal"] = 0.0     # cognitive load low
    r = detect_from_networks(a)
    assert r.recommendation == "Buy"
    assert r.coverage == 1.0
    assert all(c.covered for c in r.constructs)


def test_high_conflict_low_reward_is_sell():
    a = {n: 0.5 for n in YEO7_NETWORKS}
    a["Limbic"] = 0.0
    a["Default"] = 0.0
    a["VentralAttention"] = 1.0   # conflict high
    a["Frontoparietal"] = 1.0     # load high
    r = detect_from_networks(a)
    assert r.recommendation == "Sell"


def test_partial_coverage_is_flagged_not_fabricated():
    r = detect_from_networks({"Visual": 1.0, "DorsalAttention": 0.3})
    covered = {c.key for c in r.constructs if c.covered}
    assert "visual_sensory" in covered and "attention" in covered
    assert "reward_value" not in covered    # absent regions -> not covered
    assert r.coverage < 1.0
    assert any("NOT covered" in n for n in r.notes)


def test_input_validation():
    with pytest.raises(ValueError):
        detect_from_networks({})
    with pytest.raises(ValueError):
        detect_from_networks({"Visual": float("nan")})
    with pytest.raises(ValueError):
        detect_from_networks({"NotARealNetwork": 1.0})


def test_parcels_entrypoint_matches():
    n = 1000
    ids = np.array([i % 7 for i in range(n)])
    pa = np.full(n, 0.5)
    pa[np.isin(ids, [4, 6])] = 1.0   # Limbic + Default high
    pa[ids == 3] = 0.0               # SalVentAttn (conflict) low
    pa[ids == 5] = 0.0               # Cont (load) low
    r = detect_from_parcels(pa, ids)
    assert r.recommendation == "Buy"
    assert r.coverage == 1.0
