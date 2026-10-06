"""The real (region-weighted) buy/sell readout: parcels -> constructs -> Buy/Hold/Sell.

These test the literature-grounded logic used by RealEngine, without needing a
checkpoint or GPU — synthetic parcel activation drives the mapping.
"""
from __future__ import annotations

import numpy as np

from sapient_serving.engine import signals as sig


def _net_ids(n_parcels=1000):
    return np.array([i % 7 for i in range(n_parcels)], dtype=int)


def test_construct_ts_shapes_and_bounds():
    T, P = 100, 1000
    rng = np.random.default_rng(0)
    cts = sig.construct_ts_from_parcels(rng.random((T, P)), _net_ids(P))
    assert {c.key for c in sig.CONSTRUCTS} <= set(cts)
    for v in cts.values():
        assert v.shape == (T,)
        assert v.min() >= -1e-6 and v.max() <= 1.0 + 1e-6
    s = sig.build_signals(cts, np.arange(T, dtype=float))
    assert 0 <= s.purchase_intent <= 100
    assert s.recommendation in ("Buy", "Hold", "Sell")


def test_high_reward_low_conflict_is_buy():
    T, P = 50, 1000
    net = _net_ids(P)
    parcels = np.full((T, P), 0.5)
    parcels[:, np.isin(net, [4, 6])] = 0.95  # Limbic + Default (reward/value) high
    parcels[:, net == 3] = 0.05              # SalVentAttn (conflict) low
    parcels[:, net == 5] = 0.05              # Cont (load) low
    s = sig.build_signals(sig.construct_ts_from_parcels(parcels, net), np.arange(T, dtype=float))
    assert s.recommendation == "Buy"


def test_low_reward_high_conflict_is_sell():
    T, P = 50, 1000
    net = _net_ids(P)
    parcels = np.full((T, P), 0.5)
    parcels[:, np.isin(net, [4, 6])] = 0.05  # reward/value low
    parcels[:, net == 3] = 0.95              # conflict high
    parcels[:, net == 5] = 0.95              # load high
    s = sig.build_signals(sig.construct_ts_from_parcels(parcels, net), np.arange(T, dtype=float))
    assert s.recommendation == "Sell"
