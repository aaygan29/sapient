"""Evidence-base integrity: every construct is cited, mapped, and unique."""
from __future__ import annotations

from neurosignal.constructs import CONSTRUCTS, YEO7_NETWORKS


def test_every_construct_is_cited_and_mapped():
    keys = set()
    for c in CONSTRUCTS:
        assert c.citations, f"{c.key} has no citations"
        assert c.regions, f"{c.key} has no regions"
        assert set(c.yeo7_networks) <= set(YEO7_NETWORKS), f"{c.key} maps to a non-Yeo7 network"
        assert c.key not in keys, f"duplicate construct key {c.key}"
        keys.add(c.key)


def test_has_both_approach_and_avoidance_drivers():
    assert any(c.polarity > 0 for c in CONSTRUCTS), "need at least one approach (buy) driver"
    assert any(c.polarity < 0 for c in CONSTRUCTS), "need at least one avoidance (sell) driver"
