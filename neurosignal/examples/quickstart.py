"""Minimal example — detect constructs + buy/sell from whole-brain network activations.

    pip install -e .
    python examples/quickstart.py
"""
from neurosignal import detect_from_networks

# Example: a stimulus that strongly engages reward/value + emotion, with low conflict.
activation = {
    "Visual": 0.6,
    "Somatomotor": 0.4,
    "DorsalAttention": 0.7,
    "VentralAttention": 0.2,   # low conflict
    "Limbic": 0.95,            # high reward/value + emotion
    "Frontoparietal": 0.3,     # low deliberation/load
    "Default": 0.8,            # reward/value + memory
}

result = detect_from_networks(activation, source="example whole-brain activation")
print(f"BUY/SELL: {result.recommendation}  ({result.buy_sell_score}/100, coverage {int(result.coverage*100)}%)")
for c in result.constructs:
    print(f"  {c.label:30s} {c.score:5.1f}" + ("  (cortical proxy)" if c.cortical_proxy else ""))
