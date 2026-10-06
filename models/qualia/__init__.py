"""Qualia — the Understanding Layer built on the Mary brain encoder.

Mary (sibling package `mary/`) is the frozen engine: stimulus -> human cortical
representation (20,484 fsaverage5 vertices). Qualia is the layer that turns that
into usable understanding (perception read-outs, the Brain Alignment / Social
Brain Score, personalization, and the demo).

Design law: Qualia loads Mary from a *versioned checkpoint* (see core/registry.py).
Pointing a channel at a better Mary checkpoint upgrades Qualia with no code change
— that is the engine->layer compounding. `align()` pins to a frozen `instrument`
channel so the standard stays comparable; everything else rides `improving`.

Honesty boundary: every output is *representational/affective alignment* (how
human-like a representation is), never a claim of phenomenal feeling/consciousness.
"""
