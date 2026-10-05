# Pipeline architecture — upload → encode → stats → live digital brain

The four layers you asked for, each a separate module so they never entangle.

```
┌─ 1. INGEST ───────────────────────────────────────────────────────────────┐
│ user uploads media (video / audio / text)                                  │
│ → segmented into per-second windows (features per second, or Mary's        │
│   per-second predicted activations)                                        │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
┌─ 2. ENCODE (neural encoding engine) ──────────────────────────────────────┐
│ neurosignal.timeline.analyze_timeline(...)                                 │
│   each second → encoder (reference | learned-Mary | enrolled brain_file)   │
│              → Yeo-7 network activation                                     │
│              → metrics: valence, arousal, engagement, PERSUASION pressure, │
│                buy/sell  +  the 7 constructs                               │
│   global scaling across the whole clip → frames comparable over time       │
│   → Timeline(frames[t])                                                     │
└────────────────────────────────────────────────────────────────────────────┘
                                   │   raw second-by-second frames
                                   ▼
┌─ 3. STATS (separate layer) ───────────────────────────────────────────────┐
│ neurosignal.stats.compile_stats(timeline)                                  │
│   per-metric time-series stats (mean/peak/peak_t/auc/volatility)           │
│   "moments that matter" (peak persuasion / arousal / buy timestamps)       │
│   network dynamics (which areas move most) · intro/build/payoff roll-ups   │
│   → Stats (JSON-serializable; the compiled forms the UI consumes)          │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
┌─ 4. VISUALIZE ────────────────────────────────────────────────────────────┐
│ neurosignal.viz.render_report({label: {timeline, stats}})                  │
│   • LIVE DIGITAL BRAIN — 7 Yeo-7 cortical regions light up second-by-      │
│     second; play / scrub; persuasion sparkline with a moving playhead      │
│   • synced metric bars + buy/sell verdict + "moments that matter"          │
│   • per-person switcher (average brain vs enrolled subjects)               │
│   → self-contained interactive HTML (examples/live_brain_report.html)      │
└────────────────────────────────────────────────────────────────────────────┘
```

## Minimal usage

```python
from neurosignal import analyze_timeline, compile_stats, save_report, BrainFile

# per_second_networks: Mary's predicted Yeo-7 activation for each second of the upload
tl = analyze_timeline(per_second_networks=clip, fps=1.0)        # ENCODE
st = compile_stats(tl)                                          # STATS
save_report("report.html", {"Average brain": {"timeline": tl.to_dict(),
                                               "stats": st.to_dict()}})   # VISUALIZE

# personalized live brain (a real per-person scan): pass an enrolled brain_file
tl_person = analyze_timeline(per_second_networks=clip, brain_file=BrainFile.load("S01"))
```

## Wiring the real scan

Today the demo feeds illustrative per-second activations (the `--mock` discipline). For a
real run, the only swap is the source of `per_second_networks`: have Mary emit its 20,484-
vertex prediction per second → aggregate to Yeo-7 (`atlases.parcels_to_networks`) → feed in.
Everything downstream (stats, the live brain, personalization, the honesty layer) is unchanged.
The `examples/timeline_demo.py` script is the end-to-end reference.
