# Wiring Mary → neurosignal (ready to merge — NOT deployed)

This connects Mary's brain-encoder output to the `neurosignal` interpretation layer
(github.com/The-Sapient-Company/ag_push): Mary predicts cortical activation and
reduces it to Yeo-7 networks; `neurosignal` turns those into the cited constructs +
metrics + buy/sell signal. The model side owns the encoder; neurosignal owns the
(evidence-backed) interpretation. **Nothing has been deployed or pushed** — review,
then deploy when you're ready.

## What flows
```
Mary verts ── _reduce_networks ──► Yeo-7 networks ──► neurosignal.analyze_model_networks
                                                          └─► artifact["neuro"] = { metrics, networks, constructs, buy/sell, coverage }
```

## Changes (both repos, local, un-pushed)
**ag_push (`neurosignal`):**
- `neurosignal/inputs/model_networks.py` *(new)* — `analyze_model_networks(networks, source, modalities)` + `normalize_networks()`. Normalizes Yeo-7 naming variants (Mary uses `"Dorsal Attention"`, `"Default Mode"`; neurosignal uses `"DorsalAttention"`, `"Default"`) → runs the metric layer. **Reusable by ANY model.**
- `neurosignal/encoders/learned.py` — `LearnedEncoder` is no longer a stub; it accepts `predict_networks=callable(features)->Yeo-7` (for the full `analyze()` path).

**sapient-models (`mary/modal/serve.py`):**
- `_build_artifact()` now adds `artifact["neuro"]` from the Yeo-7 `networks` it already computes. **Additive + guarded** (try/except): a missing dep or any failure prints a note and never breaks a run; existing fields are untouched.

## To deploy (your call)
1. **Add `neurosignal` to the Modal image** (it's numpy-only for the core path). In the image builder in `serve.py`, add to `pip_install`, e.g.:
   ```python
   .pip_install("neurosignal @ git+https://github.com/The-Sapient-Company/ag_push.git")
   ```
   (or vendor the `neurosignal/` package into the repo + `add_local_dir`).
2. `modal deploy mary/modal/serve.py`
3. Run any analysis and confirm the artifact has a `neuro` block:
   ```json
   "neuro": { "metrics": [...valence, arousal, engagement, manipulation, buy/sell...],
              "constructs": [...7...], "buy_sell": .., "coverage": 1.0, "confidence": .. }
   ```

## Verified
`analyze_model_networks` was smoke-tested with Mary's exact Yeo-7 naming → coverage 1.0, all metrics + constructs computed. Name mapping confirmed.

## Reusable for the new model (and any future model)
Any model that emits Yeo-7 networks: `analyze_model_networks({...}, source="X")`.
For a model that maps features→networks itself: `get_encoder` / `LearnedEncoder(predict_networks=your_fn)` and call `analyze(...)`.

## Frontend note
`artifact["neuro"]` is **additive** — the app ignores it until we add renderers. The
modality-aware results registry we built (`src/verdict-v3/resultBlocks.ts`) is where
the constructs/metrics/buy-sell blocks will plug in. No frontend change ships with this.
