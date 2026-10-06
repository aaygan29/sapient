# QUALIA_CONTRACTS.md — locked interfaces for the Understanding Layer

*Like Mary's CONTRACTS.md: the invariants every Qualia component must respect.
Qualia is the layer; Mary (`../mary`) is the engine.*

## §0 — Engine boundary & the upgrade law
- Qualia loads Mary ONLY from a versioned checkpoint via `core/registry.py`
  (`MaryEngine.from_channel("improving"|"instrument")`). It never re-implements Mary.
- The model config travels inside the checkpoint, so a newer/better Mary is picked
  up by pointing a channel at the new `.pt` — **no Qualia code change.** This is the
  engine→layer compounding: **Mary improves ⇒ Qualia improves.**
- **Channels:** `improving` = latest Mary (demo, `understand`, `state` ride it).
  `instrument` = frozen, version-pinned (the `align`/Brain-Alignment standard runs
  ONLY here so scores stay comparable; bump deliberately).

## §1 — Output space (inherited from Mary, locked)
- **20,484 fsaverage5 vertices** (LH 10,242 + RH 10,242). → Schaefer-1000 → **Yeo-7**
  via `Mary-Papers/shared/schaefer_yeo.py`. Stable across Mary upgrades — the
  interface does not break; only prediction *quality* improves.
- Version-sensitive on a Mary architecture change: the raw penultimate **embedding
  dim** (`MaryEngine.d_model`, 768/1024) — version it for API consumers. Per-person
  heads were fit against the old core's `penultimate()` and need a cheap re-fit.

## §2 — Active modalities (today)
- Mary's active streams: **beats (768), whisper (1280), qwen_ctx (4096)** — AUDIO/TEXT.
  Video streams (slowfast, qwen_vl, got_ocr) are INACTIVE in current checkpoints.
- ⇒ Qualia today reads audio/text stimuli. Activating Mary's video streams is a Mary
  upgrade that flows through automatically (§0). Demos must not claim a visual read
  Mary can't yet make.

## §3 — Honesty boundary (enforced in every output)
- Outputs are **representational / affective alignment** (how human-like a
  representation is), labeled as such. **Never** claim phenomenal feeling/consciousness.
- `state()` network→(affect/intent/attention) labels ship with a documented, validated
  mapping + confidence — never a bare assertion.

## §4 — Engine API (what core exposes)
```
MaryEngine.from_channel(channel="improving", device=...) -> engine
engine.assemble_features(feat_dir, streams=ACTIVE_STREAMS) -> {stream:(1,T_2Hz,D_m)}
engine.encode(features, subject_idx=0) -> (T_TR, 20484) np.float32      # understand: vertices
engine.embed(features) -> (d_model,) np.float32                         # understand: stable embedding
engine.provenance() -> {channel, version, d_model, vertices, space}     # attach to every response
```
Scoring (`state`/`align`) is built ON TOP using `Mary-Papers/shared/eval_harness.py`
(`pearson_r`, `noise_ceiling`, `yeo7_decomposition`, `bootstrap_ci`) + `schaefer_yeo`.

## §5 — Family conventions
Python 3.11, torch 2.4.1; Modal profile `robert-16572`, volume `sapient-data`→`/data`,
`mary-hf-cache`→`/cache`, secrets `hf-token`/`wandb`. Env `QUALIA_MARY_ROOT` points at
the mary package; `QUALIA_<CHANNEL>_CKPT` overrides a channel checkpoint.
