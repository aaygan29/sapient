# neurosignal

Evidence-based detection of **consumer-relevant brain-activation constructs** —
reward/value, emotional resonance, attention, visual/sensory salience, memory
encoding, decision conflict, and cognitive load — plus an **approach–avoidance
buy/sell signal**, from measured or model-predicted fMRI.

Every output is backed by cited literature (see `EVIDENCE.md`), and the detector is
**honest about coverage**: constructs whose brain regions aren't present in your
input are reported as *not covered* — never silently invented.

## Install
```bash
pip install -e .                 # core (numpy only)
pip install -e ".[atlas]"        # + Schaefer-1000 / Yeo-7 atlas (nilearn) for measured fMRI
pip install -e ".[encoder]"      # + Digital Brain image encoder (torch/CLIP) — visual cortex
pip install -e ".[dev]" && pytest
```

## Analyze media (text / audio / video)
```python
from neurosignal import analyze
a = analyze(text="You're absolutely right — what a brilliant, amazing question!")
for m in a.metrics:
    print(m.label, m.score, m.interpretation)   # valence, arousal, engagement, manipulation, sycophancy, buy/sell
```
`analyze()` runs **media → brain encoder → Yeo-7 network activation → the metric layer**. The
**reference encoder** (transparent, fully auditable, runs today) is the default; pass
`encoder="learned"` to use a trained fMRI encoder (TRIBE / Sapient / Digital Brain). Audio/video
accept precomputed feature dicts now, or file paths with the `[audio]` / `[video]` extras.

```bash
neurosignal analyze --text "SHOCKING! Unbelievable, urgent deal — act now!"   # CLI
```

### Metrics (every score exposes its neuro basis)
| Metric | Range | Neuro basis |
|---|---|---|
| Affective Valence | −100..100 | limbic reward/emotion vs anterior-insula/ACC withdrawal |
| Arousal / Intensity | 0–100 | limbic (amygdala) + salience network |
| Engagement | 0–100 | dorsal attention + salience + reward |
| **Manipulation Index** | 0–100 | affective/reward drive vs analytic (frontoparietal) control — System-1 capture |
| **Sycophancy Index** (text) | 0–100 | social-reward + agreement/praise with low epistemic conflict — audits LLM flattery |
| Buy / Sell Signal | 0–100 | approach-avoidance neuroforecasting (reward − conflict) |

…plus the **7 construct activations** (reward/value, emotion, attention, visual, memory, conflict,
cognitive load) and the **Yeo-7 network profile**. Definitions + formulas in `METRICS.md`.

## Score activation you already have (measured / predicted fMRI)
```python
from neurosignal import detect_from_networks

# Yeo-7 network activations (any scale; min-max normalized internally)
r = detect_from_networks({
    "Limbic": 0.95, "Default": 0.8, "DorsalAttention": 0.7, "Visual": 0.6,
    "VentralAttention": 0.2, "Frontoparietal": 0.3, "Somatomotor": 0.4,
})
print(r.recommendation, r.buy_sell_score, r.coverage)   # Buy 88.0 1.0
for c in r.constructs:
    print(c.label, c.score, c.covered)
```

Whole-brain fMRI (Schaefer-1000 betas/z-stats):
```python
from neurosignal.inputs.parcels import detect_from_parcels
from neurosignal.atlases import load_schaefer_network_ids        # needs [atlas]
ids = load_schaefer_network_ids(1000)
r = detect_from_parcels(betas, ids)        # betas: (1000,) or (T, 1000)
```

## Use it (CLI)
```bash
neurosignal detect --networks "Visual=1.0,Limbic=0.9,VentralAttention=0.2,Frontoparietal=0.3"
neurosignal detect --parcels betas.npy --schaefer
neurosignal detect --parcels betas.npy --network-ids yeo7_ids.npy --json
```

## What it detects (the 7 constructs)
| Construct | Brain basis (Yeo-7 proxy) | Buy/Sell |
|---|---|---|
| Reward & Value | vmPFC/OFC + ventral striatum (Limbic, Default) | + buy |
| Emotional Resonance | amygdala/limbic (Limbic) | + |
| Attention Capture | dorsal attention (DorsalAttention) | + |
| Visual & Sensory Salience | V1–V4 / streams (Visual) | + |
| Memory Encoding | hippocampus/DMN (Default) | + |
| Decision Conflict & Risk | ACC + anterior insula (VentralAttention) | − sell |
| Cognitive Load | DLPFC/dmPFC (Frontoparietal) | − |

`buy_sell = 100·clip(0.5 + (approach−0.5) − ½(avoid−0.5), 0, 1)` → **Buy ≥ 60 · Hold · Sell ≤ 40**.
Full citations + formula in **`EVIDENCE.md`**.

## Inputs & honest scope
- **Whole-brain fMRI** (`detect_from_parcels` / `--schaefer`) → covers all 7 constructs. *Best.*
- **Yeo-7 network activations** (`detect_from_networks`) → covers whatever networks you provide.
- **Image → Digital Brain encoder** (`detect_from_image`, `[encoder]`) → **visual cortex only**;
  populates `visual_sensory`, reports the rest as not-covered. Don't read reward/conflict from it.
- Subcortical regions (NAcc, amygdala, hippocampus) are approximated by cortical proxies on a
  surface atlas; for direct subcortical detection use a volumetric subcortical atlas.

## Personal digital brains (enrollment) + honesty layer  *(new in 0.3.0)*
Two capabilities from the **Cortex of Anyone** program (`enrollment.py`, `validation.py`):

```python
import neurosignal as ns
from neurosignal import BrainFile, enroll_network_head
from neurosignal.encoders import get_encoder

# 1) ENROLL a person -> a portable digital-brain copy (network-level few-shot).
#    calibration = [(features, measured_yeo7), ...] from the new subject's own scan.
bf = enroll_network_head(get_encoder("reference"), calibration, subject="S01")
#    -> BrainFile (per-subject head + provenance + fingerprint); bf.save("S01")

# 2) PERSONALIZE the read-out: same stimulus, THAT person's brain (not the average).
a = ns.analyze(text="Act now — this exclusive deal disappears in minutes!", brain_file=bf)
print(a.personalized, a.provenance)   # True, {subject, fingerprint, ...}
```

The same ad scans differently per person (see `examples/personalized_scan.py`): manipulation
index 6 (average) → 87 (a reward-seeker) → 0 (a deliberator). When no `brain_file` is given,
behavior is **identical to before** — personalization is strictly opt-in.

For the **vertex-level** Mary model, `enrollment.AdditiveEncoder` / `enroll_head` express Mary's
exact `out = group_head + per_subject_head[idx]` contract; subclass `MaryAdapter` to wire the
frozen checkpoint and enroll a brand-new subject without retraining (Tier A fMRI, Tier B EEG).

The **honesty layer** (`validation.py`) ships conformal prediction intervals
(`split_conformal_regression`), the `specificity_gate` (withhold a construct when the signal is a
confound), the `Finding` contract (`gate()` refuses unvalidated numbers), and `Provenance` stamps.
Validated end-to-end on a simulator in `~/Desktop/Research/Neuro-AI/cortex-of-anyone/`.

## Robustness & security
- Pure-NumPy core; deterministic; inputs validated (finite values, known network labels, matching shapes).
- Coverage/confidence reflect what the input can actually measure — no fabricated scores.
- The image encoder loads a **pickle** (arbitrary-code-execution risk): it refuses to load unless
  `trusted=True` / `NEUROSIGNAL_TRUST_PICKLE=1`. Only load model files you trust.
- No secrets, no implicit network calls (model/atlas downloads are explicit, opt-in extras).

> Research/decision-support tool — not medical or diagnostic. Outputs are activation estimates and
> a literature-grounded heuristic, not clinical findings.
