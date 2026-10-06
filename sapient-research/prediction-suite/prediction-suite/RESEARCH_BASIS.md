# Research basis — the neuroforecasting evidence the engine is built on

Every design choice in `engine/behavioral_bridge.py` traces to a specific, citable
finding. This is the honest provenance of the Prediction Suite: it is built on the
public neuroforecasting literature, not on any private or classified work. That
literature is strong enough to stand on its own.

## The core claim: small-panel brain activity forecasts population behavior

| Finding | Source | What we take from it |
|---|---|---|
| NAcc, MPFC, insula activity predicts purchases **beyond self-report** | Knutson, Rick, Wimmer, Prelec, Loewenstein, *Neuron* 2007 | Signed region priors; value/NAcc as the strongest single "buy" signal |
| MPFC of **50 smokers forecast a 400,000-person** quit-line campaign; self-report did not | Falk, Berkman, Lieberman, *Psychological Science* 2012 | Aggregate forecasting is possible from tiny neural samples; self-report is not sufficient |
| Neural + self-report **combined R² up to 0.65**, and the effect is **content-dependent** | Falk et al., *Soc Cogn Affect Neurosci* 2016 | The self-report ensemble; the specificity/content gate |
| fMRI forecast **real chocolate sales across 63,617 shoppers**; signed region directions (NAcc/mOFC +, dlPFC/insula −) | Kühn, Strelow, Gallinat, *NeuroImage* 2016 | The signed region prior table in `SIGNED_REGION_PRIORS` |
| Brain forecasts **stay valid on non-representative samples** where behavioral forecasts degrade | Genevsky, Knutson et al., *PNAS Nexus* 2025 | The "sample-robust" read-out; why affect-weighting is preferred for aggregate work |
| Affect-Integration-Motivation: **affective** components generalize across individuals; integrative ones are idiosyncratic | Knutson & Genevsky, *Curr Dir Psychol Sci* 2018 | The AIM construct weighting in `AIM_CONSTRUCT_PRIORS` |
| Inter-subject correlation / temporal reliability of neural response predicts audience preference | Dmochowski et al., *Nature Communications* 2014 | Attention/engagement as a supporting (not driving) term |
| Trailer brain response predicts box-office / commercial success | Boksem & Smidts, *J. Marketing Research* 2015 | Memory-encoding term → recall/commercial outcome |
| EEG beta/alpha engagement index forecast banner efficiency across **291,301 users** | Kislov et al., *Brain Sciences* 2022 | Cross-modal support; roadmap for an EEG bridge |
| Review of neuroforecasting theory, metrics, and limits | Yao et al., *J. Consumer Behaviour* 2024 | Honest limitations: cost, sample size, ecological validity, reverse inference |

## How the findings map to code

- `AIM_CONSTRUCT_PRIORS` — affect (value, reward, arousal, emotion) weighted above
  integrative/attention terms, per Knutson & Genevsky 2018 and Genevsky 2025.
- `SIGNED_REGION_PRIORS` — the Kühn 2016 direction table: NAcc +1.0, mOFC +0.8,
  amygdala +0.45, dmPFC +0.5, dlPFC −0.6, insula −0.7.
- `DEFAULT_NEURAL_WEIGHT = 0.6` — the neural/self-report ensemble from Falk 2016.
- `representativeness_robust` — surfaces the Genevsky 2025 result to the user: a
  forecast carried by affect is flagged sample-robust; one carried by an integrative
  signal is flagged sample-sensitive with a wider band.
- specificity gate hand-off — Falk 2016 content-dependence: the neural term is only
  allowed to carry a claim when `mary_readouts.specificity_gate` confirms the signal
  is construct-driven, not sensory.

## What this is NOT

- It is **not** validated on your data yet. Priors set the shape; only `calibrate()`
  on real ad outcomes sets the scale, and only `Neuroforecaster` (leave-one-ad-out,
  beats self-report, passes the gate) earns the predictive claim.
- It makes **no claim to change minds.** It forecasts and explains response. The
  honesty boundary is enforced in code, not just in copy.
- It is **not** derived from any classified, intelligence-agency, or private
  government program. If a claim like that ever appears in marketing, it is false and
  should be removed — the real lineage above is stronger and defensible.

## Reproduce

`python engine/demo.py` runs the whole pipeline on synthetic data and prints the
numbers used in the frontend. Swap the synthetic block for Mary vertex exports plus
measured outcomes to produce real figures.
