# Preregistration — do these neural indicators predict ad/marketing success?

The falsifiable claim, the exact test, and the kill criteria. Written before the real data is
collected so the result cannot be rationalized after the fact.

## The claim
**H1.** The engine's reward/approach indicator (constructs: reward-value, emotional resonance,
attention capture, memory encoding — anchored in NAcc/vmPFC reward anticipation) predicts an
ad's realized aggregate outcome **above chance** AND **beyond self-report**.
**H0.** It does not (permutation p ≥ .05, or bootstrap CI includes 0, or no incremental validity
over stated preference).

## Why this is provable, not hopeful
The core indicator is already peer-reviewed-proven at aggregate scale:
- **Genevsky & Knutson (2015)** — NAcc affect forecasts aggregate microlending (Kiva) funding.
- **Genevsky, Yoon & Knutson (2017)** — NAcc forecasts aggregate crowdfunding (Kickstarter)
  success *better than the funders' own behavior*.
- **Knutson & Genevsky (2018)** — review: affective neural signal generalizes to aggregate choice.
This study reproduces that logic on **ad creative** with our engine's indicator.

## Design
- **Units:** ads (or audience segments). Target n = 100 (Prolific, ~$500) — see `power_analysis.py`.
- **Indicator (predictor):** engine approach/reward score per ad, computed blind to outcome.
- **Outcome:** a realized aggregate metric — pre/post attitude shift, recall, share-intent, or
  (retrospective arm) funding %/sales lift/vote swing.
- **Baseline to beat:** matched self-report liking/intent per ad. Neural must add *beyond* this.

## Analysis (frozen — see `validate_indicator.py`)
1. Spearman ρ(indicator, outcome), 5,000-permutation p-value, 5,000-bootstrap 95% CI.
2. **Incremental validity:** partial ρ controlling for self-report (the Knutson/Genevsky bar).
3. Pre-registered pass rule: `perm_p < .05` AND `CI lower > 0` AND `incremental ρ > 0`.

## Power (from `power_analysis.py`)
| True ρ | n for 80% power |
|---|---|
| 0.50 (aggregate anchor) | **30** |
| 0.40 | 47 |
| 0.25 (individual) | 124 |
At n=100 the study detects ρ ≥ 0.28. The aggregate effect clears this comfortably.

## Kill criteria (we will report failure)
- If `perm_p ≥ .05` at n=100 → the indicator does not predict outcome; we say so.
- If neural adds nothing beyond self-report (incremental ρ ≤ 0) → no product moat over a survey.
- No optional stopping, no outcome-metric swapping after seeing data.

## Status
Harness + power validated on simulated data at the published effect size (recovered ρ=0.59,
perm_p<.001, beats self-report — see `figures/validation_demo.png`). **Real data not yet
collected.** The demo proves the machinery; it is not evidence about real ads.
