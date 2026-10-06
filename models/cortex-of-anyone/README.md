# cortex-of-anyone (Phase-1 MVE)

Few-shot **enrollment of a personal digital brain** against Mary's additive
`group + per_subject` head — the keystone step of the
[Cortex of Anyone roadmap](../NEXT_STEPS_cortex_of_anyone_2026-06.md), plus the
reusable **`neuro_ai_core`** honesty layer extracted for the whole program.

> **Honesty boundary (read first).** This repo validates the **machinery and the
> statistics** on a *faithful simulator* of Mary's mechanism — exactly the ISM
> `--mock` discipline. No scientific claim about real brains is read off simulated
> data. To get a real result, swap `SimulatedCortex` for a `MaryAdapter`
> (see `enrollment/additive_head.py`) and re-run unchanged.

## What's here

```
neuro_ai_core/        the shared, default-on honesty layer (retires program gaps G5/G6)
  finding.py            Finding contract — every number carries its evidence or prints UNVALIDATED
  conformal.py          split-conformal intervals + honest abstention (from neurobridge)
  specificity.py        specificity gate — withhold a construct score when the signal is confounded (from Mary)
  provenance.py         encoder/data/seed manifest with a short fingerprint
enrollment/
  additive_head.py      AdditiveEncoder contract (= Mary's out = group + per_subject) + faithful simulator
  enroll.py             enroll_head(): fit a rank-16 personal head from K samples; BrainFile (portable copy)
experiments/
  run_mve.py            preregistered H1/H3/H5, dual-encoder
  run_negative_control.py  individuation OFF -> gates must fail (not rigged)
tests/                  dependency-light (numpy only); run the .py files directly
results/                Findings JSON + verdicts
```

## Run

```bash
python3 tests/test_core.py
python3 tests/test_enrollment.py
python3 experiments/run_mve.py
python3 experiments/run_negative_control.py
```

## Preregistered hypotheses → kill criteria (from the roadmap)

| # | Hypothesis | Pass requires | Simulated-machinery result |
|---|---|---|---|
| **H1** | enrolled personal head **beats the average brain** on the new person's held-out stimuli | paired Δr 95% CI excludes 0 | **PASS** — r 0.69 vs 0.54, Δr CI [0.09, 0.22], dz≈1.3 (both encoders) |
| **H3** | **identity specificity** — own head predicts the person better than any other person's head (anti-leakage) | rank-1 ID > chance, own−other CI excludes 0 | **PASS** — 100% identification, own−other Δr≈0.29 |
| **H5** | **conformal coverage** holds for a brand-new enrolled person | empirical coverage ≥ nominal | **PASS** — 0.90 coverage |
| **neg. control** | with individuation OFF, H1 & H3 **must fail** | both UNVALIDATED | **PASS** — H1 fails (enrolled hurts), H3 = 0% (chance) |

## >>> Wiring real Mary <<<

`enrollment/additive_head.py` documents the swap. Concretely:
1. `MaryAdapter.group_predict(features)` → Mary's frozen group head on cached backbone features.
2. `enroll_head` is unchanged — it fits the rank-16 per-subject delta in vertex×feature space
   from K minutes of a **held-out** subject's measured BOLD (Tier A, fMRI), or from the
   EEG→head bridge output (Tier B, EEG-only; train the bridge on **NATVIEW** concurrent EEG-fMRI).
3. Re-run `run_mve.py` against the real held-out subjects. H1/H3/H5 then read as a **scientific**
   result; this is the direct extension of *Cortex of One* (per-subject head beats the average brain).

## What this closes (program-level)

- **Gate 5 / G1 (individual predictive validity):** H1 is that test, with the H3 anti-overfit gate.
- **G2 (single-encoder):** dual-encoder verdict-stability check is built into `run_mve.py`.
- **G6 (no honesty layer):** `neuro_ai_core` is the default-on conformal + specificity + Finding + provenance layer, ready to drop into `neurosignal`.
