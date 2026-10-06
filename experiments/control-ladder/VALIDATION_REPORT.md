# MVE Control-Ladder: Apparatus Validation Report

Aayush Gandhi, 2026-06-19. No em dashes.
Code: `mve_control_ladder.py`. Raw output: `VALIDATION_RESULTS.txt`.

## What this is (and is NOT)

This is a validation of the **analysis apparatus** for the neurobehavioral-engine MVE
(`../NEUROBEHAVIORAL_ENGINE_PROTOCOL_2026-06.md`, section 3), run against **synthetic ground truth**.
It is the step that must precede any real run: prove the control ladder recovers known truth before
trusting it on data.

**It says NOTHING about real brains.** No fMRI, no TRIBE v2, no Emo-FilM were used. The real MVE is
still blocked on (a) Emo-FilM access + fMRIPrep and (b) TRIBE v2 weights on Modal, neither available in
the current environment. Reporting any real neurobehavioral number now would repeat the exact
retracted-overclaim failure mode this program is built to avoid.

## Design

Control ladder (protocol section 3): M0 (content) / M0+ (capacity-matched non-brain basis) / M1
(+behavior history) / M2 (+correct-subject neural) / M2-perm (+wrong-subject neural).

Two pre-registered gates: **Gate 1** skill(M2) > skill(M0+); **Gate 2** skill(M2) > skill(M2-perm).

Crossed factors:
- WORLD: SIGNAL (behavior depends on an individual neural component) vs NULL (behavior is pure stimulus
  content; neural is generic stimulus drive).
- SPLIT: LOSO-stim (subjects seen = ENROLLED regime) vs LOSO-subj (held-out subject = UNSEEN regime,
  fed population-only neural to faithfully model zero-shot TRIBE, which cannot individualize a new
  person).

20 seeds, RidgeCV, pooled out-of-sample R^2, paired bootstrap CI over seeds.

## Result: the apparatus is correct (Gate 2 confusion matrix is perfect)

| WORLD / SPLIT | regime | Gate 1 (M2>M0+) | Gate 2 (M2>M2-perm) | expected Gate 2 |
|---|---|---|---|---|
| SIGNAL / LOSO-stim | ENROLLED | PASS +0.626 | **PASS +0.449** [+0.378,+0.526] | FIRE |
| SIGNAL / LOSO-subj | UNSEEN | no | no (+0.028 n.s.) | silent |
| NULL / LOSO-stim | ENROLLED | PASS +0.087 | no (-0.001) | silent |
| NULL / LOSO-subj | UNSEEN | PASS +0.034 | no (-0.001) | silent |

**Gate 2 fires iff individual neural signal exists AND the subject is enrolled.** Zero false positives
in the two NULL cells; correctly silent in the UNSEEN/SIGNAL cell (individual signal cannot transfer to
a person the encoder never saw). This is the specificity the protocol's load-bearing gate requires.

## Two findings the validation produced (both fed back into the protocol)

1. **Gate 1 is necessary but NOT sufficient for "neurobehavioral."** Gate 1 passes even in the NULL
   world (NULL/ENROLLED +0.087, NULL/UNSEEN +0.034) because the neural features carry the true
   stimulus->behavior basis and beat a *random* content lift on population grounds alone, with zero
   individuation. Therefore Gate 1 measures **population value** and Gate 2 measures **individual
   value**. They map onto the two product regimes: Gate 1 = the unseen-audience product; Gate 2 = the
   enrolled-user product. Reporting a Gate 1 pass as "we read the individual" would be an overclaim.
2. **The enrolled/unseen split is real and load-bearing.** With a faithful (population-only) neural
   provider for unseen subjects, individuation is undetectable in the UNSEEN regime by construction.
   Any future real-data claim of individual neural prediction for *new* people is therefore structurally
   suspect unless those people were enrolled (had real per-subject data). The protocol now reports the
   two regimes separately with different ceilings.

## Bugs this validation caught before real data (the point of the exercise)

- **Metric aggregation.** Per-fold R^2 averaged within a held-out stimulus is degenerate (content is
  constant across subjects at a fixed stimulus, giving R^2 down to -158). Fixed to pooled out-of-sample
  R^2 across all held-out predictions.
- **A cheating neural provider.** The first pass handed unseen subjects their individualized neural
  features, which TRIBE cannot produce zero-shot; this spuriously fired Gate 2 in the UNSEEN regime.
  Fixed by feeding population-only neural to held-out subjects.

## Reproduce

```bash
cd neurobehavioral-mve
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 mve_control_ladder.py
```
(Single-threaded BLAS is ~100x faster here: thousands of tiny RidgeCV fits oversubscribe cores otherwise.)

## What unblocks the REAL MVE (next, not done here)

1. Emo-FilM (OpenNeuro, access + fMRIPrep) OR a substitute paired neuro+behavior set already local.
2. TRIBE v2 weights deployed on Modal (the Sapient-1 inference pattern) + a second encoder for M2b.
3. The section 2D encoder-training-leakage audit (exclude any Emo-FilM stimulus in TRIBE's training set).
4. Swap the synthetic data loader in `mve_control_ladder.py` for the real feature loader; the model
   ladder, gates, splits, and statistics carry over unchanged. The apparatus is ready.
</content>
