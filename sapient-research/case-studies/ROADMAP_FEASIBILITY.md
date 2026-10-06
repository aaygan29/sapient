# Roadmap feasibility guideline

Scores every task in the pasted 5-phase roadmap against **what already exists in this repo**.
Columns: **Ease** (build cost), **Relevance** (moves the product), **Importance** (blocks
other work / investor-critical), **Verdict**. 1 = low, 5 = high.

The roadmap assumes a greenfield Python project. This repo already has the organs, more
grounded: `neurosignal/` (local construct engine), Mary/TRIBE encoder on Modal,
`ag_push/prediction-suite/` (behavioral bridge + readouts), and a **chain audit**
(`neurosignal/CHAIN_AUDIT_2026-07-16.md`) that already diagnoses the core gap.

## Phase 1 — Data acquisition

| Task | Ease | Rel | Imp | Verdict |
|---|---|---|---|---|
| **1C Retrospective forensic (historical/Super Bowl ads)** | 5 | 4 | 4 | ✅ **DONE (this folder).** $0, uses real engine, honest. Best first move. |
| 1A NSD validation of encoder | 2 | 5 | 5 | ⏸ **High value, not cheap.** Needs NSD download (~GB) + the Mary/TRIBE encoder off Modal. This is the real "does the encoder work" test. Fund after case studies. |
| 1A StudyForrest / NARPS / DEAP cross-checks | 2 | 3 | 3 | ⏸ Defer. Nice-to-have convergent validation; heavy data, low marginal investor value now. |
| 1B Synthetic ad grid + Prolific study | 3 | 5 | 5 | 🎯 **Case Study B.** ~$500, the cheapest *real human* validation. Do right after this folder ships. Not faked here. |

## Phase 2 — Evaluation framework

| Task | Ease | Rel | Imp | Verdict |
|---|---|---|---|---|
| Pillar 1 Construct validity (discriminant/convergent/predictive) | 3 | 4 | 4 | ◑ **Partly exists.** `neurosignal/validation.py` + specificity gate already enforce discriminant/confound control. Wrap them into a single `construct_validity` report — moderate. |
| Pillar 2 Segment sensitivity (digital twins) | 2 | 3 | 2 | ⏸ Defer. `DigitalTwinPanel` in roadmap is a `base * weight` stub — not credible without per-segment data. Don't ship a fake segment score. |
| Pillar 3 Temporal robustness / decay | 4 | 2 | 2 | ○ Low priority. `neurosignal/timeline.py` already gives second-by-second; a decay curve is a small add if a customer asks. |

## Phase 3 — Product integration

| Task | Ease | Rel | Imp | Verdict |
|---|---|---|---|---|
| `/evaluate` product API (not raw activations) | 3 | 5 | 4 | ◑ **Largely exists** as `ag_push/prediction-suite/api/prediction.ts`. Needs the goal-weighting + report shape from the roadmap layered on. Real work, but don't rebuild in Python — the product is TS. |
| DB schema (campaigns/evaluations/segment/arc) | 4 | 3 | 3 | ○ Easy but premature. Add when the API report shape is frozen; otherwise migration churn. |
| Multi-modal encoder / dashboard | 1 | 4 | 3 | ⏸ Large. Existing `src/verdict-v3` + `PredictionSuite.tsx` are the surfaces. Coordinate with teammate (PR #96) before touching. |

## Phase 4 — Case studies

| Task | Ease | Rel | Imp | Verdict |
|---|---|---|---|---|
| A Retrospective political forecast | 5 | 4 | 4 | ✅ **DONE here** (merged with C). |
| B Prolific validation | 3 | 5 | 5 | 🎯 Next. Funded, cheap, real humans. |
| C Competitive benchmark (Super Bowl) | 5 | 4 | 3 | ✅ **DONE here.** |

## Phase 5 — Code integration checklist

| Task | Verdict |
|---|---|
| `product/api.py` new | ❌ Don't. Product is TypeScript; extend `prediction.ts`, not a parallel Python API. |
| `models/digital_twins.py` | ⏸ Defer until segment data exists (see Pillar 2). |
| `models/multimodal_encoder.py` | ⏸ Large; coordinate with teammate. |
| `frontend/dashboard.tsx` | ⏸ Surfaces exist; coordinate with PR #96. |
| `.github/workflows/validation.yml` CI | ◑ **Cheap + real.** `neurosignal/tests/` already has 13 test files — wire them into CI. Good standalone next commit. |

## Attack order (easy × relevant × important, non-overlapping)

1. ✅ **Case Study A+C retrospective** — this folder. Shipped.
2. 🎯 **Prolific study (B)** — ~$500, the real-human validation. Highest marginal proof.
3. ◑ **Construct-validity report** — wrap existing `validation.py` primitives into one artifact.
4. ◑ **CI wiring** (`neurosignal/tests` → GitHub Actions) — cheap, protects everything above.
5. ⏸ **NSD encoder validation** — the big one; fund deliberately, it needs the Modal encoder + data.

> Sequencing rule applied (per request): each step is **self-contained** — no "run tests then
> immediately add a feature." CI wiring (4) is deliberately placed *after* the artifacts it
> protects exist, not before.
