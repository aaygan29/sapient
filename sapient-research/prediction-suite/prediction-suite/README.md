# ag_push — Sapient Prediction Suite

Turns Mary's predicted brain response into a **calibrated, gated forecast of behavior**,
and ships a front-end panel to show it. Built on the public neuroforecasting literature
(Knutson 2007; Falk 2012/2016; Kühn 2016; Genevsky 2025) — see
[`RESEARCH_BASIS.md`](./RESEARCH_BASIS.md).

> Scope note: this suite **forecasts and explains response**. It makes no claim to change
> minds, and it withholds any construct claim that fails the specificity gate. The honesty
> boundary is enforced in code, not just in copy.

Background, the paper syntheses this was built from, the roadmap, and the pricing/go-to-market
spec live in [`docs/`](./docs/).

---

## What each piece does

| Layer | File | One line |
|---|---|---|
| **BehavioralBridge** | `engine/behavioral_bridge.py` | Brain → market forecast, weighting the anticipatory-affect signal (NAcc, mOFC) the literature shows generalizes to aggregate choice. Signed region priors from Kühn 2016; affect weighting from Knutson & Genevsky 2018; sample-robustness flag from Genevsky 2025. |
| **PredictiveValue** | `engine/neuroforecasting.py` | Validates the forecast against real outcomes leave-one-ad-out, with a permutation null; the brain term must beat self-report to earn the claim (Falk 2016). |
| **BrainTrajectory** | `engine/brain_trajectory.py` | Second-by-second engagement/arousal/memory arc; flags attention peaks and drops with concrete edit recommendations. |
| **PersonaMap** | `engine/persona_map.py` | Per-subject phenotype (how a brain deviates from the population) + few-shot transfer to new content from 1–3 examples. |
| **Grounded read-outs** | `engine/mary_readouts.py` | Existing confound-controlled, meta-analytically-grounded construct layer with the specificity gate every claim passes through. |
| **Front-end** | `frontend/PredictionSuite.tsx`, `frontend/preview.html` | Drop-in React panel (Tailwind + lucide + motion, no new deps) and a self-contained static preview that renders with zero build. |
| **API contract** | `api/prediction.ts` | Typed request/response between the engine and the app; wire into `server.ts` or a Vercel function. |
| **Real-data harness** | `data/validate_real.py`, `data/loaders.py` | Runs the validation machinery on a real open ad-spend → sales dataset (ISLR Advertising) as a smoke test. |

## Run it

```bash
cd ag_push/engine && python3 demo.py          # full pipeline, synthetic (no data needed)
cd ag_push/data   && python3 validate_real.py  # validation harness on REAL outcomes
open ag_push/frontend/preview.html             # static front-end preview
```

Verified output (this build):
- `demo.py` — brain r=0.95 vs self-report r=0.63 → **PASS**; trajectory peak :15s; phenotype {value, attention, memory}.
- `validate_real.py` — ad-spend → sales **r=0.868, p=0.0002**; ranks TV > radio > newspaper (the known ISLR result). Machinery is sound on real numbers.

## Wire into the app

1. `frontend/PredictionSuite.tsx` → import as a tab/route in `src/App.tsx`; it renders an
   illustrative payload until you pass a live `forecast`.
2. `api/prediction.ts` → add as a route in `server.ts` (or a Vercel serverless fn); the
   handler shells out to the engine or the `sapient-serving` model pool.
3. Engine reads Mary's 20,484-vertex predictions; no new frontend dependencies.

## Status & honesty

- **Not yet validated on Sapient data.** Literature priors set the *shape*; only
  `calibrate()` on real ad outcomes sets the *scale*, and only `Neuroforecaster`
  (beats self-report + passes the gate) earns the predictive claim.
- **Real neural validation** needs Mary vertex exports + measured ad outcomes
  (CTR / sales / recall). The call shape is already identical to `validate_real.py`.
- **Provenance is public.** Nothing here derives from private or classified work; the
  cited paradigm (a 400,000-person campaign forecast from 50 brains) is the real,
  defensible lineage.
