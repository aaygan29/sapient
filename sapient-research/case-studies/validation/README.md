# validation/ — the case study we run to *prove* the indicators work

This is the real test of the product claim: **do these specific neural indicators predict ad /
marketing success?** It is designed to be able to fail.

| File | What |
|---|---|
| `PREREGISTRATION.md` | The falsifiable claim, frozen analysis, power, and kill criteria. **Read first.** |
| `validate_indicator.py` | The harness: cross-validated ρ, permutation p, bootstrap CI, and incremental validity over self-report. Plug real `(indicator, outcome, self_report)` into `run(...)`. |
| `power_analysis.py` | Proves feasibility: required n at the published Genevsky-Knutson effect sizes. |
| `figures/` | `validation_demo.png` (harness recovers a true effect), `power_curves.png`. |

## Run
```bash
python3 power_analysis.py       # feasibility: aggregate effect needs n≈30
python3 validate_indicator.py   # harness operating-characteristics demo
```

## The honest state
- **Proven already (published):** the core indicator (NAcc reward anticipation) forecasts
  *aggregate* market outcomes better than behavior (Genevsky-Knutson). That is the existence proof.
- **Proven here (machinery):** the harness recovers a true effect and shows neural beating
  self-report on data simulated at that effect size.
- **Not yet proven here (real ads):** the live Prolific / campaign run. That is the ~$500 next
  step, and it is powered (n=100 ⇒ MDES ρ≈0.28). When it runs, this same harness returns the
  verdict — pass or fail — with no post-hoc wiggle room.

Nothing in this folder claims real ad outcomes have been predicted yet. It proves the claim is
**testable, powered, and grounded in published effects** — and gives the exact instrument that
will settle it.
