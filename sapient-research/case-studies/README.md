# case-studies/

Standalone, investor-facing case studies. **Isolated from the product** — nothing here is
imported by `server.ts`, `src/`, or the API. Safe to hand to investors or delete without
touching the app.

## Contents

| File | What |
|---|---|
| `INVESTOR_BRIEF.md` | The one-page story + the two figures. Start here. |
| `retrospective_ad_pilot.py` | Runnable pilot — scores famous ads through the real `neurosignal` engine and regenerates figures. |
| `ad_descriptors.json` | Transparent, auditable ad → Yeo-7 descriptor + outcome dataset with sources. |
| `figures/` | Generated PNGs (regenerated on every run). |
| `results.json` | Machine-readable scores + Spearman ρ (regenerated). |
| `in-silico-run/` | **The full chain executed end-to-end** with the strengthened engine + real empirical-fMRI reference + bootstrap CIs. See its `REPORT.md`. |
| `INVESTOR_PANEL.md` + `investor_panel.py` | Product framing: grounded / discriminating / explainable, and how scores are justified relative to each other. |
| `validation/` | **The case study we run to PROVE it** — preregistered, powered, falsifiable test that these neural indicators predict ad success. Start at `validation/PREREGISTRATION.md`. |

## Honesty contract

These are **predicted construct scores on expert descriptors**, not measured brains, and
**not** validation of the predicted-brain → behavior link. See
[`../neurosignal/CHAIN_AUDIT_2026-07-16.md`](../neurosignal/CHAIN_AUDIT_2026-07-16.md).
Every figure states this in red. Do not strip the caveat.

## Run

```bash
cd case-studies
python3 retrospective_ad_pilot.py
```

Requires `numpy` + `matplotlib` (already used by `neurosignal`). No network, no secrets,
deterministic.

## Roadmap slot

This is **Case Study A (retrospective political) + C (competitive benchmark)** from the
investor roadmap, delivered honestly at $0. Case Study B (Prolific validation, ~$500) is the
funded next step and is intentionally *not* faked here.
