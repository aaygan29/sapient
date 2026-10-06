# in-silico-run/

**The case studies executed end-to-end, in silico, with the strengthened engine.** This is the
"run it all the way through" folder: descriptors → `neurosignal` constructs → the data-grounded
`neuroforecast` composite (aggregate mode, bootstrap 95% CI), scored under **both** references —
coordinate-grounded *and* the **real empirical fMRI** reference (NeuroVault mixed-gambles gain/loss
group maps projected onto Schaefer-2018/Yeo-7).

## Run
```bash
cd case-studies/in-silico-run
python3 run_in_silico.py
```
Regenerates `figures/`, `REPORT.md`, and `results_in_silico.json`. Deterministic (fixed seeds).

Prereq (already committed, only needed to rebuild the empirical reference from scratch):
```bash
cd ../../neurosignal
python3 -m neurosignal.data_derive.derive_mid_empirical_reference   # pulls real fMRI from NeuroVault
```

## Outputs
| File | What |
|---|---|
| `REPORT.md` | The write-up + per-ad table + both figures. **Start here.** |
| `figures/insilico_buy_sell_ci.png` | Per-ad buy/sell with bootstrap 95% CI (empirical reference). |
| `figures/insilico_ref_comparison.png` | Coordinate vs empirical reference sensitivity. |
| `results_in_silico.json` | Machine-readable. |

## Honest reading (do not strip)
The empirical reference is **real group fMRI** but **cortical-only** (Schaefer has no subcortical
NAcc), so it conflates attention with aversion and skews high-arousal ads pessimistic — a genuine
cortical-proxy limitation, surfaced not hidden. Every score is a **predicted** read-out on an
expert descriptor. This demonstrates the machinery with calibrated intervals; it is **not**
validation of the predicted-brain → behavior link. See
[`../../neurosignal/CHAIN_AUDIT_2026-07-16.md`](../../neurosignal/CHAIN_AUDIT_2026-07-16.md).
