# Sapient

A consolidated archive of the Sapient research program: an in-silico neuroforecasting system that predicts cortical responses to media and attaches calibrated, falsifiable behavioral claims to them. This repository is a personal record of the work, assembled as the company winds down.

The central thesis of the program is stated in [`PAPER.md`](PAPER.md): the stimulus-to-cortex encoder is not the durable asset. The durable asset is the calibration layer that decides, per measure, whether a predicted neural signal has earned the behavioral claim attached to it. Most construct scores do not clear that bar, and the system is designed to withhold them when they do not.

## Layout

| Path | What it is |
|---|---|
| [`PAPER.md`](PAPER.md) | Synthesis paper: the evidence ladder, the specificity gate, and the control-ladder decomposition of forecasting value into population and individual terms. |
| `sapient/` | The application and research monorepo. React/TS frontend, Express/API backend, and the `neurosignal` Python package that maps predicted cortical activation to constructs and a buy/sell composite. |
| `sapient-models/` | The Sapient-1 and Sapient-2 stimulus-to-cortex encoders. |
| `neurosignal/` | Standalone copy of the `neurosignal` package (construct mapping, conformal validation, enrollment). See its `EVIDENCE.md` for the literature behind every construct. |
| `neurobehavioral-mve/` | The minimum viable experiment: a control ladder with two preregistered gates, validated on synthetic ground truth. See `VALIDATION_REPORT.md`. |
| `Whitepaper/`, `DARPA_AI_Forge/` | Program documents and capability statements. |
| `docs/` | Supporting write-ups, including [`DATA.md`](docs/DATA.md) for excluded datasets. |
| `*.pdf` | Program papers and briefs. |

## What is and is not included

- **Model weights** are stored with [Git LFS](https://git-lfs.com). Clone with `git lfs install` configured to retrieve them, or the checkpoint arrives as a pointer file.
- **The OpenLAV / LIRIS-ACCEDE video dataset is not included.** It is third-party research data under a redistribution-restricted EULA. The ingestion scripts, metadata maps, and derived feature caches are kept so the pipeline is reproducible once the dataset is obtained directly from its source. See [`docs/DATA.md`](docs/DATA.md).
- Dependency trees (`node_modules`, virtual environments) and caches are excluded; reinstall from the lockfiles in each component.

## Reproducing the synthetic apparatus check

```bash
cd neurobehavioral-mve
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 mve_control_ladder.py
```

This recovers the gate verdicts reported in `VALIDATION_REPORT.md` on synthetic data. It validates the analysis apparatus, not any brain.

## Status of the science

The honest current status, documented across the component READMEs and in `PAPER.md`: the synthetic control ladder recovers the correct verdict in every cell of its crossed design with zero false positives, and no real-data construct measure has yet cleared the evidence ladder to a validated behavioral tier. The withholding is the intended behavior, not a gap to paper over.
