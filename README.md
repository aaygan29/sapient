# Sapient

In-silico neuroforecasting: predict the cortical response to a piece of media, then attach a calibrated, falsifiable claim about how an audience will behave. This repository is a curated record of the research program, assembled as the company winds down. It keeps the analysis and model code that explains how the system works, and leaves out the application plumbing (frontend, billing, infrastructure).

The argument of the whole program is in one place, [`PAPER.md`](PAPER.md): the stimulus-to-cortex encoder is not the durable asset. The durable asset is the calibration layer that decides, per measure, whether a predicted neural signal has earned the behavioral claim attached to it. Most construct scores do not clear that bar, and the system withholds them when they do not.

## How the pieces fit

```
stimulus ──> encoder ──> predicted cortical activation ──> constructs ──> calibrated read-out
           (models/)                                     (neurosignal/)   (gates + conformal)
                                                                                │
                              validated against behavior ─────────────────────┘
                              (experiments/, sapient-research/)
```

## Layout

| Path | What it is | Maps to |
|---|---|---|
| [`PAPER.md`](PAPER.md) | The centralized research paper: evidence ladder, specificity gate, and the control-ladder decomposition of forecasting value into population and individual terms. | whole program |
| `models/` | The stimulus-to-cortex encoders: `sapient1`, `sapient2`, and the `mary` / `qualia` lines. Encoder code and configs; trained weights are not included (see below). | Section 1, 2 |
| `neurosignal/` | Maps predicted activation to constructs and a buy/sell composite, with split-conformal intervals, the specificity gate, and per-subject enrollment. See its `EVIDENCE.md`. | Sections 3, 4 |
| `experiments/control-ladder/` | The minimum viable experiment: a nested control ladder with two preregistered gates, validated on synthetic ground truth. See `VALIDATION_REPORT.md`. | Section 5, 6.1 |
| `sapient-research/buy-moment/` | Buy-moment detectors, connectome and multivariate fusion, and validation against behavior. | Section 6.2 |
| `sapient-research/eval-pipeline/` | The fMRI evaluation and fine-tuning pipeline (download, parcellate, fine-tune, run on Modal). | Section 2, 6 |
| `sapient-research/case-studies/` | Validation, in-silico runs, and the semantic demo, with pre-registrations and power analyses. | Section 6 |
| `sapient-research/serving/` | The serving layer for the read-out, with a client, examples, and tests. | Section 7 |
| `sapient-research/prediction-suite/` | The prediction suite and its research basis. | Sections 3, 6 |
| `sapient-research/docs/` | Architecture and API notes. | reference |
| `Whitepaper/`, `*.pdf` | Program papers and briefs. | reference |

## Running the core experiment

The synthetic apparatus check behind the paper's main table:

```bash
cd experiments/control-ladder
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 mve_control_ladder.py
```

This recovers the gate verdicts in `VALIDATION_REPORT.md` on synthetic data. It validates the analysis apparatus, not any brain.

## What is deliberately not here

- **Trained model weights** (for example the ~700 MB TRIBE-v2 checkpoint) are not committed. They are large and not regenerable from this repository alone. See [`docs/DATA.md`](docs/DATA.md).
- **The OpenLAV / LIRIS-ACCEDE video dataset** is third-party research data under a redistribution-restricted EULA and is excluded. The ingestion scripts and metadata maps are kept so the pipeline is reproducible once the dataset is obtained from its source. See [`docs/DATA.md`](docs/DATA.md).
- **Application plumbing** (React frontend, billing, auth, infrastructure) is left out on purpose. This repository is the research, not the product deployment.

## Honest status

The synthetic control ladder recovers the correct verdict in every cell of its crossed design with zero false positives. No real-data construct measure has yet cleared the evidence ladder to a validated behavioral tier. The withholding is the intended behavior, documented in `PAPER.md` and the component READMEs.
