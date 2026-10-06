# Who, not what: the individuation gap in in-silico neuroforecasting

A submission-ready paper arguing that non-invasive and in-silico brain/biosignal readouts can
recover **who** a person is (identity) and forecast **populations**, but cannot read **what** an
individual is thinking or will do: the individual signal is not in the input without a per-subject
enrollment price, and that price currently buys identity rather than behavior.

**Read the paper:** [`paper/main.pdf`](paper/main.pdf) (single column).

## Results
1. **The individuation gap** (Lean 4): aggregation strictly improves the population term; the zero-shot individual term is exactly zero. `proof/IndividuationGap.lean`.
2. **Who, not what**: identity is recoverable (fingerprint 100%; behavioral fingerprint 14× chance) but individual behavioral content abstains.
3. **Probabilistic falsification**: a Bayes-optimal forecaster adds exactly zero individual value at zero-shot (recovery r=0.00, ΔAUC=0.00); enrollment buys only a modest, saturating gain. `code/prob_falsification.py`.
4. **Fusion is not a free lunch**: multimodal fusion reduces variance only under independence; added modalities inject nuisance variance, so individual ρ² is non-monotone.
5. Grounded in measurement-reliability results and bounded against invasive methods (optogenetics; intracortical BCI) that do reach the individual.

## Contents
- `paper/main.tex`, `paper/main.pdf` — the paper (canonical, single-column).
- `figures/` — publication figures (PDF + PNG); regenerate with `code/make_figures.py`.
- `proof/IndividuationGap.lean` — the gap theorems (Lean 4 / Mathlib).
- `code/make_figures.py`, `code/prob_falsification.py` — reproducible figures + simulation.

## Reproduce
```bash
python code/make_figures.py        # needs matplotlib, numpy, scipy
python code/prob_falsification.py
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Target venue
**TMLR (Transactions on Machine Learning Research)**, submitted on OpenReview. TMLR has rolling
submission (no deadline) and evaluates whether claims are correct and supported rather than
novelty or impact, which suits a rigorous result with honest negative findings. See
[`paper/SUBMISSION.md`](paper/SUBMISSION.md). Journal alternative: PLOS ONE or Imaging Neuroscience
(both rolling, soundness-based).

## Status
All stats were verified against the analysis outputs; all citations were verified against PubMed.
The synthetic control ladder and the Bayesian simulation validate the apparatus and the
probability claim. Reported real-data numbers should be regenerated from frozen commits before
camera-ready.
