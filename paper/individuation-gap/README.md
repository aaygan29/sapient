# Who, not what: the individuation gap in in-silico neuroforecasting

A submission-ready paper arguing that non-invasive and in-silico brain/biosignal readouts can
recover **who** a person is (identity) and forecast **populations**, but cannot read **what** an
individual is thinking or will do, because the individual signal is not in the input without a
per-subject enrollment price, and that price currently buys identity rather than behavior.

**Read the paper:** [`paper/main.pdf`](paper/main.pdf) (5 pages, two-column).

## Five main claims
1. **The individuation gap** (Lean-formalized): aggregation strictly improves the population term; the zero-shot individual term is exactly zero. `proof/IndividuationGap.lean`.
2. **Who, not what**: across three real datasets, identity is recoverable (fingerprint 100%; behavioral fingerprint 14× chance) but individual behavioral content abstains.
3. **Probabilistic falsification**: a Bayes-optimal forecaster adds exactly zero individual value at zero-shot (recovery r=0.00, ΔAUC=0.00); enrollment buys only a modest, saturating gain. `code/prob_falsification.py`.
4. **Fusion is not a free lunch**: multimodal fusion reduces variance only under independence; added modalities inject nuisance variance, so individual ρ² is non-monotone.
5. **Grounded in the reliability literature** (Marek 2022, Elliott 2020, Gordon 2017) and bounded against invasive methods that *do* reach the individual (optogenetics; intracortical BCI).

## Contents
- `paper/main.tex`, `paper/main.pdf` — the paper (canonical).
- `figures/` — publication figures (PDF + PNG), regenerate with `code/make_figures.py`.
- `proof/IndividuationGap.lean` — the gap theorems (Lean 4 / Mathlib).
- `code/make_figures.py`, `code/prob_falsification.py` — reproducible figure + simulation code.
- `NOVELTY.md`, `INTEGRATION.md` — the honest novelty gate and how prior projects supply the proof and real-data arms.

## Reproduce
```bash
# figures + simulation (needs matplotlib, numpy, scipy)
python code/make_figures.py
python code/prob_falsification.py
# paper
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Honest status
Position / methods paper with a formal result, a probabilistic falsification, and convergent
real-data evidence. The real-data arms were run under separate protocols in prior work
(convergent evidence, not one preregistered study); the decision-fMRI individual link abstains at
the boundary of power; all numbers should be regenerated from frozen commits before camera-ready.

## Venue
See the conversation's venue note. NeurIPS 2026 workshop deadlines (Aug–Sep) and ICLR 2027
(Sep 25) are closed as of Oct 2026. Best near-term fundable targets: a NeurIPS 2026 Paris
satellite workshop with a late/rolling CFP if open, the N.E.W. (Neuroscience of the Everyday
World) conference (Nov 2026, poster track), or an OpenReview/EasyChair venue with an open
deadline. Format is adaptable to a NeurIPS/ICLR workshop style file.
