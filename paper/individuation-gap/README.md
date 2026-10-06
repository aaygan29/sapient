# Who, not what: the individuation gap in in-silico neuroforecasting

Non-invasive and in-silico brain/biosignal readouts can recover **who** a person is (identity) and
forecast **populations**, but cannot read **what** an individual is thinking or will do: the
individual signal is not in the input without a per-subject enrollment price, and that price
currently buys identity rather than behaviour.

## Submission builds
- **`tmlr/`** — the primary, anonymized **TMLR** submission (archival journal, rolling, OpenReview).
  `tmlr/main.pdf` is the paper; `tmlr/supplementary.zip` is the code + proofs;
  `tmlr/OPENREVIEW_FIELDS.md` has every copy-pasteable submission field. Official `tmlr.sty` included.
- **`facct/`** — the **ACM FAccT** version, reframed around accountability and transparency
  (compiles on Overleaf; FAccT 2027 CFP not yet open).
- **`paper/`** — a generic single-column build of the same paper.

## Results
1. **The individuation gap** (Lean 4): aggregation strictly improves the population term; the zero-shot individual term is exactly zero.
2. **Who, not what**: identity is recoverable (fingerprint 100%; behavioural fingerprint 14x chance) but individual behavioural content abstains.
3. **Probabilistic falsification**: a Bayes-optimal forecaster adds exactly zero individual value at zero-shot; enrollment buys only a modest, saturating gain.
4. **Fusion is not a free lunch**: multimodal fusion reduces variance only under independence; added modalities inject nuisance variance, so individual rho^2 is non-monotone.
5. Grounded in measurement-reliability results; bounded against invasive methods (optogenetics; intracortical BCI) that do reach the individual.

## Reproduce
```bash
python code/make_figures.py        # figures (matplotlib, numpy, scipy)
python code/prob_falsification.py  # Bayesian enrollment simulation
cd tmlr && pdflatex main.tex && pdflatex main.tex
```

## Status
All statistics verified against the analysis outputs; all citations verified against PubMed.
Synthetic and Bayesian results validate the method and the probability claim. Real-data numbers
should be regenerated from a fixed commit before camera-ready. Both TMLR and FAccT are archival;
submit the same content to only one.
