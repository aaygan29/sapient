# Novelty assessment: is there a real contribution here?

Written as an honest gate, not a pitch. The question is whether anything in the Sapient
program is a contribution that has not already been made. The answer is a qualified yes,
confined to a narrow wedge, and only if the paper is framed as a methods-and-impossibility
result rather than an empirical breakthrough.

## What is already done (do not claim these)

- **Neuroforecasting of aggregate consumer behavior** is a mature field with its own reviews
  and theory. Yao et al. (2024, *Journal of Consumer Behaviour*) survey the theories
  (affect-integration-motivation, frontal asymmetry, inter-subject correlation) and the
  metrics. Rawnaque et al. (2020, *Brain Informatics*) review the technology. A paper whose
  message is "neural signals forecast market behavior" adds nothing.
- **Reverse-inference risk and Neurosynth checks.** That regional activation does not imply a
  construct, and that meta-analytic databases can quantify specificity, is standard (Poldrack
  2006; Hutzler 2013, *NeuroImage*; Calzavarini et al. 2022, *Synthese*). Re-stating it is not
  a contribution.
- **Conformal prediction, population-vs-individual designs, LOO optimism, permutation and
  external-validation controls.** All established (Vovk et al. 2005; Yeung et al. 2022,
  *NeuroImage*; Gell et al. 2024, *Neuropsychopharmacology*; Chopra et al. 2024, *Science
  Advances*). Using them is good practice, not novelty.

## The wedge that appears to be open

Every result above assumes **measured** neural data. The Sapient setting forecasts from a
**predicted (in-silico) brain** produced by a stimulus-to-cortex encoder. That shift changes
what must be true for a forecast to be valid, and those conditions do not appear to be
formalized anywhere:

1. **Content-baseline gate for predicted neural indices.** When the "neural" signal is itself
   predicted from stimulus features, it can inherit the stimulus's predictive power without
   adding anything. The gate: a predicted index must beat a content-only model built from the
   same features before it may be called neural. This is a specific, falsifiable condition for
   encoder-based neuromarketing, and it is where most such claims fail.

2. **A structural impossibility result for zero-shot individuation.** A zero-shot encoder
   cannot produce person-specific neural features for a subject it never saw. Therefore
   individual-level forecasting value is undetectable by construction in the unseen regime,
   regardless of model quality. We show this with a wrong-subject permutation crossed with an
   enrolled-versus-unseen split: the individual-value gate fires only when individual signal
   exists and the subject is enrolled, and is provably silent otherwise.

3. **A refusal-first protocol** combining the above with the specificity gate, so the system
   returns a construct-labeled number only when it has a localizer, beats content, and carries
   a calibrated interval.

The honest novelty is the combination of (1) and (2) applied to in-silico neuroforecasting,
plus the negative finding that most construct measures do not clear (1) and that (2) is
structural, not a tuning problem.

## Why this is not yet a strong-venue paper

- **All validation is synthetic.** The control ladder recovers known truth on simulated data.
  No real neural-plus-behavior test has cleared the gates. A main-conference empirical claim
  would be rejected on this alone, correctly.
- **The ingredients are individually known.** The contribution is synthesis plus an
  impossibility result, which is a methods/position contribution, not a discovery.

## Honest placement

- **Fit:** a methods or position paper, or a registered report. Candidate venues: a NeurIPS or
  ICLR workshop on trustworthy or evaluation methods, *Imaging Neuroscience*, or *Aperture
  Neuro*. Not a main-track empirical submission in its current state.
- **What would upgrade it:** run the content-baseline gate and the two forecasting gates on one
  real paired neural-plus-behavior dataset with a verified absence of encoder training leakage.
  If a predicted index beats content on real data, claim (1) becomes empirical. If it does not,
  the negative result is itself the paper.

## Bottom line (standalone)

There is a real contribution, but it is narrow: validity conditions and a zero-shot
individuation impossibility result for forecasting from predicted brains. Framed that way it is
not redundant. Framed as "we forecast behavior from brain activity" it is redundant and should
not be submitted.

## Upgrade via integration (see INTEGRATION.md)

The narrow wedge becomes a substantially stronger, harder-to-dismiss contribution once the
program's existing assets are integrated, because they supply exactly the proof and real data the
standalone note lacked:

- A **Lean 4 machine-checked proof** that population aggregation strictly improves forecasting
  (value-decision-stack) gives the population term a formal guarantee.
- **Three independent real paired datasets** in three domains converge on the gap: vision
  encoding (digital-brain, real 7T fMRI) shows encoder-bounded individuation; decision fMRI
  (decision-phenotype, NARPS) shows the individual link abstaining while population and cross-lab
  links establish; the NARPS gamble arm (behavioral_decoding) shows the content baseline
  dominating the aggregate.

With these, the claim is no longer "here is a gate" but "the individuation gap is a structural
law of in-silico neuroforecasting, proven and convergent across domains." That is a position /
methods paper with a formal result and multi-dataset support, which is genuinely novel. The
remaining honest gaps: the arms were run under separate protocols (so it is convergent evidence,
not one pre-registered study), one arm (affectprint on Emo-FilM) is not yet run, and all numbers
need re-verification against the repos. The strongest form is a registered report that runs all
arms under one protocol.
