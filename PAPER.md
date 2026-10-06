# Calibration is the moat: an evidence ladder and specificity gates for in-silico neuroforecasting

Aayush Gandhi

## Abstract

In-silico encoders now predict whole-cortex fMRI responses to arbitrary stimuli, and a natural commercial claim follows: that such predictions can forecast audience behavior and individual preference. We argue that the encoder is not the hard part. Open models already predict cortical responses well, and predictive encoders have been released by several groups, so the stimulus-to-cortex map is close to commoditized. The hard part is knowing when a predicted neural number means what its label says. We present a measurement discipline for in-silico neuroforecasting built from three components: an evidence ladder that separates content baselines from measured and predicted neural indices, a specificity gate that withholds any construct score lacking a validated anatomical localizer, and a control ladder that decomposes forecasting value into a population term and an individual term through two preregistered gates. On synthetic ground truth with known structure, the control ladder recovers the correct verdict in every cell of a crossed world-by-regime design, firing its individual-value gate only when individual neural signal exists and the subject is enrolled, with zero false positives in the null worlds. Applied to real data, the same discipline withholds most construct scores: an emotion index does not beat a lexical-sentiment content baseline, and a named manipulation index is withheld because no localizer for persuasion exists. We report these null and withheld outcomes as the main result. The contribution is not a better brain model. It is a falsifiable protocol that states, per measure, whether a predicted neural signal has earned the behavioral claim attached to it.

## 1. Introduction

A predictive model of the brain invites a specific misuse. Once a model maps a stimulus to a pattern of cortical activation, it is tempting to read that pattern through the names that neuroscience has attached to brain regions, and then to sell those names as measurements of attention, emotion, reward, or persuasion. The inference runs backward: from activation in a region to the presence of a mental construct. This is reverse inference, and it is unreliable whenever a region participates in more than one function, which is the common case.

This paper treats the reverse-inference problem as the central engineering constraint of in-silico neuroforecasting rather than a footnote to it. We take as given a stimulus-to-cortex encoder that predicts activation on a cortical surface, and we ask a narrower question: for which measures derived from that surface can we state a behavioral claim that would survive being described literally and tested against a baseline that uses no brain at all.

Three observations motivate the design.

First, the encoder is not a durable advantage. Large stimulus-to-cortex models have been released openly, and newer sequence models trained on very large electrophysiology corpora promise generalization to unseen people. Any program that locates its value in owning the encoder is building on ground that is already shifting.

Second, the best-validated neuroforecasting signal in the literature is subcortical. Anticipatory activity in the nucleus accumbens predicts aggregate choice where self-report fails, across several domains (Knutson et al., 2007; Genevsky and Knutson, 2015; Genevsky, Yoon and Knutson, 2017). A surface encoder cannot measure it. A cortical proxy for a subcortical nucleus is a different object, and citing the subcortical literature for a cortical proxy is an overclaim, not an approximation.

Third, individual prediction and population prediction are not the same product, and they do not succeed or fail together. A measure can forecast the crowd while failing to forecast the person. Audience-preference work using inter-subject correlation of evoked responses found exactly this asymmetry: reliability of neural processing predicted the preferences of the larger audience with greater accuracy than it predicted the individuals from whom the signal was recorded (Dmochowski et al., 2014).

From these observations we build a discipline with three parts, described in Sections 3 through 5, and we report what happens when it is applied, in Section 6. The recurring result is withholding. Most construct scores do not clear their gate. We argue that a system which reliably refuses to report an unearned number is more valuable than one which always reports a confident one.

## 2. Related work

**Neuroforecasting.** The claim that neural signals forecast aggregate behavior beyond self-report is established for anticipatory reward signals (Knutson et al., 2007; Genevsky, Yoon and Knutson, 2017) and for neural reliability as a predictor of audience preference (Dmochowski et al., 2014). Our work does not add a new forecasting signal. It adds a test that decides whether a predicted signal inherits the validity of a measured one.

**Reverse inference.** The limits of inferring mental states from regional activation are well documented, and coordinate-based meta-analytic databases exist precisely to quantify how diagnostic a region is for a term. We operationalize this by refusing to ship any construct whose name has no adequately powered localizer in such a database.

**Encoding models.** Stimulus-to-response encoders predict cortical activity from video, audio, and language features. We treat the encoder as a component with a known failure mode, not as the contribution, and we require that any predicted index beat a content-only baseline before it is called neural.

**Conformal prediction.** Distribution-free prediction intervals with finite-sample coverage guarantees provide a way to attach honest uncertainty to a point estimate. We use split conformal intervals and empirical coverage as the uncertainty layer, and we treat high abstention as acceptable behavior rather than as failure.

## 3. The evidence ladder

Every measure is placed on a four-tier ladder according to what grounds it.

- **T0, content baseline.** What the stimulus alone predicts, with no brain. This is the mandatory comparison for every neural claim.
- **T1, measured neural index with a published behavioral link.** Shipped with a conformal interval.
- **T2, predicted neural index.** Inherits T1 validity only if the encoder is validated for the relevant region and the predicted index beats its T0 baseline.
- **T3, construct-labeled score with no validated localizer.** Never shipped. Withheld by the specificity gate.

The ladder makes the central rule explicit: a predicted neural number is T2, not T1, until it is shown to beat content. A named construct with no localizer is T3 and does not ship at all. The rule that follows is blunt. Any neural measure that does not beat its content baseline is a content measure with a brain-shaped label.

## 4. The specificity gate

The specificity gate decides whether a construct name is allowed. For a candidate construct, the gate queries a coordinate-based meta-analytic database for studies associated with the construct term and for the spatial match between the construct's canonical regions and the term's meta-analytic map. A construct fails the gate when its term has no adequately powered study set, or when its named regions are not the best spatial match for its own term.

The gate is adversarial by design. It is meant to withhold. In the deployed battery it withheld a manipulation index, because a search for persuasion-related studies returned no usable localizer. The same procedure found that most of a ten-construct battery had a named construct that was not its own best spatial match, which is the signature of naming by hope rather than by operation. The repair is to rename each measure for what it computes, a contrast between networks, so that its validity claim becomes falsifiable, and then to let the gate decide which survive.

## 5. The control ladder and two gates

Forecasting value has two components that must be separated, because they map to two different products: predicting an unseen audience, and predicting an enrolled individual.

We define a control ladder of nested models: content only; a capacity-matched non-brain basis; plus behavior history; plus the correct subject's neural features; and plus a wrong subject's neural features as a permutation control. Two preregistered gates run on this ladder.

- **Gate 1**, population value: the correct-subject model beats the capacity-matched non-brain basis.
- **Gate 2**, individual value: the correct-subject model beats the wrong-subject permutation.

The two gates are crossed with two regimes. In the enrolled regime a subject has real per-subject data, modeled by a leave-one-stimulus-out split. In the unseen regime a held-out subject receives population-only neural features, which faithfully models a zero-shot encoder that cannot individualize a new person.

## 6. Experiments

### 6.1 Apparatus validation on synthetic ground truth

Before trusting the control ladder on real data, we validate that it recovers known truth. We generate synthetic data under a crossed design. The WORLD factor is SIGNAL, where behavior depends on an individual neural component, versus NULL, where behavior is pure stimulus content and neural activity is generic stimulus drive. The SPLIT factor is the enrolled versus unseen regime above. We use ridge regression with cross-validated regularization, pooled out-of-sample coefficient of determination, and a paired bootstrap over 20 seeds.

The individual-value gate fires in exactly one cell, and is correctly silent in the other three.

| World / regime | Gate 1 (population) | Gate 2 (individual) | Expected Gate 2 |
|---|---|---|---|
| SIGNAL / enrolled | pass, +0.626 | pass, +0.449, CI [+0.378, +0.526] | fire |
| SIGNAL / unseen | no | no, +0.028 n.s. | silent |
| NULL / enrolled | pass, +0.087 | no, -0.001 | silent |
| NULL / unseen | pass, +0.034 | no, -0.001 | silent |

Two conclusions follow, both load-bearing for how results may be reported.

First, Gate 1 passes even in the NULL world, because neural features carry the true stimulus-to-behavior basis and beat a random content lift on population grounds alone, with no individuation. Gate 1 therefore measures population value and Gate 2 measures individual value. Reporting a Gate 1 pass as evidence that the system reads an individual would be an overclaim.

Second, in the unseen regime with a faithful population-only neural provider, individuation is undetectable by construction. Any future claim of individual neural prediction for a new person is structurally suspect unless that person was enrolled. The two regimes are reported separately, with different ceilings.

The exercise also caught two bugs before any real data was touched: a degenerate per-fold metric that collapsed when content was constant across subjects at a fixed stimulus, corrected to a pooled out-of-sample metric; and a neural provider that leaked individualized features to unseen subjects, which spuriously fired the individual gate until it was fed population-only features.

These numbers are from synthetic ground truth. They validate the apparatus, not any brain. They are reported as an apparatus check, which is the claim the design supports.

### 6.2 What the discipline withholds on real data

Applied to real measures, the discipline withholds more than it ships.

- An emotion index built from predicted activation does not beat a lexical-sentiment content baseline on an emotion-labeled corpus. The lexical baseline reaches an area under the curve of 0.714. The neural lift over a content-matched baseline is +0.004, with a confidence interval of [-0.024, +0.031], which does not clear Gate 1. The emotion index stays T2.
- A manipulation index is withheld by the specificity gate, because no localizer for persuasion exists in the meta-analytic database.
- Memorability is largely a property of content. On a video memorability dataset, content features predict memorability at a correlation of +0.4487, against a shuffled control of +0.0057. A neural memorability measure must beat this content baseline to earn a T1 label, and reporting memorability without the content comparison would attribute to the brain what the stimulus already explains.

The headline recommendation that survives is a measure we do not label with a construct at all: inter-subject correlation of the response timecourse, a model-free property of the responses with a direct published link to audience preference (Dmochowski et al., 2014). Because it carries no construct label, it cannot fail a specificity gate, and because it predicts the audience rather than the individual, it is honest about being a population instrument. The open limit is that inter-subject correlation computed across predicted subjects is not the same object as across measured subjects, and must be validated against measured reliability before it ships as T1.

## 7. Discussion

The results point one direction. The encoder is commoditizing, the strongest neuroforecasting signal is out of reach of a surface model, individuation does not transfer to unseen people, and most construct-labeled neural measures do not beat content. What remains valuable is the layer that states all of this per measure: the ladder that demands a content baseline, the gate that withholds unvalidated construct names, the two-gate decomposition that separates population value from individual value, and the conformal interval that abstains when it should. The defensible asset is calibration, not the encoder.

This reframes the product question. A system that always returns a confident construct score is easy to build and easy to disprove. A system that returns a number only when that number has beaten its baseline, carried a validated label, and earned an interval is harder to build and harder to dismiss. The withholding is the feature.

## 8. Limitations

The control-ladder results in Section 6.1 are synthetic. They establish that the apparatus recovers known truth and do not establish any neurobehavioral effect. The real minimum viable experiment remains blocked on paired neural and behavioral data with a verified absence of encoder training leakage. The specificity gate inherits the coverage and biases of the meta-analytic database it queries, so a construct can fail the gate for want of studies rather than for want of reality. Inter-subject correlation across predicted subjects risks inflation toward one if per-subject predictions are near-identical, which would measure the model rather than the audience. None of the real-data measures in Section 6.2 has yet cleared the ladder to T1, which is the honest current status of the battery.

## 9. Conclusion

We presented a measurement discipline for in-silico neuroforecasting that separates content from brain, withholds construct names without localizers, and decomposes forecasting value into a population term and an individual term with preregistered gates. On synthetic ground truth the apparatus recovers the correct verdict in every cell with zero false positives. On real data it withholds most construct scores, which is the intended behavior. The contribution is a protocol that decides, per measure, whether a predicted neural signal has earned its behavioral claim.

## 10. Code and reproducibility

This paper is the centralized write-up for the code in this repository. Each claim points to the module that backs it.

| Claim or component | Where it lives |
|---|---|
| Stimulus-to-cortex encoders (Section 1, 2) | `models/sapient1`, `models/sapient2`, `models/mary`, `models/qualia` |
| Construct mapping, buy/sell composite, specificity gate, split-conformal intervals, enrollment (Sections 3, 4) | `neurosignal/` (see `neurosignal/EVIDENCE.md`, `neurosignal/validation.py`, `neurosignal/enrollment.py`) |
| Control ladder, two preregistered gates, synthetic validation (Sections 5, 6.1) | `experiments/control-ladder/mve_control_ladder.py`, `verify_gate1_independent.py`, `VALIDATION_REPORT.md` |
| Content baselines and the emotion and memorability nulls (Section 6.2) | `sapient-research/buy-moment/`, `sapient-research/case-studies/validation/` |
| Evaluation and fine-tuning pipeline (Section 2, 6) | `sapient-research/eval-pipeline/` |
| Read-out serving (Section 7) | `sapient-research/serving/` |

The synthetic apparatus check that produces the Section 6.1 table runs with:

```bash
cd experiments/control-ladder
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 mve_control_ladder.py
```

Trained weights and the OpenLAV / LIRIS-ACCEDE videos are not distributed here; see `docs/DATA.md`.

## References

Bartra, O., McGuire, J. T., and Kable, J. W. (2013). The valuation system: a coordinate-based meta-analysis of BOLD fMRI experiments. NeuroImage, 76, 412-427.

Dmochowski, J. P., Bezdek, M. A., Abelson, B. P., Johnson, J. S., Schumacher, E. H., and Parra, L. C. (2014). Audience preferences are predicted by temporal reliability of neural processing. Nature Communications, 5, 5567.

Genevsky, A., and Knutson, B. (2015). Neural affective mechanisms predict market-level microlending. Psychological Science, 26(9), 1411-1422.

Genevsky, A., Yoon, C., and Knutson, B. (2017). When brain beats behavior: neuroforecasting crowdfunding outcomes. Journal of Neuroscience, 37(36), 8625-8634.

Knutson, B., Rick, S., Wimmer, G. E., Prelec, D., and Loewenstein, G. (2007). Neural predictors of purchases. Neuron, 53(1), 147-156.

Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
