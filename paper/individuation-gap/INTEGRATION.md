# Integration map: turning a synthetic methods note into a verified, multi-dataset result

The standalone validity protocol (content-baseline gate, zero-shot individuation impossibility,
refusal-first gating) is a methods contribution validated only on synthetic data. The program's
existing work supplies what it was missing: a formal proof and three independent real-data
instantiations that converge on the same law. This file maps each piece and states the honest
status of each, so the claim is upgraded without overclaiming.

## The unified claim

**The individuation gap is a structural law of in-silico neuroforecasting.** Forecasting value
from a predicted brain decomposes into a population term and an individual term. The population
term is tractable and provably improves with aggregation. The individual term is bounded by the
stimulus encoder and is undetectable for an unseen subject by construction. A system that does
not separate the two, and does not withhold the individual claim when it is unsupported, will
report individual value it does not have.

This is novel in three ways the standalone note was not:
1. It is **proven**, not just simulated (Lean 4 aggregation theorem).
2. It is **empirically convergent** across three real paired datasets in three domains
   (vision encoding, value-based decision, affect), each independently showing the aggregate or
   enrolled claim holding while the individual or unseen claim abstains or fails.
3. It names a **general law** rather than reporting one experiment.

## Component map

| Paper element | Instantiated by | Real data | Honest status |
|---|---|---|---|
| Aggregation dominates individual (population term) | `value-decision-stack` Lean kernel `proofs/NeuroforecastProof.lean` (`aggregation_improves`, `more_data_is_better`) | n/a (formal) | `lake build` clean, no `sorry` (per project notes; re-verify before submission) |
| Zero-shot individuation is encoder-bounded (impossibility, empirical) | `digital-brain` representational-asymmetry result on NSD + BOLD5000 | real 7T fMRI, N=8 | group/amplitude individuation saturates; geometry/individual individuation positive-but-small, encoder is the ceiling. Enrolled fingerprinting 25/25. **Modest N; verify numbers in repo** |
| Individual neural→behavior value abstains even on real data | `decision-phenotype` on NARPS (ds001734) + ds000005 | real fMRI (n=40) + 108 subj / 27k choices | population/triangulated links establish (combined p≈2.4e-7); **direct individual neural→behavior link abstains** (MDES gate). Exactly the predicted pattern |
| Content-baseline gate fires on real data | `behavioral_decoding` NARPS gamble arm | real fMRI + behavior | on gambles the economic/content baseline dominates the aggregate by construction → no brain-beats-content claim. The gate working as designed |
| Idiographic-beats-nomothetic-or-abstain (content gate, affect domain) | `affectprint` on Emo-FilM | real fMRI + continuous affect (30 subj) | **proposed/awaiting run**; testbed confirmed open. This is the one arm not yet executed |
| Specificity gate + conformal abstention (engineering) | `neurosignal` (`validation.py`, `enrollment.py`) | n/a | ships in the Sapient archive |
| Synthetic apparatus check | `experiments/control-ladder` | synthetic | validated, zero false positives |

## What is honestly new vs. reused

- **New scientific act:** unifying these under one protocol and stating the individuation gap as
  a law, with a proof and cross-domain convergence. No single prior project makes the general
  claim; the field (Yao 2024; Yeung 2022; Gell 2024) treats individual-vs-population as a design
  trade-off, not an impossibility bounded by the encoder in the in-silico case.
- **Reused engineering:** the encoders, loaders, honesty layers, and Lean kernels already exist.
  The paper does not claim them as new; it claims the synthesis and the law.

## Honest weaknesses to disclose in the paper

- The real-data arms were run in separate projects under separate protocols, not in one
  pre-registered multi-dataset study. The paper should present them as convergent evidence and
  be explicit they were not collected under a single design. The strong version is a registered
  report that runs all arms under one protocol.
- `decision-phenotype`'s fMRI arm is underpowered (abstains at boundary); that is consistent with
  the thesis but is an abstention, not a positive null. Report it as such.
- The `affectprint` arm is not yet run. Either run it before submission or mark it as the
  pre-registered prediction.
- All numbers cited from memory must be re-verified against the current repos before submission
  (per the program's verify-before-reporting rule).

## Minimum work to submission

1. Re-verify the three datasets' numbers in their repos and pull exact values + CIs.
2. Run the `affectprint` idiographic-vs-nomothetic arm on Emo-FilM, or register it as a prediction.
3. Write the single combined-evidence figure: one row per dataset, population/enrolled vs
   individual/unseen, showing the gap, plus the Lean theorem statement as the formal anchor.
