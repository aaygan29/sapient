# A grounded measure battery for Sapient

**2026-07-16. Written after the specificity gate withheld the Manipulation Index.**

The gate withheld Manipulation because there is no validated localizer for "persuasion" (0 Neurosynth
studies out of 14,371). That is not an isolated defect. It is the predictable result of a naming
convention, and the same convention produced 9 of 10 KPIs whose named construct is not their best
spatial match. This document proposes the replacement.

**Citation confidence is marked throughout.** Entries marked `[recall]` are from memory and MUST be
verified against the actual paper before any of this reaches a customer or a manuscript. Entries
marked `[read 2026-07-16]` were read in full today. This program has already retracted one paper for
a confound and one for invalid code; a battery built on remembered citations would be a third.

---

## 1. The principle: name the operation, not the hope

The current battery is named for **constructs**: Manipulation, Reward Valuation, Emotional Salience.
Each name is a claim that the measured pattern indexes that construct and not something else. That
claim is reverse inference, and it is the thing Neurosynth exists to check. It failed for 9 of 10.

A measure should be named for **what it computes**, so its validity claim is falsifiable:

| Reverse-inference name (unfalsifiable) | Operational name (falsifiable) |
|---|---|
| "Manipulation Index" | "Limbic-minus-frontoparietal contrast" |
| "Reward Valuation (cortical proxy)" | "Limbic+Default network mean" |
| "Visual Attention" | (not computable: visual streams are gated off) |

Rename first. A measure that cannot survive being described literally should not ship.

---

## 2. The evidence ladder

| Tier | Meaning | Ships to customers? |
|---|---|---|
| **T0** | Content baseline. What the stimulus alone predicts, with no brain at all. | As the comparison, always |
| **T1** | **Measured** neural index with a published behavioral link | Yes, with conformal interval |
| **T2** | **Predicted** (in-silico) neural index. Inherits T1 validity ONLY if the encoder is validated for that ROI and beats T0 (Gate 1). | Only after Gate 1 clears |
| **T3** | Construct-labeled score with no validated localizer | **Never.** Withheld by the specificity gate |

Everything Sapient ships today is T3 or an un-gated T2. The GoEmotions run is the only adequately
powered Gate 1 test we have, and it did not clear (+0.004, CI [-0.024, +0.031], recomputed
2026-07-16).

---

## 3. The architectural constraint that kills the best measure

**The single best-validated neuroforecasting signal is NAcc anticipatory affect.** It predicts
aggregate market outcomes where behavioral self-report fails, replicated across crowdfunding,
microlending, video engagement and stock markets `[recall: Knutson et al. 2007 Neuron; Genevsky &
Knutson 2015; Genevsky, Yoon & Knutson 2017; Tong et al. 2020; Stallen, Borg & Knutson 2021]`.

**Sapient cannot measure it.** NAcc is subcortical. Mary/Sapient-1 predict 20,484 fsaverage5 vertices,
a *cortical surface*. Yeo-7 is `Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default` — all
cortical. `types.py:13` already concedes this: `cortical_proxy: bool  # subcortical region
approximated by a cortical proxy`, and "Reward Valuation (cortical proxy)" admits it in its own name.

A cortical proxy for NAcc is not NAcc. It is a T3 claim wearing a T1 citation. **The Knutson lineage
cited in `constructs.py:10` does not transfer to this architecture**, and citing it as if it does is
the most consequential overclaim in the product.

Options, honestly: (a) drop the NAcc claim; (b) add subcortical coverage to the encoder (a real
modeling project, not a config change); (c) validate the cortical proxy against measured NAcc on a
paired dataset and report the attenuation. Today the product implicitly assumes (c) has been done.
It has not.

---

## 4. The battery

### T1-A. Neural reliability (ISC) — **the recommendation**

**What it computes:** inter-subject correlation of the response timecourse. How similarly different
brains track the same stimulus, second by second.

**Why it is the right headline measure:**
- **It is cortical.** No subcortical proxy problem. Measurable on exactly what Mary predicts.
- **It has a direct published link to audience preference**, which is the product's actual question.
  `[VERIFIED 2026-07-16 via nature.com/articles/ncomms5567: Dmochowski, Bezdek, Abelson, Johnson,
  Schumacher & Parra 2014, "Audience preferences are predicted by temporal reliability of neural
  processing", Nature Communications 5:5567, 29 Jul 2014.]` My recalled citation said 5:4567. It is
  **5567**. That is the kind of error this section exists to catch.

  The verified abstract is **stronger and more useful than I remembered**, in three ways:
  - **It is EEG, not fMRI.** Cheap, deployable, and directly relevant to the EEG direction.
  - **ISC of evoked EEG predicts interest and preference "among thousands"**, against real broadcast
    TV with known audience response (social media activity + ratings). That is Sapient's exact
    question, already answered in the literature, with the modality we can actually deploy.
  - **"Ratings of the larger audience are predicted with GREATER accuracy than those of the
    individuals from whom the neural data is obtained."**

  That third finding is load-bearing and cuts two ways at once. **ISC is a POPULATION instrument, not
  an individual one.** It predicts the crowd better than it predicts the person in the scanner. So:
  - It is an excellent fit for B2B brand intelligence (predict the audience), which is what the
    company sells today.
  - It is evidence AGAINST the personal-digital-brain thesis, and it converges with our own N=8
    "encoding is not identity" null and with the Gate-2 individuation problem. Three independent
    lines now say the same thing: the aggregate is tractable, the individual is not.
  Do not market ISC as personalization. It is the opposite of personalization, and that is why it
  works.
- **It links to memory** `[recall: Cohen & Parra 2016, eNeuro]`, and memorability is a real behavioral
  outcome we already hold data for (BMD, 1,102 videos).
- **It is model-free.** ISC is a property of the responses, not of a construct label, so it cannot
  fail a specificity gate the way a named construct can. There is no reverse inference to make.
- **We already compute it.** `noise_ceiling_isc.npy` and the `ISC > 0.05` responsive mask are in
  Mary's training path (`mary_multi_stable_resp`). The read-out simply never exposes it.
  `neurosignal` has no ISC at all. **This is the largest gap between what we can measure and what we
  report.**

**Honest limit:** in-silico ISC across *predicted* subjects is not the same object as ISC across
*measured* subjects. If the per-subject heads are near-identical, predicted ISC is inflated toward 1
by construction and measures the model, not the audience. **This must be validated against measured
ISC before shipping**, and it is a T2 until it is.

### T1-B. Anterior insula reversal signal

**What it computes:** AIns activity preceding trend reversals.
**Status:** replicated in our own work (`neurobridge`: AUC 0.595/0.660 across two preregistered
experiments, cluster-bootstrap CIs excluding chance, region-specific — NAcc and MPFC did *not*
replicate in both). AIns is cortical-adjacent and more tractable than NAcc.
**Limit:** modest AUC; the conformal layer abstains on ~75% of trials. That abstention is a feature.

### T2. Predicted-BOLD network contrasts (the current battery, demoted)

Keep them, rename them operationally, and gate them. They are T2 at best and cannot rise to T1 until
they clear Gate 1 against a content-matched baseline in the relevant domain. Note the audit's finding
that the KPI family has **effective rank 2.23** — ten numbers are approximately two. Report two.

### T0. Content baselines (mandatory comparison)

- **Memorability from content**: established today on BMD, r = +0.4487, shuffled control +0.0057
  (`memoryprint/results/bmd_content_calibration.json`). Memorability is largely intrinsic to content
  `[recall: Isola, Xiao, Torralba & Oliva 2011, "What makes an image memorable?", CVPR]`.
- **Lexical sentiment**: AUC 0.714 on GoEmotions (recomputed).

**Any neural measure that does not beat its T0 is not a neural measure. It is a content measure with
a brain-shaped logo.**

### Retired

- **Manipulation Index** — no localizer; withheld by the gate as of PR #122.
- **Visual Attention / Face Processing / Visual Pull** — computed from a Visual network the model
  cannot populate while `ACTIVE_STREAMS = ('beats','whisper','qwen_ctx')`. Currently derived from
  audio and transcript. Either turn the visual streams on (they carry ~10% of accuracy, CI excludes
  zero) or stop reporting these.

---

## 5. From the papers read today

**`[read 2026-07-16]` Zhang, Mehta & McDonald 2026, ACM THRI 15(4) Art. 88, "The Neuroscience of
(Dis)Trust".** fNIRS + gaze, N=57. Proactive takeover → *suppressed* DLPFC + lower trust; no takeover
→ *increased* DLPFC + higher trust.

Its value here is **cautionary, and it is sharp**: this paper reads DLPFC↑ as **trust**, while our own
`prefrontal-manipulative-speech` draft reads DLPFC↑ as **manipulation**. Same region, opposite
constructs, both published claims. They cannot both be diagnostic. This is textbook reverse inference
and an *independent external* confirmation of why the specificity gate exists. **Do not add a DLPFC-
based trust measure.** The lesson is the opposite: DLPFC is among the least specific regions in the
brain and should anchor no construct.

Its *methodological* contribution is worth stealing: it pairs neural + gaze + behavior + self-report
on the same events. That is the T1 shape. Gaze entropy in particular is cheap, deployable, and
behavioral.

**`[read 2026-07-16]` Phadikar, Fouladivanda, Eierud, Iraji, Wu, Paulus, Kuplicki, Misaki & Calhoun
2026, HBM, "Group Joint ICA (gjICA)".** Concurrent EEG-fMRI fusion, N=121, 63 multimodal components.
Real method, correct lineage. **Not a fit for this battery**: resting-state, group-level, descriptive.
It produces components and connectivity, not a stimulus→response encoder. It belongs to the Tier-C
enrollment arm (NATVIEW), not to a customer-facing measure.

**`[read 2026-07-16]` Lalazar, "Introducing Descartes", Hemispheric, 2026-07-16.** 6B-param EEG model,
250k hours, 100k participants, claims generalization to unseen people. **Strategic, not technical**:
it is the third death of encoder-as-moat (after Meta open-sourcing TRIBE v2 and our own N=8
"encoding is not identity"). It also promises "confidence intervals... like a blood test" — which is
precisely the conformal layer we already have published and shipped. **Calibration is the moat, not
the encoder.**

---

## 6. What I would build, in order

1. **Expose ISC.** It is the best-validated cortical measure of the thing we actually sell, and it is
   already computed in the training path and thrown away. Validate predicted-ISC against measured-ISC
   before it ships as T1.
2. **Rename everything operationally.** Cheap, no science required, and it makes every remaining
   overclaim visible immediately.
3. **Report two numbers, not ten.** Effective rank is 2.23. The other eight are decoration.
4. **Drop or validate the NAcc claim.** It is cited in `constructs.py` and is not measurable on a
   cortical surface model.
5. **Run Gate 1 on video/audio dimensional affect.** The only open test that could promote anything
   from T2 to T1.

---

## 7. Verification debt

Every `[recall]` citation above needs checking against the actual paper before use. I am confident in
the Knutson/Genevsky neuroforecasting lineage and in Dmochowski 2014 as the audience-preference ISC
result, but "confident from memory" is exactly the standard that produced the retractions this
program is still paying for. **Verify before any of this is cited externally.**
