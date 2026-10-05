# Neurobehavioral Prediction Engine: Experimental Protocol (At Scale)

Aayush Gandhi. Pre-registration draft, 2026-06-19. No em dashes.

Source of truth for prior results: `digital-brain/MLCB_2026_submission/` (the N=8 "encoding is not
identity" null), `_program_docs/COUNCIL_REVIEW_PROGRAM_2026-06.md` (Gate 5 fails program-wide),
`_program_docs/VIRTUAL_BRAINS_PROGRAM.md` (the army / synthesis arm), `neurobridge` (conformal
abstention), `mary-readout-validation` (specificity gate). This protocol is written to survive the
Council: every claim is an operational definition with a pre-registered gate and a kill criterion.

---

## 0. The one sentence this experiment answers

**Does a neural prior (predicted brain activity) add out-of-sample predictive value over behavior-alone
and content-alone baselines, for an individual's measured reaction, at scale and under abstention?**

If yes, "neurobehavioral" is real and is the moat. If no, the product is a behavioral engine with a
neuro credibility story, and we say so. Either outcome is a decision, not a failure.

---

## 1. The asymmetric architecture (what we build, not what we collect)

We do NOT collect paired fMRI-plus-behavior at scale. We borrow the neural layer, mint the behavior
layer, and synthesize only the thin bridge.

| Layer | Source | Cost | Role |
|---|---|---|---|
| **Neural (frozen prior)** | TRIBE v2 (Meta, open weights, 1115 h / 700+ subj, zero-shot new subjects). Second encoder: an Algonauts-2025 / VIBE ensemble. | ~0 (rented) | Predicted whole-cortex BOLD as features. Never trained by us. |
| **Bridge (neural to behavior map)** | Public datasets pairing neural responses AND measured behavior for the SAME subjects (Sec 2). | small, fixed | The load-bearing calibration. Learned once, validated hard. |
| **Behavior (the flywheel)** | Sapient telemetry: Scan/Verdict/Mary outcomes, Tier-1 behavioral channels (eye, voice, dwell), real campaign outcomes (CTR, conversion, engagement). | accrues free | The proprietary, compounding asset. The ground truth for the product. |
| **Synthesis (augmentation)** | Virtual-brain army (synthetic connectomes + Hopf dynamics) + LLM population priors. | compute only | Expands the thin bridge; cheap population behavior. Its own validation gate (Sec 6). |

The novelty over Aaru/Simile (pure LLM agents) and over Meta (population encoder, no behavior, no
abstention) is the *fusion plus the honesty layer*: neural grounding + behavioral truth + calibrated
abstention.

---

## 2. Datasets (concrete, public, with role). "Pull from wherever."

### 2A. Bridge datasets (paired neural + measured behavior, same subjects) — the core
| Dataset | N / modality | Behavior label | Role |
|---|---|---|---|
| **Emo-FilM** | ~30 subj, 7T fMRI, naturalistic film | continuous valence/arousal/13 emotion ratings | Primary affect bridge. Continuous reaction is the cleanest Gate-5 target. |
| **NARPS** (ds001734) | 108 subj, fMRI | gamble accept/reject + RT | Decision/value bridge; individual choices. |
| **IBL brain-wide map** | mice, ephys | choice + RT | Cross-species decision generalization (optional gate). |
| **Courtois NeuroMod (Friends)** | 6 subj, deep fMRI | post-hoc engagement / behavioral tasks | Deep per-subject route (proven individual-modeling). |
| **NSD + behavioral** (8 subj, 7T) | dense fMRI | memory/recognition behavior | Memory construct; same 8 subjects span imagery/synthetic for generalization. |
| **HCP** (~1000 subj) | rest + task fMRI | task performance, traits | Identity substrate (Finn 2015 connectome fingerprint) + population manifold for synthesis. |
| **AOMIC / Cam-CAN / MPI-LEMON** | 100s subj, fMRI + behavior | mood, traits, task | Multi-dataset robustness (kills G3 single-dataset fragility). |

### 2B. Identity substrate (for the individual layer)
HCP resting connectomes (the layer that fingerprints people where stimulus-RDMs failed). NSD-iEEG /
tNSD / MEG for high-SNR idiosyncratic channels if available.

### 2C. Proprietary (Sapient)
Scan/Verdict/Mary outcome logs; Tier-1 behavioral channels; any campaign with a downstream metric.
This is the only NEW data, and it is already being minted. Export to a frozen, versioned table with a
provenance manifest (commit + checksum + collection date) per the program lockfile requirement.

All public datasets are downloaded by script with checksums logged; no raw data is committed; a
`DATA_MANIFEST.json` records source, version, checksum, license, and access date for every set.

### 2D. Encoder-training leakage audit (Gate 0) — RESOLVED 2026-06-19 from the downloaded config
The frozen encoder is a leakage vector. We obtained the actual TRIBE v2 training manifest from the
released checkpoint config (`neurobehavioral-mve/tribev2_weights/config.yaml`). **TRIBE v2 was trained
on exactly four studies:**

  `Algonauts2025Bold, Lahner2024Bold, Lebel2023Bold, Wen2017`  (subject_layers: n_subjects = 25)

Audit conclusions (fold into `DATA_MANIFEST.json`):
- **Emo-FilM is NOT in the training set -> leakage-safe. Confirmed primary bridge dataset.**
- **EXCLUDE from eval (confirmed-contaminated):** Algonauts 2025 (built on the CNeuroMod / *Friends*
  movie-fMRI corpus, so NeuroMod/Friends is contaminated as presumed), Lahner 2024 (BOLD Moments),
  Lebel 2023 (narratives story-listening), Wen 2017 (video fMRI). Any bridge stimulus or subject from
  these is disqualified.
- **Enrolled population is concrete:** the 25 subjects with trained `subject_layers`. Out of the box,
  `model.predict` returns the AVERAGE-subject prediction (`average_subjects: false` means individualized
  heads exist only for those 25). This is the hard boundary between the ENROLLED regime (individuation
  possible) and the UNSEEN regime (population-only), exactly as the apparatus validation modeled.
- Second encoder (VIBE / Algonauts-2025 ensemble, M2b arm): repeat this audit on its manifest before use.

---

## 3. The estimand: nested incremental validity (Council-revised)

**The load-bearing problem the Council surfaced:** the neural prior is a deterministic function of the
stimulus. TRIBE-predicted BOLD = f(stimulus, subject), so it is collinear with the stimulus content
already in M0. A naive Delta(M2 - M1) > 0 is therefore consistent with "predicted BOLD is just a
higher-dimensional, more nonlinear stimulus basis than CLIP," which is NOT neuroscience. The estimand
below is restructured so the neural layer must beat two controls that hold stimulus-content and
feature-capacity fixed. This is the single revision that makes a positive result trustworthy.

For outcome y (an individual's measured reaction to a stimulus), compare on held-out data:

- **M0 Content-only:** stimulus features (CLIP/audio/text embeddings). The Aaru-style baseline.
- **M0+ Capacity-matched content:** M0 expanded to the SAME effective dimensionality as the neural
  features, via a non-brain transform (random-weight TRIBE, i.e. the same architecture with
  untrained weights, AND/OR a random-projection expansion of M0). Holds "bigger nonlinear basis"
  constant so any M2 win cannot be attributed to feature capacity alone.
- **M1 Behavior-only:** M0+ plus the person's own behavioral history / Tier-1 channels.
- **M2 Neurobehavioral:** M1 + TRIBE-predicted BOLD features (the neural prior).
- **M2-perm Subject-permuted neural:** identical to M2 but TRIBE predicts a RANDOM OTHER subject's
  brain for each trial. Holds the stimulus-driven (population) component of predicted BOLD constant
  and isolates whether the *individual* neural conditioning carries any weight.
- **M2b Second-encoder:** M2 with the encoder swapped (VIBE / Algonauts-2025 ensemble). Robustness
  arm. NOTE (Gimli): if both encoders are stimulus-driven and leak identically, the swap is not an
  independent control; M2-perm is the independent control, M2b is the robustness check.

**Primary estimand (revised, pre-registered):** the neural layer earns its place only if BOTH hold,
out-of-sample, multi-seed, with bootstrap CIs (over subjects) excluding zero:
1. **skill(M2) > skill(M0+)** — beats a capacity-matched non-brain basis (the "fancy encoder" null), AND
2. **skill(M2) > skill(M2-perm)** — beats the same encoder on the wrong subject (the individuation null).

Delta(M2 - M1) is reported as a secondary descriptive, not the headline. Skill = explained variance for
continuous y; AUC / calibrated log-loss for discrete y; all skills normalized by BOTH the neural SNR
ceiling AND the behavioral test-retest reliability of y (Elrond/Legolas).

**Feature-capacity matching (Elrond):** equal hyperparameter budget is not equal capacity. Match the
effective dimensionality across arms (same PCA rank / same ridge effective-DoF target) so M2 cannot win
on raw dimensionality. Report effective DoF per arm.

**Idiographic estimand, split by regime (Gate-5 fix):** the idiographic>nomothetic test has teeth ONLY
for *enrolled* subjects (where TRIBE is conditioned on real per-subject data). Report two regimes
separately, with different ceilings and different claims:
- **Enrolled regime:** per-person M2 vs nomothetic (group-average) on held-out trials; the genuine
  individuation test. Must BEAT nomothetic out-of-sample, not merely beat chance.
- **Unseen regime:** zero-shot TRIBE for a new person is structurally population-neural; here the only
  legitimate claim is population-level incremental validity (M2 > M0+), NOT individuation. Do not let
  an unseen-regime M2 win be reported as "reads the individual."

---

## 4. Pipeline (engineering, end to end)

1. **Ingest + harmonize.** Download bridge datasets; fMRIPrep where needed; resample all neural targets
   to a common space (fsaverage5, 20484 vertices, the Sapient-1 convention) and a common parcellation
   (Schaefer-400 + 7 networks) so datasets are comparable. ComBat harmonization across sites/scanners.
2. **Stimulus alignment.** Time-align each stimulus to its neural and behavioral stream at 1 Hz.
   Extract content features (CLIP-ViT, Whisper/audio, text encoder) as M0 inputs.
3. **Neural features.** Run TRIBE v2 (frozen) on each stimulus to get predicted BOLD per vertex per
   second. Reduce to per-ROI / per-network summaries + a low-dim manifold (PCA/UMAP) for the model.
   Repeat with the second encoder for the M2b arm.
4. **Behavior features.** Per-subject history, Tier-1 channels, demographic priors.
5. **Identity layer.** Per-subject functional connectome (HCP-style) as a conditioning embedding for
   M2 (connectome-conditioned encoder, the Virtual Brains Phase-1 method).
6. **Models.** Ridge / gradient-boosted baselines first (interpretable, fast), then a shared encoder
   with per-subject connectome embedding. Same hyperparameter search budget for every Mx (fairness).
7. **Specificity gate** (lifted from `mary_readouts.py`): spin-test nulls, ROI-specificity, a sensory
   control contrast. Mandatory, default-on. A construct that fails specificity is not reported.
8. **Conformal abstention** (lifted from `neurobridge`): distribution-free coverage; the engine
   abstains under cross-subject shift rather than confabulate. Target coverage >= 0.90.
9. **Aggregate** with the `Finding` contract (from spikeprint): every number carries
   dataset / baseline / effect size / CI / n / passed-gate.

---

## 5. Cross-validation and statistics (kills single-seed / single-dataset fragility)

- **Splits:** leave-one-subject-out (generalization to new people) AND leave-one-stimulus-out
  (generalization to new creative). Report both. Never split within a stimulus or within a subject for
  the headline number (that leaks).
- **Multi-seed:** >= 20 seeds for every model; report mean +/- SD, not a point estimate. (The encoder
  is deterministic; seeds randomize the head init AND the CV-fold assignment, which is where the real
  variance lives.)
- **Power / minimum-detectable-effect (Gandalf, pre-registered):** before the MVE, run a power analysis
  for the N (~30 for Emo-FilM) under leave-one-subject-out. State the smallest skill gain we care about
  (pre-register MDE, e.g. delta-R^2 >= 0.02 of the behavioral noise ceiling) and confirm 30 subjects can
  detect it at 80% power; if not, expand to NARPS+AOMIC for the MVE or treat a null as
  "underpowered / inconclusive," NOT as a kill. A null is only a true negative if it is adequately
  powered.
- **Multi-dataset:** every headline claim replicated on >= 2 independent bridge datasets (e.g.
  Emo-FilM AND NARPS for the relevant construct). A claim that holds on one dataset only is labeled
  T1 (suggestive), not T4.
- **Noise ceiling / SNR matching:** before any individuation claim, SNR-match conditions (the N=8
  lesson: amplitude differences masquerade as identity). Report skill as a fraction of the noise
  ceiling.
- **Effect sizes + CIs** everywhere; bootstrap CIs over subjects. No bare p-values.
- **Correction:** FDR across ROIs/constructs; pre-registered primary outcome to avoid garden-of-forking.

---

## 6. The synthesis / augmentation arm (validated separately, never assumed)

Synthesis is a hypothesis, not a convenience. It gets its own gate before it is allowed into the engine.

**6A. Virtual-brain army (Virtual Brains Phase 2).** Fit a normalizing flow / VAE over real individual
connectomes; sample synthetic connectomes; attach Hopf/Stuart-Landau dynamics near criticality;
condition the frozen encoder on each. Gate (all must pass): (1) topology match to real distribution
(small-worldness sigma 2-3, modularity Q 0.55-0.65, rich-club; two-sample n.s.); (2) discriminator
neural-Turing-test at chance (AUC ~ 0.5 real vs synthetic held-out); (3) complexity (LZc) in the
empirical human range. Kill: if synthetic is discriminable from real, it is an artifact generator;
report as such and do not augment with it.

**6B. LLM population priors.** Use LLM-simulated population behavior (purchase-intent literature: ~90%
test-retest at population level) as a cheap behavioral prior fused into M1/M2. Gate: the LLM prior must
improve population-level calibration WITHOUT degrading individual-level skill; if it pulls predictions
toward a population mean and hurts idiographic Delta, it is gated out.

**Augmentation test:** train M2 with vs without synthetic augmentation; augmentation is adopted only if
it improves held-out skill on REAL subjects. Synthetic data that only improves synthetic-test skill is
rejected.

---

## 7. Pre-registered gates and kill criteria (the decision rule)

The engine "works" only if ALL of the following hold on held-out, real data, replicated on >= 2 datasets:

0. **Leakage clear (Gate 0):** no bridge eval stimulus/subject is in the encoder's training corpus
   (§2D). *(else M2 is memorization)* — checked FIRST; terminal if it fails and cannot be excluded.
1. **Beats capacity-matched content:** skill(M2) > skill(M0+), CI excludes 0. *(the brain is a better
   stimulus basis than a random lift = POPULATION value, the unseen-audience product)*. NOTE from the
   apparatus validation (`neurobehavioral-mve/VALIDATION_REPORT.md`): Gate 1 passes even in a pure-NULL
   world with zero individuation, because the neural features carry the true stimulus->behavior basis.
   So Gate 1 is necessary but NOT sufficient for "neurobehavioral"; a Gate-1 pass must never be reported
   as "reads the individual."
2. **Beats subject-permutation:** skill(M2) > skill(M2-perm), CI excludes 0. *(INDIVIDUAL neural signal
   is real, not generic stimulus drive = the enrolled-user product)*. This is the load-bearing gate; in
   synthetic validation it fired iff individual signal existed AND the subject was enrolled (4/4 correct,
   zero false positives).
3. **Idiographic > nomothetic (enrolled regime):** per-person M2 beats the group-average model out-of-sample. *(anti-overfit; the Gate-5 crux; teeth only for enrolled subjects)*
4. **Encoder robustness:** the verdict does NOT flip between M2 and M2b. *(robustness check, not the independent control; M2-perm is the independent control)*
5. **Specificity:** the specificity gate passes for every reported construct. *(no reverse-inference confabulation)*
6. **Calibration:** conformal coverage >= nominal under cross-subject shift. *(honest abstention)*

Secondary descriptive (not a gate): Delta(M2 - M1) and skill(M1) - skill(M0).

**Master kill criterion:** if Gate 1 OR Gate 2 fails after adequately-powered, multi-seed, multi-dataset
analysis, the neural layer is not load-bearing (a positive Delta over M1 alone is then attributable to
feature capacity or generic stimulus drive, not the brain). We do not ship "neurobehavioral." We ship
"calibrated behavioral prediction" (M1 + abstention), drop the neuro claim to a credibility story, and
publish the negative. A null at Gate 1/2 counts as a kill ONLY if the power check (§5) confirms the MVE
could have detected the pre-registered MDE; an underpowered null is "inconclusive," not a kill.
The honest negative ("individual neurobehavioral variance is not recoverable at in-silico fMRI
resolution") is itself a clean, fundable outcome in the tradition of the topology and anesthesia nulls,
and the behavioral-engine pivot (M1 + abstention) ships either way.

---

## 8. Phasing (MVE first, scale second)

**Phase A: MVE (2-4 weeks, ~$ low).** One construct (valence/arousal) on ONE bridge dataset (Emo-FilM)
+ frozen TRIBE v2 + Sapient telemetry. Pre-flight: §2D leakage audit + §5 power/MDE check. Run the full
control ladder M0 / M0+ / M1 / M2 / M2-perm and Gates 1, 2, 3. The two numbers that decide scale-up:
skill(M2) - skill(M0+) and skill(M2) - skill(M2-perm). All inputs already exist; no new collection.

**Phase B: Scale (2-3 months).** Add NARPS, NSD, HCP, AOMIC; add the second encoder (Gate 4); add the
specificity + conformal layers (Gates 5, 6); multi-seed, multi-dataset, leave-one-subject-out at full
size. This is the publishable, pitch-ready result.

**Phase C: Synthesis (parallel, 2-3 months).** Build and gate the virtual-brain army and LLM priors
(Sec 6); adopt augmentation only if it improves real-subject skill.

**Phase D: Product wiring.** Fold the validated map + abstention into Mary/Scan as the live engine; the
provenance manifest and `Finding` contract become the API's confidence layer.

---

## 9. Compute and infra

- TRIBE v2 + second encoder inference on Modal (existing Sapient-1 pattern; weights in a volume).
- fMRIPrep + harmonization on a batch queue; cache derived features (the 695 MB N=8 matrices pattern).
- Public data: scripted download + checksum; raw deleted after feature extraction (the 16 GB -> 1.1 GB
  hygiene already practiced). One `DATA_MANIFEST.json` + one lockfile (upstream commit + seeds) for the
  whole run (closes G5).
- Estimated Phase-A compute: low hundreds of GPU-hours (inference-only on frozen encoders). Phase B/C
  scale with dataset count, still inference-dominated, no encoder training.

---

## 10. Deliverables

1. `DATA_MANIFEST.json` + lockfile (provenance for the whole run).
2. `bridge/` pipeline (ingest -> harmonize -> features -> models -> gates), reproducible from manifest.
3. A `Finding`-contract results table: every Mx, every dataset, every gate, with effect size + CI.
4. One figure: incremental-validity ladder (M0 -> M1 -> M2 -> M2b) with noise-ceiling-normalized skill.
5. The go/no-go decision memo against the Sec 7 kill criteria.
6. If green: a preprint ("a calibrated, abstaining, idiographic neurobehavioral prediction engine
   validated against measured behavior") and the revised pitch moat ("trust layer", not "5 data tiers").
   If red: the negative-result preprint + the de-risked behavioral-engine pivot.

---

### One-paragraph summary
We freeze Meta's open neural encoder as a prior, mint behavior from Sapient's own telemetry, and
calibrate the thin neural-to-behavior bridge on public datasets that pair both for the same subjects
(Emo-FilM, NARPS, NSD, HCP, AOMIC). We test one pre-registered question: does the neural prior beat
behavior-only and content-only, out-of-sample, idiographically, under a second-encoder control, a
specificity gate, and conformal abstention. An MVE on Emo-FilM decides in weeks whether to scale.
Synthesis (a topology-gated virtual-brain army and LLM population priors) is built in parallel and
admitted only if it improves skill on real subjects. Green means "neurobehavioral" is the moat; red
means we ship a calibrated behavioral engine and publish the negative. Every outcome is a decision.
</content>
</invoke>
