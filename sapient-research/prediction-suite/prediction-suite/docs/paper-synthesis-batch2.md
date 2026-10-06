# Sapient Program: Extended Synthesis (Batch 2)
## 11 High-Relevance Papers + Integration with Batch 1

**Status:** Second wave of papers significantly deepens personalized brain encoding, direct brain-text decoding, and neurocognitive-aligned AI architecture.

---

## The 11 New High-Relevance Papers

### **Core Brain Encoding & Latent Spaces**

#### **2601.13297 — Multifaceted Neural Representation of Words in Naturalistic Language**
- **What:** Identifies 8 interpretable latent dimensions (phonological, syntactic, semantic, grammatical, emotional, abstractness, morphological, predictability) spanning word encoding across cortical systems.
- **Impact on Sapient:** Directly solves the "what are the latent dimensions?" problem for language in Mary. Instead of hand-picking KPIs, use these 8 dimensions as a neuroscientifically-grounded basis for semantic read-outs.
- **Integration:** Replaces language stream of Mary with dimension-aligned decoder. For every word/phrase in video transcript, predict each of the 8 dimensions from cortical response.

#### **2601.02010 — CATS Net: Neural Network for Human Concept Formation, Understanding, Communication**
- **What:** Learned concept spaces emergently align with neurocognitive semantic models and measured brain-response structures. Concepts are self-organized in brain-like way.
- **Impact on Sapient:** Validate that Mary's latent concept space matches human brain's concept organization. If Mary learns a "luxury" concept, does it cluster similarly to how brains encode luxury-related words/imagery?
- **Integration:** Add CATS Net as an auxiliary validator in Scan. Measure concept-space alignment between Mary's learned representations and ground-truth brain concepts.

#### **2508.11672 — BCNE: Unsupervised Manifold Learning from Dynamic Brain Data**
- **What:** Captures brain-state trajectories by learning temporospatial correlations in dynamic fMRI/EEG. Discovers latent manifolds of brain states.
- **Impact on Sapient:** Mary is currently static (stimulus → regional response). BCNE reveals *dynamics* — how does the brain's response trajectory unfold over 10s as a brand ad plays? Phase transitions? Attractor states?
- **Integration:** Wire BCNE into Scan as trajectory modeling. For each subject, learn their personalized brain-state manifold during brand exposure. Few-shot prediction: new brand → predict trajectory on their manifold.

#### **2507.02908 — Hyperbolic Brain Graphs for Neurocognitive Decline Analysis**
- **What:** HKGF encodes hierarchical brain functional connectivity in hyperbolic space. Captures multiscale organization (local circuits → global networks).
- **Impact on Sapient:** Mary treats brain regions atomically. HKGF shows hierarchy matters: visual cortex → V2 → ventral stream → semantic cortex. Personalized hierarchical brain graphs enable scale-aware predictions.
- **Integration:** Replace Mary's flat regional model with subject-specific hyperbolic brain graph. Each subject's cortical hierarchy varies slightly; HKGF captures this.

---

### **Direct Brain-to-Content Decoding**

#### **2509.07202 — Neurocognitive Modeling for Text Generation: EEG-to-Text via Gemma 2B + RNN**
- **What:** Combines EEG encoder + Gemma 2B LLM to directly generate text from brain signals. Shows brain-to-language decoding at scale.
- **Impact on Sapient:** This is the inverse of Mary. Mary: content → brain. This paper: brain → content. Together, they form a **bidirectional brain-AI loop**.
- **Integration:** Future Cortex-of-Anyone feature: given a subject's Mary fMRI + this paper's EEG-to-text decoder, predict what they'd *say* about a brand (their verbalized thought) + their neural response simultaneously. Creates a personalized "thought + brain" readout.

#### **2507.07157 — Interpretable EEG-to-Image Generation with Semantic Prompts**
- **What:** Text-mediated framework: EEG → semantic captions → images. Achieves interpretable visual decoding from brain signals.
- **Impact on Sapient:** Inverse of Mary's video→fMRI pipeline. Can now close the loop: subject sees brand video → Mary predicts fMRI → infer what *visual imagery* they're conjuring → reconstruct/describe it.
- **Integration:** Wire into Scan as a "what are they visualizing?" read-out. Subject exposure to brand ad → Mary fMRI → this paper's method → semantic description of their imagined response. Adds interpretability to otherwise opaque regional predictions.

---

### **Brain-Inspired AI Architectures**

#### **2512.02280 — Bridging the Gap: Cognitive Autonomy in Artificial Intelligence**
- **What:** Analyzes core AI deficits (no intrinsic self-monitoring, no meta-cognitive awareness) and proposes neurocognitive-inspired fixes.
- **Impact on Sapient:** Verdict currently uses PAGRL (4-layer governance). This paper provides theoretical grounding: embed *intrinsic* cognitive monitoring into Verdict's decision-making (self-checking, doubt, escalation under uncertainty).
- **Integration:** PAGRL layer 3 (agent-level rules) now includes meta-cognitive gates: "Am I confident in this recommendation? Do I need to self-monitor before acting?"

#### **2510.13826 — Towards Neurocognitive-Inspired Intelligence: Structural Mimicry to Functional Cognition**
- **What:** Modular brain-inspired architecture emphasizing integration, embodiment, adaptive control grounded in neurocognition.
- **Impact on Sapient:** Provides the *system design* blueprint for Cortex-of-Anyone. Mary (perception) + CraniMem (memory) + PAGRL (executive) + heterogeneity (personalization) are modules. This paper shows how to integrate them into a single adaptive cognitive system.
- **Integration:** Use this paper's design principles to architect Cortex-of-Anyone as a modular, embodied, neurocognitive system (not just a bag of classifiers).

#### **2503.11299 — BriLLM: Brain-Inspired Large Language Model**
- **What:** LLM implementing signal-flow learning with static semantic mapping + dynamic signal propagation simulating electrophysiological brain dynamics.
- **Impact on Sapient:** Replace Verdicts generic LLM backbone with BriLLM. Verdict recommendations now flow through a brain-like signal pipeline, not transformer attention. Adds interpretability (signal flow = neural dynamics) + efficiency.
- **Integration:** Pilot BriLLM in Verdict recommendations. Does brain-inspired signal flow produce more human-aligned recommendations than standard LLM?

---

### **Brain-AI Alignment & Human-AI Cognition**

#### **2508.14869 — The Prompting Brain: Neurocognitive Markers of Expertise in Guiding LLMs**
- **What:** fMRI study reveals neural signatures (temporal-frontal connectivity) of prompt engineering expertise. Human brain shows *distinct neural states* when good vs. bad at directing AI.
- **Impact on Sapient:** Humans interact with Verdict recommendations. This paper shows their *brain state* reflects whether they're using Verdict well. Could measure expertise in using Sapient products via brain imaging.
- **Integration:** Scan can measure "Verdict-interaction expertise": acquire fMRI while users draft marketing campaigns using Verdict recommendations. Predict their neural state → infer whether they're using Verdict optimally. Personalized training recommendations.

#### **2508.10057 — Large Language Models Show Signs of Alignment with Human Neurocognition During Abstract Reasoning**
- **What:** LLMs form pattern representations clustering similarly to brain activity during abstract reasoning tasks.
- **Impact on Sapient:** Validates brain-AI alignment. Mary's learned representations should cluster like brain activity. This paper provides a benchmark: if Mary's latents cluster differently from brain activity, something's wrong.
- **Integration:** Diagnostic: train Mary; extract latent representations for test stimuli; compare clustering to fMRI-derived brain activity clustering. If aligned, Mary is learning brain-like abstractions. If not, retrain.

---

## Integration Matrix: Batch 1 + Batch 2

| Component | Batch 1 Paper | Batch 2 Paper | Integrated Function |
|-----------|---------------|---------------|-------------------|
| **Core Encoder** | TRIBE v2 (2605.04326) | — | Tri-modal baseline |
| **Video Fine-tuning** | CoTZero (2602.08339) | — | Hierarchical compositional reasoning |
| **Language Semantics** | — | 2601.13297 (8 dimensions) | Interpretable language read-outs |
| **Concept Alignment** | — | CATS Net (2601.02010) | Validate concept space = brain concepts |
| **Brain Dynamics** | — | BCNE (2508.11672) | Trajectory modeling (not just static responses) |
| **Personalized Hierarchy** | Dalla Porta (receptor maps) | HKGF (2507.02908) | Hierarchical brain graphs per subject |
| **Memory Consolidation** | CraniMem (2603.15642) | — | Few-shot episodic learning |
| **Hippocampal Logic** | HipDETR (2606.27831) | — | Memory-guided read-outs |
| **Brain-to-Text Inverse** | — | EEG-to-Text (2509.07202) | Decode subject's verbalized thought |
| **Brain-to-Image Inverse** | — | EEG-to-Image (2507.07157) | Decode subject's visual imagery |
| **Executive Control** | PAGRL (2604.25684) | Cognitive Autonomy (2512.02280) | Meta-cognitive self-monitoring gates |
| **AI Agent Architecture** | — | BriLLM (2503.11299) + Neurocog-Inspired (2510.13826) | Brain-like signal-flow Verdict engine |
| **Brain-AI Alignment Validation** | — | LLM Alignment (2508.10057) | Diagnostic: latent clustering vs. fMRI |
| **Human Expertise in Using AI** | — | Prompting Brain (2508.14869) | Measure user expertise via fMRI (Scan feature) |

---

## Revised Roadmap: Batch 2 Accelerators

### **Phase 1: Validation (2 mo) — NOW INCLUDES ALIGNMENT CHECKS**
1. **FIX 1–3:** Video pipeline + construct-valid latents (Batch 1)
2. **NEW:** Brain-AI alignment validation (2508.10057)
   - Train Mary; extract latents for test stimuli
   - Compare latent clustering to fMRI-derived brain activity
   - Pass gate: clustering within threshold
3. **NEW:** 8-dimensional language semantics (2601.13297)
   - Replace audio stream with dimension-aligned decoder
   - For each word in transcript, predict all 8 dimensions from Mary's latent space

### **Phase 2: Personalization (4 mo) — NOW INCLUDES CONCEPT + TRAJECTORY**
1. **FIX 4:** Subject-specific heterogeneity (Batch 1) + hyperbolic brain graphs (2507.02908)
   - Build subject's personalized cortical hierarchy (not just region-level modulation)
2. **NEW:** Brain-state trajectory modeling (BCNE, 2508.11672)
   - Move beyond static response prediction
   - Model how subject's brain response *evolves* during brand exposure
   - Capture phase transitions, attractor states
3. **NEW:** Concept-space alignment (CATS Net, 2601.02010)
   - Verify Mary's learned "brand appeal" concept clusters like brain's semantic clusters
   - Few-shot validation: new brand → predict concept position → verify against fMRI

### **Phase 3: Cognitive Security + Human Expertise (6 mo) — BIFURCATES INTO TWO TRACKS**

**Track A: Defense (WARDEN)**
- Misinformation resilience (Batch 1)
- Meta-cognitive monitoring (Cognitive Autonomy, 2512.02280)
- Self-checking gates in PAGRL

**Track B: User Enablement (NEW)**
- Measure user expertise in using Verdict via their brain state during interaction (2508.14869)
- Brain-to-text decoder predicts what users *intend* to communicate via marketing campaigns (2509.07202)
- Personalized training: if user's Verdict-interaction brain state is suboptimal, suggest workflow changes

### **Phase 4: Multimodal + Inverse Decoding (12+ mo) — OPENS BI-DIRECTIONAL LOOP**
1. **Forward (Mary):** Brand video → fMRI prediction
2. **Inverse (NEW):**
   - Brain-to-content: fMRI → infer visual imagery the subject conjures (2507.07157 EEG-to-image)
   - Brain-to-text: fMRI/EEG → verbalize their thought response (2509.07202 EEG-to-text)
3. **Cortex-of-Anyone closes loop:**
   - Enroll → personalize → predict thought + brain + behavior simultaneously
   - Test new brand → predict subject's neural response + their verbalized reaction + their imagery
   - Validate prediction against real response

---

## Critical New Capabilities Unlocked by Batch 2

### **1. Interpretability Through Inverse Decoding**
Mary currently predicts regional fMRI (opaque). Now:
- Extract subject's *visualized* content from their fMRI (2507.07157)
- Extract subject's *verbalized* thought from their fMRI (2509.07202)
- Result: "Subject saw this brand ad → Mary predicted fMRI → decoded to 'They're imagining the product in their home' + 'I like this'" — INTERPRETABLE & ACTIONABLE

### **2. Dynamics, Not Just Snapshots**
Mary currently: frame at time T → regional response. Now:
- Use BCNE to model brain-state trajectory as brand ad plays (0–30s)
- Capture when emotional response peaks, when attention wanes, when memorability sets in
- Result: "Ad achieves emotional peak at :15, attention drops at :25" — TEMPORAL OPTIMIZATION

### **3. Concept-Level Reasoning**
Mary currently: region A activates. Now:
- Use CATS Net to map activation to learned concepts (brand appeal, trustworthiness, deceptiveness)
- Validate concepts cluster like human brains
- Result: Verdict can reason about *concepts*, not just regions — "This ad triggers high deceptiveness-concept activation in vulnerable subjects" — COGNITIVE SECURITY

### **4. Hierarchical Personalization**
Mary currently: global receptor maps. Now:
- Use hyperbolic brain graphs to personalize cortical hierarchy (2507.02908)
- Each subject has unique V1→V2→V4→IT pathway differently weighted
- Result: "Subject A's visual-to-semantic pathway is 3× stronger than population average" — FINE-GRAINED PHENOTYPING

### **5. Brain-Like AI Agents**
Verdict currently: standard LLM recommendations. Now:
- Use BriLLM signal-flow backbone (2503.11299)
- Use neurocognitive-inspired modular architecture (2510.13826)
- Use meta-cognitive self-monitoring (2512.02280)
- Result: Verdict thinks like a brain thinks — more interpretable, more human-aligned, more robust

---

## Code Targets: Batch 2 Implementation

**Repository structure (hypothetical, add to sapient-models):**
```
sapient-models/
├── encoders/
│   ├── mary_core.py          # TRIBE v2 baseline (Batch 1)
│   ├── language_8d.py        # 8-dimensional semantic decoder (2601.13297)
│   ├── bcne_trajectory.py    # Brain-state trajectory modeling (2508.11672)
│   └── hyperbolic_graphs.py  # Personalized cortical hierarchy (2507.02908)
├── inverse/
│   ├── eeg_to_text.py        # Brain-to-thought decoding (2509.07202)
│   ├── eeg_to_image.py       # Brain-to-imagery decoding (2507.07157)
│   └── semantic_alignment.py # CATS Net validator (2601.02010)
├── validation/
│   ├── brain_ai_alignment.py # Latent clustering diagnostic (2508.10057)
│   └── concept_validator.py  # Concept-space alignment check
├── agents/
│   ├── brillm_backbone.py    # Brain-inspired LLM for Verdict (2503.11299)
│   ├── metacog_gates.py      # Self-monitoring in PAGRL (2512.02280)
│   └── neuroarch.py          # Modular neurocognitive architecture (2510.13826)
└── human_ai/
    ├── user_expertise_fmri.py # Measure prompt-engineering brain state (2508.14869)
    └── collaborative_cognition.py  # Human-AI brain alignment during interaction
```

**Phase 1 PR checklist (Batch 2 adds):**
- [ ] Wire 8-dimensional language semantics (2601.13297)
- [ ] Brain-AI alignment diagnostic (2508.10057)
- [ ] Concept-space validator (CATS Net, 2601.02010)

**Phase 2 PR checklist (Batch 2 adds):**
- [ ] BCNE trajectory modeling (2508.11672)
- [ ] Hyperbolic brain graphs (2507.02908)

**Phase 4 PR checklist (Batch 2 adds):**
- [ ] EEG-to-text decoder (2509.07202)
- [ ] EEG-to-image decoder (2507.07157)
- [ ] BriLLM Verdict backbone (2503.11299)

---

## Open Questions for Sapient Team

1. **Batch 2 Priority:** Which high-impact capability would unlock the most value first?
   - Interpretable inverse decoding (see subject's thought)?
   - Trajectory dynamics (temporal optimization)?
   - Brain-like Verdict agent (conceptual reasoning)?

2. **Data Requirements:** Do you have:
   - Longitudinal fMRI for the same subjects across multiple brand exposures (needed for trajectory modeling)?
   - Concurrent EEG + fMRI during brand viewing (needed for inverse decoding validation)?

3. **TRIBE v2 Partnership Status:** Has d'Ascoli's team responded? If yes, should we:
   - License checkpoint + fine-tune with Sapient's brain + brand data?
   - Or build Mary independently but benchmark against TRIBE v2?

4. **Defense Use Case:** Is WARDEN cognitive-security product ready for Batch 2 misinformation features, or should it wait?

5. **Human-AI Loop:** Is measuring user expertise via their brain state (2508.14869) in scope, or too speculative for MVP?

---

## Summary: Batch 1 + Batch 2 = Complete Brain-AI System

**Batch 1 (10 papers):**
- Core architecture (TRIBE v2, B[FM]², Dalla Porta)
- Governance + memory (PAGRL, CraniMem, HipDETR)
- Security + grounding (CoTZero, Hallucination study, NeuroCognition)

**Batch 2 (11 papers):**
- Latent semantics + concepts (8D language, CATS Net)
- Dynamics + hierarchy (BCNE trajectories, hyperbolic graphs)
- Inverse decoding (EEG→text, EEG→image)
- Brain-like AI agents (BriLLM, neurocog architecture)
- Human-AI brain alignment (prompt-engineering expertise, LLM alignment)

**Together:** Mary (stimulus → brain) + Inverse decoders (brain → thought/imagery) + Concept reasoning + Trajectory dynamics + Personalized hierarchy + Brain-like Verdict agent + User expertise measurement = **Cortex-of-Anyone flagship: a truly personalized, bidirectional, neurobiologically-grounded brain-AI system.**

---

**Generated:** 2026-07-11 | **Batch 2 Status:** Ready for prioritization
