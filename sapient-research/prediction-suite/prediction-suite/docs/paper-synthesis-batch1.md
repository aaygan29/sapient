# Sapient Company Program: Paper Synthesis & Evolution Roadmap

## Program Context
**Sapient-1:** In-silico fMRI encoder (Mary/Scan/Verdict SaaS) — tri-modal (video/audio/text) → regional brain responses for B2B brand intelligence.

**Current State:** Mary encodes audio+text only; video streams masked. Visual cortex unresponsive. Regional read-outs confounded with global signal. Three validation gates needed before shipping.

**Goal:** Transform Mary into a **personalized, construct-valid, neurobiologically-grounded encoder** for idiosyncratic brand psychology.

---

## The 10 Papers: Three Integration Vectors

### **Vector 1: CORE ARCHITECTURE SCALING (TRIBE v2 + B[FM]²)**

#### **2605.04326 — TRIBE v2: Foundation Model of Vision, Audition, Language for In-Silico Neuroscience**
**Authors:** d'Ascoli, Rapin, Benchetrit, Brooks, et al.
- **What it does:** Tri-modal (video/audio/language) foundation model trained on 1,000+ hours of fMRI across 720 subjects
- **Key achievement:** Predicts held-out fMRI from novel stimuli, tasks, subjects; recovers canonical neuroscience findings in-silico
- **Direct relevance to Sapient:**
  - **This IS your encoder's north star architecture.** TRIBE v2's exact training objective (predict brain from tri-modal input) matches Mary's.
  - TRIBE v2 uses 720 subjects; Mary's checkpoint presumably smaller (Sapient proprietary). Scaling from 720→∞ subjects is the multi-subject generalization strategy.
  - TRIBE v2 extracts **interpretable latent features** revealing fine-grained multisensory integration. This solves your read-out problem (FIX 3): instead of regional KPIs, use TRIBE's latent subspace to define construct-aligned dimensions.

**Immediate implementation:**
1. Architecture parity check: Does Mary's encoder structure match TRIBE v2's tri-modal fusion strategy? (Likely yes, but audit for differences in visual stream ordering, attention gating, multimodal bottleneck.)
2. **Replace FIX 3's hand-picked KPI read-outs** with a meta-analytically-grounded latent projection learned from TRIBE v2's interpretable features. Withold scores when latents correlate >0.7 with audio confounds.
3. **Collaborative opportunity:** Contact d'Ascoli + team for TRIBE v2 checkpoint access. Mary could be a Sapient-proprietary fine-tune of TRIBE v2 on brand stimuli + individual-subject idiosyncratic maps (Mary's added value).

---

#### **2606.20812 — B[FM]²: Brain Foundation Model via Flow Matching**
**Authors:** Hwang, Zhang, Dai, Kontras, Vanmarcke, De Vos, Fiete, Liang
- **What it does:** EEG foundation model (flow matching, no patches/tokenization) that generates synthetic EEG indistinguishable from real data; 30× more efficient than prior EEG models.
- **Key achievement:** Recovers brain signal fine structure; scales to clinical/BCI tasks.
- **Relevance to Sapient:**
  - **Complementary modality:** Mary predicts fMRI (slow, regional). B[FM]² generates EEG (fast, high temporal resolution). Together = full spatiotemporal brain readout.
  - **Synthetic data generation:** B[FM]²'s ability to generate realistic synthetic EEG opens a path to **"personalized brain augmentation"** — given a subject's Mary fMRI readout, synthesize their EEG under novel stimuli without scanning. Could accelerate Sapient's Cortex-of-Anyone enrollment (fewer scans per subject).
  - **Efficiency gain:** 30× compression in pretraining budget suggests Mary could optimize similarly (flow matching instead of transformer masking?).

**Immediate implementation:**
1. Architecture audit: Could Mary adopt flow matching for the fMRI → latent → regional read-out pathway? (Speculative; fMRI has different temporal structure than EEG.)
2. **Post-Mary roadmap:** Wire B[FM]² as an EEG auxiliary encoder in the Cortex-of-Anyone stack. Enroll via brief fMRI + EEG; Mary predicts fMRI from brand stimuli; B[FM]² predicts EEG from Mary's latents.

---

### **Vector 2: PERSONALIZED BRAIN DYNAMICS (Dalla Porta PNAS + Hippocampus-DETR)**

#### **PNAS 2026 — Spatially Structured Heterogeneity Shapes Large-Scale Cortical Dynamics**
**Authors:** Dalla Porta, Fousek, Destexhe, Sanchez-Vives
- **What it does:** Large-scale cortical model constrained by muscarinic receptor heterogeneity (M1/M2 maps from transcriptomics + PET). Shows regional molecular diversity enhances synchronization, information flow, state transitions.
- **Key finding:** Biologically-aligned heterogeneity is an *organizing principle* linking molecules to macroscale function.
- **Relevance to Sapient:**
  - **The missing layer in Mary:** Mary maps stimuli → regional fMRI but treats each subject's brain as a black-box fixed point. Dalla Porta shows regional heterogeneity (cholinergic maps, receptor density) shapes the *dynamics* of response.
  - **Personalization bottleneck:** Two subjects see the same ad; their fMRI differs. Why? Dalla Porta: molecular heterogeneity. Solution = subject-specific receptor maps as a **personalization layer**.
  - **Data source:** Allen Brain Atlas (transcriptomic + immunostaining for M1/M2) + ADNI/OASIS PET (in-vivo receptor binding). Can be mapped to your subjects' structural MRI (e.g., T1 cortical thickness → receptor density proxy).

**Immediate implementation:**
1. **FIX 4 (post-FIX 3):** Add a personalization encoder to Mary. For each subject:
   - Extract/infer their regional muscarinic heterogeneity from fMRI + T1 + optional PET.
   - Use Dalla Porta's AdEx-MF framework to compute region-specific excitability modulation factors.
   - Multiply these factors into Mary's regional read-outs (gate-level personalization).
2. **Data acquisition:** Partner with radiology groups collecting T1 + 7T fMRI + optional [11C]UCB-J (receptor PET). Build a small (N=50) idiosyncratic cortical heterogeneity atlas.

---

#### **2606.27831 — Hippocampus-DETR: Explicit Memory Object Detection Framework**
**Authors:** Shi, Ma, Xu, Yang, Liang
- **What it does:** Computer vision model with explicit memory mimicking hippocampal substructure (DG → CA3 → CA1 → subiculum). Achieves pattern separation, completion, importance filtering.
- **Relevance to Sapient:**
  - **Mary reads "what" is on screen; it doesn't read "who remembers."** Brand psychology depends on autobiographical resonance — does this ad trigger a memory? Hippocampus-DETR's pattern-separation + completion could augment Mary.
  - **Architecture integration:** Wire Hippocampus-DETR's memory encoder as an auxiliary stream into Mary. Video → (Mary fMRI predictor) + (HipDETR memory router); memory-weighted read-outs capture "this ad is *personally meaningful*."
  - **Clinical application:** Sapient's defense-use case (WARDEN program, cognitive security): subjects with amnestic MCI show reduced HipDETR pattern-completion; Mary + HipDETR together predict "vulnerability to deception via forgetting."

**Immediate implementation:**
1. Build a Mary auxiliary stream: video → HipDETR → semantic memory flags (is this scene novel/familiar/emotionally weighted?).
2. Train on brand-and-autobiographical-video dataset (user-recorded clips + brand ads; fMRI of both).
3. Read-out gate: withold "emotional value" construct if HipDETR memory-score contradicts it (e.g., claim emotional response to brand that HipDETR flags as utterly novel/unmemorable).

---

### **Vector 3: NEUROCOGNITIVE GROUNDING + EVALUATION (Agents, Cognitive Tests, Reasoning)**

#### **2604.25684 — Think Before You Act: Neurocognitive Governance Model for Autonomous AI Agents**
**Authors:** Bandara, Gore, Gunaratna, Rajapakse, et al.
- **What it does:** Maps human executive function (inhibitory control, deliberation, hierarchical rule-checking) to LLM agent reasoning. Pre-Action Governance Reasoning Loop (PAGRL) = 4-layer governance rule set (global → domain → agent → situational).
- **Key achievement:** 95% compliance accuracy, zero false escalations in production supply-chain workflow.
- **Relevance to Sapient:**
  - **Not about Mary, but about Verdict.** Verdict is the SaaS interface that recommends actions to marketing teams ("increase spend on this segment"; "shift messaging to this frame"). Today's decision-making is likely heuristic.
  - **Neurocognitive governance:** Embed executive-function-style reasoning into Verdict. Before recommending a marketing action, consult a 4-layer rule set:
    - **Global:** Brand guidelines, legal compliance.
    - **Domain:** Industry norms (e.g., healthcare marketing forbids certain psychological angles).
    - **Agent (Mary):** "Does this recommendation align with the predicted brain responses?"
    - **Situational:** Real-time external context (market sentiment, regulatory flags).
  - **Transparency:** PAGRL's interpretability (each rule layer logged) satisfies QUALIA_CONTRACTS §3 honesty boundary.

**Immediate implementation:**
1. Implement PAGRL in Verdict's recommendation engine.
2. Mary's read-outs feed layer 3; external data (news, sentiment, regulatory) feed layer 4.
3. Log every decision path; expose to customers (regulatory + brand transparency requirement).

---

#### **2603.02540 — A Neuropsychologically Grounded Evaluation of LLM Cognitive Abilities**
**Authors:** Haznitrama, Ardi, Oh
- **What it does:** NeuroCognition benchmark based on 3 neuropsych tests (Raven's matrices, spatial working memory, Wisconsin Card Sort). Shows LLMs exhibit a general factor but fail on foundational cognitive abilities.
- **Relevance to Sapient:**
  - **Mary as a cognitive model:** Is Mary capturing genuine cognition, or surface pattern correlation? NeuroCognition could audit this.
  - **Immediate use:** Benchmark Sapient's Scan product (cognition assessment tool for brand-vulnerability screening). Does Scan's cognitive model align with human neuropsych? Use NeuroCognition as a validation suite.
  - **Evolution:** Extend NeuroCognition to measure *brand-specific* cognitive biases (susceptibility to emotional appeals, resistance to contradictory evidence, etc.).

**Immediate implementation:**
1. Adapt the 3 NeuroCognition tests to brand-psychology domain (e.g., "Raven's for marketing: visual-semantic pattern-finding under brand noise").
2. Collect fMRI + NeuroCognition-adapted task data in Mary's cohort.
3. Verify Mary's regional read-outs align with the neuropsych constructs (e.g., high "abstract reasoning" read-out correlates with Raven's performance).

---

#### **2603.15642 — CraniMem: Cranial-Inspired Gated and Bounded Memory for Agentic Systems**
**Authors:** Mody, Panchal, Kar, Bhowmick, Karani
- **What it does:** Agent memory design (episodic buffer + long-term knowledge graph) with goal-conditioned gating, utility tagging, scheduled consolidation. Mimics human memory consolidation.
- **Relevance to Sapient:**
  - **Mary as an agent:** Mary doesn't remember previous brand interactions with a subject. Each prediction is stateless.
  - **Personalization scaling:** CraniMem's consolidation loop (replay high-utility memories, prune low-utility) could let Mary learn idiosyncratic brand preferences over time without catastrophic forgetting.
  - **Integration:** Wire CraniMem into Scan/Verdict as a subject-specific episodic memory of past brand exposures, outcomes, and neural responses. Over repeated interactions, Scan/Verdict improves at predicting subject's reaction to *new* brands (few-shot transfer).

**Immediate implementation:**
1. Implement CraniMem as a subject-specific module in Scan.
2. Each brand interaction (video shown → fMRI predicted → outcome tracked) is an episodic trace.
3. High-utility traces (surprising predictions, strong outcomes) replay into a knowledge graph of "this subject's brand-response phenotype."
4. Few-shot prediction: for a new brand, query the knowledge graph for compositionally similar ads; use CraniMem's retrieval to warm-start Mary.

---

#### **2602.08339 — CoTZero: Annotation-Free Human-Like Vision Reasoning via Hierarchical Synthetic CoT**
**Authors:** Du, Niu, Shen, Xu
- **What it does:** Vision model trained via dual-stage data synthesis (bottom-up atomic primitives → top-down compositional structure) + Cognitively Coherent Verifiable Rewards (CCVR). Achieves human-like compositional and verifiable reasoning.
- **Relevance to Sapient:**
  - **Mary's visual reasoning:** Mary's video encoder (slowfast, qwen_vl, got_ocr) extracts low-level features. CoTZero shows compositional reasoning ("person + brand logo + emotional expression → trust signal") requires hierarchical training.
  - **Interpretability:** CoTZero's hierarchical reasoning is explainable (global structure → local details). This aligns with Sapient's need for interpretable read-outs (FIX 3b: the KPIs must reflect *interpretable* latent dimensions, not opaque network activations).

**Immediate implementation:**
1. Fine-tune Mary's video encoders using CoTZero's dual-stage synthetic data approach.
2. Prioritize training on compositional brand-psychology stimuli (e.g., brand logo + demographic face + emotional context).
3. Use CCVR to reward predictions that align with known neuroscience (e.g., visual-cortex response correlates with Scen clarity; amygdala response correlates with threat-value).

---

### **Vector 4: HUMAN COGNITION GROUNDING (Hallucination Detection, Interpersonal Synchrony)**

#### **2605.16953 — How do Humans Process AI-generated Hallucination Contents: A Neuroimaging Study**
**Authors:** Zhu, Zhong, Ye, Du, Zhou, Ai, Liu
- **What it does:** EEG study (27 subjects) of responses to AI-generated hallucinations. Shows distinct neural patterns (semantic integration, memory retrieval, cognitive load) for hallucinated vs. real content. Misjudged hallucinations fail to trigger standard fact-verification neural pathway.
- **Relevance to Sapient:**
  - **Brand misinformation as hallucination:** Competitors deploy AI-generated fake testimonials, false claims, deepfakes. Sapient's defense product could detect "neural signatures of deception vulnerability" — when a subject's brain *fails* to trigger fact-verification.
  - **Implementation:** Train Mary on brands where subjects unknowingly see hallucinated content (fake testimonials, false claims). Mary learns the neural signature of "belief without verification." Predict this signature for new subjects → identify manipulation-vulnerable phenotypes.
  - **WARDEN program alignment:** This neural-signature approach is exactly the "cognitive security reasoning instrument" WARDEN aims for.

**Immediate implementation:**
1. Create a brand-misinformation stimulus set (real + AI-generated-false brand claims; fMRI of viewing/judging).
2. Train auxiliary Mary encoder on ERP components from this study (semantic integration, memory retrieval).
3. Read-out: "Fact-verification resilience" score. High = subject's brain triggers skepticism to false claims. Low = vulnerable to misinformation.

---

#### **2606.03700 — Who Is in Mind Matters: Attachment Representations in Early Childhood Synchronize Child-Adult Interacting Brains**
**Authors:** Su, Xu, Wu, Wang, Li, Yang, Liu, Liu, Tong, Zhang, Guo, Jiang
- **What it does:** Child-adult EEG synchrony experiment. Shows partner-belief (child believes they're talking to mother vs. stranger) modulates interbrain synchrony (P4 temporoparietal junction, attachment-designated). Synchrony strength correlates with attachment security.
- **Relevance to Sapient:**
  - **Brand-consumer "attachment."** Consumers have "attachment" to brands (loyalty, emotional investment). Sapient could measure this via neural synchrony (brand-video → subject's Mary fMRI + simultaneous peer's EEG during joint viewing).
  - **Market research application:** B2B insight: does this brand campaign enhance synchrony between consumer and influencer? High synchrony = persuasive, emotionally resonant.
  - **Interpersonal dynamics:** Extend Mary to joint decision-making scenarios (e.g., family watching an ad together). Predict not individual fMRI, but cross-subject synchrony patterns.

**Immediate implementation:**
1. (Speculative, lower priority) Develop Mary-EEG joint encoding module for multi-subject scenarios.
2. Near-term: Use this paper's EEG→synchrony methods to validate Sapient's in-person brand testing (collect paired subject EEG during co-viewing of brand content).

---

## Summary Matrix: Paper → Sapient Action

| Paper | Core Insight | Immediate Action | Roadmap |
|-------|-------------|------------------|---------|
| **TRIBE v2** | Tri-modal fMRI foundation model scales to 720+ subjects | Architecture parity; adopt interpretable latent read-outs (FIX 3 replacement) | Collaborative checkpoint; Mary as proprietary fine-tune |
| **B[FM]²** | Flow matching + synthetic EEG generation | Optional flow-matching architecture audit | Post-Mary: EEG auxiliary stream + subject-specific augmentation |
| **Dalla Porta PNAS** | Molecular heterogeneity organizes macroscale dynamics | **FIX 4:** Subject-specific receptor heterogeneity personalization layer | Recruit T1+PET+7T fMRI cohort for idiosyncratic maps |
| **HipDETR** | Explicit memory module for pattern completion | Auxiliary memory router into Mary; memory-weighted read-outs | Pattern-separation audit for amnesia/manipulation vulnerability |
| **Think Before You Act** | Neurocognitive governance (PAGRL) for agents | Embed PAGRL into Verdict recommendation engine | Rule-layer transparency + customer auditability |
| **NeuroCognition** | Neuropsych benchmark for cognitive grounding | Validate Scan's cognitive model against Raven/WCSt analogs | Brand-cognitive-bias extension |
| **CraniMem** | Goal-conditioned agent memory + consolidation | Subject-specific episodic memory in Scan; few-shot transfer | Knowledge-graph learning over repeated brand interactions |
| **CoTZero** | Hierarchical compositional reasoning + CCVR training | Fine-tune Mary video encoders on compositional stimuli | Interpretability alignment (latent dimensions = neuroscience constructs) |
| **Hallucination Study** | EEG signatures of deception vulnerability | Train auxiliary Mary on misinformation; fact-verification score | WARDEN defense product (cognitive-security readout) |
| **Attachment Synchrony** | Interbrain synchrony as attachment measure | Validate Sapient's in-person brand testing | Multi-subject Mary-EEG for joint decision-making |

---

## Phased Integration Roadmap

### **Phase 1: Validation + Core Fixes (Next 2 months)**
1. **FIX 1+2+3:** Complete Mary's video pipeline (validate encoders, wire extractors, replace KPIs with TRIBE-style latents).
2. **Architecture parity:** Audit Mary vs. TRIBE v2; align on fusion strategy.
3. **Scope:** No new features; de-risk the existing product.

### **Phase 2: Personalization (Months 3–6)**
4. **FIX 4:** Integrate Dalla Porta's heterogeneity model. Recruit N=50 T1+7T+fMRI cohort; map muscarinic heterogeneity.
5. **CraniMem integration:** Add episodic memory layer to Scan.
6. **Scope:** Sapient now personalizes to individual subjects (not just population average).

### **Phase 3: Cognitive Security + Interpersonal Dynamics (Months 6–12)**
7. **Hallucination study:** Train auxiliary Mary on misinformation; add "fact-verification resilience" read-out to Verdict.
8. **CoTZero fine-tuning:** Compositional video encoding; hierarchical reasoning.
9. **WARDEN alignment:** Fact-verification + executive-function (PAGRL) governance.
10. **Scope:** Sapient becomes a cognitive-security instrument (defense use case).

### **Phase 4: Multi-Subject + EEG Augmentation (Months 12+)**
11. **B[FM]²:** EEG auxiliary stream for spatiotemporal grounding.
12. **Joint decision-making:** Multi-subject Mary-EEG for interpersonal dynamics.
13. **Scope:** Cortex-of-Anyone flagship (personalized digital brains for enrollment, experimentation, prediction).

---

## Code Pointers & Next Steps

**Sapient repo structure:**
- `qualia/core/mary_engine.py` — ACTIVE_STREAMS mask + forward pass
- `qualia/core/mary_engine_PATCH.py` — Video gating + validation guard
- `validate_video_streams.py` — Video encoder training check
- `mary/modal/serve.py` — Regional read-out + KPI logic (FIX 3 target)
- `code/mary_readouts.py` — Proposed construct-valid, gated weights

**Next PR checkpoints:**
1. **PR: FIX 1+2+3** — De-risk video + read-out validity.
2. **PR: TRIBE latent projection** — Replace KPI_FROM_NETWORKS with interpretable latents (adopt TRIBE v2's discovery).
3. **PR: Heterogeneity layer** — Dalla Porta personalization (gated by cohort availability).
4. **PR: CraniMem episodic memory** — Subject-specific learning in Scan.
5. **PR: PAGRL governance** — Verdict recommendation safety.

---

## Open Questions for Team

1. **TRIBE v2 collaboration?** Is d'Ascoli's team open to a Sapient partnership (Mary as proprietary fine-tune)?
2. **Cohort size:** How many subjects is Mary trained on? Can you scale to 720+ for true foundation-model parity?
3. **Video encoders:** Are slowfast/qwen_vl/got_ocr *present* in the checkpoint (just masked), or absent entirely?
4. **Proprietary data:** How much of Sapient's edge is checkpoint capacity vs. proprietary brand+brain dataset? (Shapes decision: collaborate with TRIBE team vs. build solo.)
5. **Defense use case:** Is WARDEN cognitive-security integration greenlit for Sapient roadmap, or separate product?

---

**Generated:** 2026-07-11 | **Synthesis status:** Ready for team review + prioritization
