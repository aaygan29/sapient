# DARPA AI Forge RFI — Response Draft (The Sapient Company)

Status: DRAFT for review. RFI published 2026-06-01, response deadline 2026-06-22.
Program offices: DARPA IPTO + NSF + NIST CAISI. PM: Matthew Marge.
Thrust areas: AI interpretability, AI control, adversarial robustness for national security.
No em dashes used.

> Note on fit: AI Forge funds university-led foundational research and is an RFI, not a
> funding solicitation. Sapient responds as an industry contributor with a specific,
> defensible capability: neural-grounded interpretability of persuasion and manipulation
> in AI systems. This maps most directly to the interpretability and AI control thrusts.

---

## Respondent
The Sapient Company. Point of contact: Aayush Gandhi, CTO, aayush@thesapientcompany.com.
Type: small business / startup. Prior relevant work: in-silico fMRI encoding (Sapient-1),
manipulation and sycophancy detection in language models (neurosignal), and neural-grounded
evaluation of AI outputs.

## 1. The problem we address
Current interpretability work explains what a model computes. It does not measure the effect
a model has on the human receiving its output. For national security, the second question is
the one that matters: is an AI system persuading, manipulating, or coercing the operator, and
can we detect that in real time and at scale. We propose that human-response grounding is a
missing axis of interpretability and AI control.

## 2. Capability
- Sapient-1: an in-silico fMRI encoder that predicts whole-cortex activation to a stimulus,
  20,484 vertices at 1 Hz, conditioned per subject. It lets us estimate how a given AI output
  drives human attention, reward and value, emotional arousal, and decision conflict.
- neurosignal: an evidence-based detector that scores model outputs for affective valence,
  arousal, engagement, manipulation, and sycophancy, with each score tied to cited neural
  literature and withheld when the signal is confounded. This is auditable interpretability,
  not a black box.

## 3. Relevance to the three thrusts
- Interpretability: we add a human-effect layer. Instead of only asking what a model
  represents, we measure the predicted neural and behavioral impact of its outputs on people.
- AI control: a calibrated, real-time manipulation and sycophancy signal is a control
  primitive. It flags when an AI is steering an operator and can gate or escalate.
- Adversarial robustness: manipulation is an adversarial attack on the human in the loop.
  Detecting it protects the human-AI team, the softest target in contested environments.

## 4. What we would contribute to a pre-competitive forum
- An open benchmark for AI-on-human persuasion and manipulation, grounded in neural data.
- A shared, auditable metric layer (the neurosignal construct set) other labs can adopt.
- A bridge between the computational-neuroscience and AI-safety communities, which rarely
  share evaluation tooling today.

## 5. Open questions for AI Forge
- Would the forum value human-response grounding as a recognized interpretability axis.
- What data-governance and consent model would the program require for human-response data
  used in national-security evaluation.

## 6. Caveats stated plainly
Sapient-1 produces predicted activations, not measured fMRI. Results to date are
proof-of-concept scale. We say so in all materials and withhold scores when confounded.

---

### TODO before submitting
- Confirm the RFI submission channel (the DARPA page says it will be updated; verify whether
  responses go through SAM.gov, a BAA, or an email address). Do not submit until confirmed.
- Decide whether to respond as The Sapient Company or as an individual researcher (Aayush),
  given the university-led framing. A named academic collaborator would strengthen this.
