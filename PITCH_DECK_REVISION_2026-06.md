# Sapient Pitch Deck — Science-Accurate Copy Revision

Date: 2026-06-15. Source of truth: `sapient/README.md`, `sapient/CLAUDE.md`,
`neurosignal/README.md`, `Whitepaper/whitepaper.tex`. No em dashes used.

Purpose: keep the deck's structure and design, but fix the places where the copy
either overclaims, blurs the science, or uses a vague buzzword where a concrete,
defensible fact is stronger. Each slide below = what it says now, what to change, why.

---

## Slide 1 — Title
Now: "Superhuman Intelligence / the cognition protocol for machines"
Keep. It is a strong cold open. No science to fix.

## Slide 2 — The Future (market stats)
Keep the four stats. One honesty tweak: the stats are third party projections,
which is fine, but the sub head "none of them do that yet" is the real wedge.
Tighten to: "Every one of these needs to understand the human on the other side in
real time. None of them can yet."

## Slide 3 — The Problem
Keep. "Intelligence reads your words, it does not read you" is the core line.
Replace "cognitive state" (vague) with the concrete list you can actually predict:
"attention, emotion, reward and value, decision conflict, and memory."

## Slide 4 — What we're building
Keep the diagram. Sub line should name the mechanism, not just "state of being":
"A real time read of attention, emotion, reward, decision conflict, and memory for
every human a machine interacts with."

## Slide 5 — Traction
Numbers are fine as stated (37K+ waitlist, 10M+ views, 5K+ scans across 1,265 orgs).
Make sure the deck and every outbound pitch use the SAME numbers. The Refactor
submission used 5,000+ scans / 1,265 orgs / 37K+ waitlist / 10M+ views. Lock these.

## Slide 6 — Market size
Keep TAM/SAM/SOM. No change.

## Slide 7 — Business model
Keep the four layers. Two precision fixes:
- Layer 1 "Per Call API": today this is the Scan / Mary read out. Say "live today"
  only for what is live. Per the README, Scan and Mary are live, so this is fair.
- Layer 3 "Data Licensing": frame as the forward plan, not current revenue, unless a
  dataset has actually been licensed. Honest framing protects you in diligence.

## Slide 8 — Our Moat (the 5 data tiers) — BIGGEST SCIENCE FIX
This slide is where diligence will push hardest. Right now Tiers 1 to 5 read as if
all are in hand. They are not, and that is fine if you label them. Relabel by status:

- Tier 1 Behavioral Signal (eye movement, voice tone, dwell time): LIVE / collecting.
- Tier 2 Neural Signal (EEG, brain electrical activity and timing): IN PROGRESS.
- Tier 3 Synthesized fMRI: this is the actual core IP and it is live today. Reword
  from "Deep brain mapping from EEG" to what the model truly does:
  "Sapient-1 predicts whole cortex fMRI activation from a stimulus, 20,484 vertices
  at 1 Hz, conditioned per subject." That is the real, defensible sentence. Lead with it.
- Tier 4 Neural Implant: ROADMAP. Label it.
- Tier 5 Neural Decoding (fMRI-conditioned omnimodal generative): RESEARCH / ROADMAP.

Why: the "Cortex of One" result (per subject head beats the average brain out of
sample) is your strongest single scientific claim. It belongs on this slide, in one
line, because it is the thing competitors cannot easily copy.

Add one honesty line that actually builds trust with technical investors:
"These are predicted activations, not measured fMRI, and we withhold a score when the
signal is confounded." This is straight from neurosignal's design and it reads as
rigor, not weakness.

## Slide 9 — Competition
Keep the 2x2. Accurate as far as Affectiva (single modal affect), Hume (voice),
Inworld (vertical). Fine.

## Slide 10 — Team
Keep both bios. On the CTO bio, "evaluated AI driven models of the brain for
statistical validity in emotional processing, neural encoding and decoding, and
coercive manipulation" is accurate to the research program. Good. Consider adding the
one quantified proof point: the per subject encoder result.

## Slide 11 — The Ask
$2M pre seed at $20M. Keep. Tie milestones to the moat slide: "Foundational model V1"
= Tier 3 at scale; "first proprietary dataset" = Tiers 1 and 2 collected.

---

## Cross-deck cleanups
1. Pick ONE name and use it everywhere: the deck says "The Sapient Company"; the
   product README says the model is "Sapient-1." Keep company = The Sapient Company,
   model = Sapient-1. Do not let "Mary" and "Qualia" leak into investor copy without a
   one line definition (Mary = the live workspace, Qualia = the model line in the
   whitepaper).
2. Replace every "cognitive state" / "state of being" with the concrete construct
   list. Specificity reads as a real model; abstraction reads as a pitch.
3. Lead the moat with Tier 3 (the live fMRI encoder), not Tier 4/5 (implant), so the
   first thing investors see is the thing that already works.
