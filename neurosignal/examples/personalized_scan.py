"""Same stimulus, different PEOPLE -> different read-outs.

Demonstrates the Cortex-of-Anyone integration: an enrolled `brain_file` personalizes
the neurosignal read-out, so a real scan returns individual results instead of the
average-brain score. (Network-level heads here are illustrative; in production they are
produced by enrolling a subject — Tier A fMRI or Tier B EEG — see enrollment.py.)
"""
import neurosignal as ns
from neurosignal import BrainFile

STIM = "Act now — this exclusive deal disappears in minutes. Don't miss out!"

# Two enrolled individuals with different reward / analytic-control profiles.
reward_seeker = BrainFile(subject="reward_seeker",
                          network_head={"Limbic": +0.35, "Default": +0.20, "Frontoparietal": -0.25})
deliberator = BrainFile(subject="deliberator",
                        network_head={"Limbic": -0.20, "Frontoparietal": +0.35, "VentralAttention": +0.15})

runs = {"AVERAGE BRAIN (baseline)": ns.analyze(text=STIM),
        "reward_seeker (enrolled)": ns.analyze(text=STIM, brain_file=reward_seeker),
        "deliberator (enrolled)": ns.analyze(text=STIM, brain_file=deliberator)}

keys = ["valence", "arousal", "engagement", "manipulation", "buy_sell"]
print(f"\nStimulus: {STIM!r}\n")
print(f"{'read-out':<28}" + "".join(f"{k:>13}" for k in keys) + "   personalized")
print("-" * 100)
for label, a in runs.items():
    sc = {m.key: m.score for m in a.metrics}
    row = f"{label:<28}" + "".join(f"{sc.get(k, float('nan')):>13.1f}" for k in keys)
    print(row + f"   {a.personalized}")
print("\nProvenance (deliberator):", runs["deliberator (enrolled)"].provenance)
