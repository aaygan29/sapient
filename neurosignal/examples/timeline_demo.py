"""End-to-end: video (illustrative per-second activations) -> encode -> stats -> live brain.

Builds a 12s ad arc, runs it for the AVERAGE brain and two ENROLLED people, compiles the
stats layer, and writes the interactive live-brain report. The per-second networks here
stand in for what the learned Mary encoder emits per second (no scientific claim — same
--mock discipline); swap in Mary's per-second predictions to drive it from real video.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurosignal import BrainFile
from neurosignal.stats import compile_stats
from neurosignal.timeline import analyze_timeline
from neurosignal.viz import render_report, save_report

HERE = os.path.dirname(os.path.abspath(__file__))


def ad_arc():
    frames = []
    for t in range(12):
        if t < 4:
            f = {"Visual": 0.62, "Default": 0.6, "Somatomotor": 0.4, "DorsalAttention": 0.42,
                 "VentralAttention": 0.22, "Limbic": 0.3, "Frontoparietal": 0.52}
        elif t < 8:
            f = {"Visual": 0.72, "Default": 0.5, "Somatomotor": 0.42, "DorsalAttention": 0.72,
                 "VentralAttention": 0.42, "Limbic": 0.58 + 0.05 * (t - 4), "Frontoparietal": 0.46}
        else:
            f = {"Visual": 0.6, "Default": 0.4, "Somatomotor": 0.4, "DorsalAttention": 0.6,
                 "VentralAttention": 0.86, "Limbic": 0.9, "Frontoparietal": 0.2}
        frames.append(f)
    return frames


SUBJECTS = {
    "Average brain": None,
    "Reward-seeker": BrainFile(subject="reward_seeker",
                               network_head={"Limbic": +0.20, "Frontoparietal": -0.18, "VentralAttention": +0.10}),
    "Deliberator": BrainFile(subject="deliberator",
                             network_head={"Limbic": -0.15, "Frontoparietal": +0.22, "VentralAttention": -0.08}),
}

reports = {}
arc = ad_arc()
for label, bf in SUBJECTS.items():
    tl = analyze_timeline(per_second_networks=arc, fps=1.0, brain_file=bf,
                          source=f"ad-arc(illustrative) :: {label}")
    st = compile_stats(tl)
    reports[label] = {"timeline": tl.to_dict(), "stats": st.to_dict()}
    print(f"{label:<16} {st.headline}")

# 1) full standalone report
save_report(os.path.join(HERE, "live_brain_report.html"), reports,
            title="Neuro Read-out — Live Digital Brain",
            subtitle="Same 12s clip, three brains. Play to watch areas light up second-by-second.")
# 2) embeddable widget html (the div+script only)
with open(os.path.join(HERE, "live_brain_widget.html"), "w") as fh:
    fh.write(render_report(reports, title="Neuro Read-out — Live Digital Brain",
                           subtitle="Same 12s clip, three brains — play to watch the cortex light up."))
# 3) raw data (the 'stats' layer output) for any other front-end
with open(os.path.join(HERE, "timeline_data.json"), "w") as fh:
    json.dump(reports, fh, indent=2)

print("\nwrote: live_brain_report.html, live_brain_widget.html, timeline_data.json")
