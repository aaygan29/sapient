"""Run the live digital brain end-to-end through the serving engine.

Works TODAY with the mock engine (no checkpoint). Set SAPIENT_ENGINE=real with a
checkpoint to drive it from the real Sapient-1 forward pass — the adapter is unchanged.

    python3 examples/run_live_brain.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)                                   # sapient_serving
sys.path.insert(0, os.path.join(REPO, "..", "neurosignal"))  # sibling neurosignal

from sapient_serving.engine import build_engine            # noqa: E402
from sapient_serving.engine.base import Stimulus           # noqa: E402
from sapient_serving.engine.live_brain import save_live_brain  # noqa: E402

engine = build_engine()  # mock unless SAPIENT_ENGINE=real|digital_brain
stim = Stimulus(
    transcript=("Soft music. A beautiful, cinematic reveal of the new luxury model. "
                "The crowd is stunned. Exciting, aspirational, premium — desire and reward. "
                "Then: act now, limited time, don't miss out."),
    filename="luxury_ad.mp4",
)

out = save_live_brain(
    os.path.join(HERE, "live_brain.html"), engine, stim,
    subject_idxs=(0, 1, 2),
    subject_labels=["Subject 0 (default)", "Subject 1", "Subject 2"],
    title="Sapient-1 — Live Digital Brain",
)
print(f"engine: {engine.model_version}")
print(f"wrote: {out}")
print("open it in a browser and press Play; toggle subjects to see per-person brains.")
