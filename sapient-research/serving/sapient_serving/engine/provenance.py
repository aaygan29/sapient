"""Honest provenance surfaced at /v1/info.

Non-negotiable integrity rule: this API serves *Sapient-1*, NOT 'Mary' (the
6-stream research model in the papers). We never present the papers' metrics as
this model's. A technical client/investor checking should find the disclosure here.
"""
from __future__ import annotations

from ..schemas import InfoResponse
from ..settings import settings


def info() -> InfoResponse:
    return InfoResponse(
        name="Sapient-1 Encoding API",
        served_model=settings.model_version,
        is_paper_model=False,
        provenance={
            "mode": ("live (Sapient encoder)" if settings.engine_mode == "real"
                     else "digital-brain — REAL visual-cortex predictions (faces/scenes/bodies/text/vision); "
                          "purchase-intent here is a visual-engagement proxy, not a reward measurement"
                     if settings.engine_mode == "digital_brain"
                     else "sandbox — illustrative outputs, no model weights loaded"),
            "model": "Sapient-1 — trimodal brain encoder (video+audio+text -> fMRI).",
            "architecture_basis": "Derived from the published TRIBE v1 paper architecture; "
            "weights trained from scratch by The Sapient Company.",
            "NOT_the_paper_model": "This is NOT 'Mary', the 6-stream research encoder described in "
            "'Cortex of One' / 'The Convergence Paper'. Sapient-1 is a clean-room, license-clean "
            "re-build (3 streams: V-JEPA2, W2V-BERT, Llama-3.2-3B; d_model=1152; 4 subjects). "
            "It does not reproduce those papers' reported metrics.",
            "training_data": "CC0 only (CNeuroMod CC0 subjects + BOLD Moments + Narratives augmentation).",
            "signal_grounding": "Buy/sell signals follow approach−avoidance neuroforecasting "
            "(Knutson 2007) and are informed by Kapoor 2023, Çakir 2018, Shang 2018, Vlăsceanu 2014. "
            "See docs/CONSUMER_NEURO_REFERENCES.md.",
        },
        output_space={
            "vertices": 20484,
            "atlas": "fsaverage5 (10,242 vertices / hemisphere)",
            "rate_hz": 1,
            "parcellation": "Schaefer-1000",
            "roi_decomposition": "Yeo-7 networks (canonical mapping pending confirmation from repo atlas code)",
        },
        metric_note="Per-ROI / per-parcel values are the model's PREDICTED activation over the "
        "stimulus window. They are NOT Pearson correlations against measured fMRI (the paper "
        "headline metric requires ground-truth brain recordings, which this API does not have for "
        "novel content).",
        licenses={
            "model": "Apache-2.0",
            "text_encoder_llama_3_2_3b": "Llama 3.2 Community License",
            "vjepa2": "MIT",
            "w2v_bert": "MIT",
        },
        notice="Outputs are model predictions for research/experimentation, not medical or "
        "diagnostic information.",
    )
