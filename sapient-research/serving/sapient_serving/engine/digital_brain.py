"""DigitalBrainEngine — REAL image→visual-cortex encoder (the published Digital Brain model).

Loads a trained GeometryAwareDigitalBrain (geo_subjXX.pkl) + CLIP ViT-L/14 (CPU) and
predicts activation across 25 visual-cortex ROIs for an uploaded image, grouped into
interpretable visual-engagement constructs.

HONESTY (surfaced at /v1/info): this model reads VISUAL CORTEX — faces, bodies, scenes,
words, low-level vision — NOT reward/value/conflict circuitry. The purchase-intent number
here is a *visual-engagement proxy*, not a reward measurement, and it is image-based
(a single frame), not a video time-series.

Config (env):
  SAPIENT_DB_MODEL  = path to a geo_subjXX.pkl (required)
  SAPIENT_DB_REPO   = digital-brain repo root (for `import src.geometry_aware_encoder` unpickling)
  SAPIENT_DB_DEVICE = cpu (default)
"""
from __future__ import annotations

import io
import os
import pickle
import sys

import numpy as np

from ..schemas import ConstructScore, PeakMoment, Prediction, Signals, SignalTimeline
from . import plotting
from . import postprocess as pp
from .base import Stimulus

# Visual-cortex ROI -> interpretable construct groups (Algonauts/NSD ROIs); weights are
# the visual-engagement composite (all positive — these are attention/engagement drivers).
DB_GROUPS = [
    ("face", "Face & Social Engagement", ("FFA-1", "FFA-2", "OFA"),
     "Fusiform/occipital face areas — social & face salience.", 0.26),
    ("body", "Body & Action", ("EBA", "FBA-2"),
     "Extrastriate/fusiform body areas — embodied / action response.", 0.16),
    ("scene", "Scene & Context", ("PPA", "RSC", "OPA"),
     "Parahippocampal / retrosplenial place areas — context & setting.", 0.16),
    ("visual", "Low-level Visual Salience", ("V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "early"),
     "Early visual cortex — raw visual salience.", 0.22),
    ("text", "Text & Branding", ("VWFA-1", "VWFA-2", "OWFA"),
     "Visual word-form areas — on-screen text / logo reading.", 0.10),
    ("higher", "Higher-Order Visual", ("ventral", "lateral", "parietal", "midventral", "midlateral", "midparietal"),
     "Higher visual streams — integrated scene / object processing.", 0.10),
]
_POS_W = sum(w for *_, w in DB_GROUPS)


class DigitalBrainEngine:
    model_version = "digital-brain (real, visual cortex)"

    def __init__(self) -> None:
        model_path = os.getenv("SAPIENT_DB_MODEL")
        repo = os.getenv("SAPIENT_DB_REPO")
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(
                "DigitalBrainEngine needs SAPIENT_DB_MODEL (path to a geo_subjXX.pkl) and "
                "SAPIENT_DB_REPO (digital-brain repo root). Use SAPIENT_ENGINE=mock for the sandbox."
            )
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            import torch  # noqa: F401
            import src.geometry_aware_encoder  # noqa: F401  (registers classes for unpickling)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"DigitalBrainEngine deps missing (torch + digital-brain src): {exc}") from exc

        with open(model_path, "rb") as fh:
            self._brain = pickle.load(fh)
        self.model_version = f"digital-brain:{getattr(self._brain, 'subj_id', '?')} (real, visual cortex)"
        self._device = os.getenv("SAPIENT_DB_DEVICE", "cpu")
        self._clip = None
        self._proc = None

    # ---- CLIP image -> 1024-d feature (heavy; loaded lazily) ----
    def _clip_features(self, image_bytes: bytes) -> np.ndarray:  # pragma: no cover - needs CLIP download
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        if self._clip is None:
            self._proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self._clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self._device).eval()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with torch.no_grad():
            inp = self._proc(images=[img], return_tensors="pt").to(self._device)
            out = self._clip.vision_model(pixel_values=inp["pixel_values"])
            return out.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 1024)

    def predict_from_features(self, feats: np.ndarray, *, include_full_parcels: bool = False) -> Prediction:
        """Core readout (no CLIP): CLIP features -> real ROI predictions -> constructs -> signals."""
        preds = self._brain.predict(np.asarray(feats, dtype=np.float32))  # {roi: (1, n_vox)}
        roi_act = {roi: float(np.mean(v)) for roi, v in preds.items()}

        group_act = {}
        for key, _label, rois, _desc, _w in DB_GROUPS:
            vals = [roi_act[r] for r in rois if r in roi_act]
            group_act[key] = float(np.mean(vals)) if vals else 0.0

        vals = np.array(list(group_act.values()), dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        norm = {k: (0.5 if hi - lo < 1e-9 else (v - lo) / (hi - lo)) for k, v in group_act.items()}

        engagement = sum(w * norm[key] for key, _l, _r, _d, w in DB_GROUPS) / _POS_W
        pi = float(np.clip(100.0 * (0.35 + 0.65 * engagement), 0.0, 100.0))
        rec = "Buy" if pi >= 60 else ("Sell" if pi <= 40 else "Hold")

        constructs = [
            ConstructScore(key=key, label=label, score=round(norm[key] * 100.0, 1),
                           networks=list(rois), description=desc)
            for key, label, rois, desc, _w in DB_GROUPS
        ]
        roi_scores = {label: round(norm[key] * 100.0, 1) for key, label, _r, _d, _w in DB_GROUPS}
        timeline = SignalTimeline(
            seconds=[0.0, 1.0],
            purchase_intent=[round(pi, 1), round(pi, 1)],
            attention=[round(norm["visual"] * 100.0, 1)] * 2,
            emotion=[round(norm["face"] * 100.0, 1)] * 2,
        )
        signals = Signals(
            purchase_intent=round(pi, 1), recommendation=rec, confidence=0.70,
            constructs=constructs, timeline=timeline,
            peak_moments=[PeakMoment(t_seconds=0.0, purchase_intent=round(pi, 1), label="Image response")],
        )
        summary = pp.summarize_parcels(
            np.clip(np.array(list(roi_act.values()), dtype=float), None, None),
            include_full=include_full_parcels,
        )
        image = plotting.roi_bar_image(roi_scores, title="Predicted visual-cortex response (Digital Brain)")
        return Prediction(roi_scores=roi_scores, parcels=summary, signals=signals,
                          image_data_uri=image, n_timepoints=1)

    def predict(self, stim: Stimulus, *, include_full_parcels: bool = False) -> Prediction:
        image_bytes = stim.video_bytes or stim.audio_bytes  # treat the upload as an image / first frame
        if image_bytes is None:
            raise RuntimeError("DigitalBrainEngine needs an image upload (a frame), not transcript-only.")
        return self.predict_from_features(self._clip_features(image_bytes),
                                          include_full_parcels=include_full_parcels)
