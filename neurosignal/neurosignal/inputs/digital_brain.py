"""Optional input: predict VISUAL-cortex activation from an image via the Digital Brain
geometry-aware encoder (CLIP ViT-L/14 -> trained per-ROI MLP), then run detection.

SCOPE / HONESTY: this encoder reads VISUAL cortex only (V1-V4, face/body/scene/word areas).
It populates the `visual_sensory` construct; reward / emotion / conflict require whole-brain
fMRI (use detect_from_parcels / detect_from_schaefer). The result's coverage + notes say so.

SECURITY: the trained model is a Python pickle, and unpickling executes arbitrary code. We
REFUSE to load it unless you pass trusted=True (or set NEUROSIGNAL_TRUST_PICKLE=1). Only load
model files you produced or fully trust.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

from ..detect import detect_from_networks
from ..types import DetectionResult


def _load_encoder(model_path: str, repo_path: str | None, trusted: bool):
    if not (trusted or os.getenv("NEUROSIGNAL_TRUST_PICKLE") == "1"):
        raise PermissionError(
            "Refusing to unpickle a model file (arbitrary-code-execution risk). "
            "Pass trusted=True or set NEUROSIGNAL_TRUST_PICKLE=1, and only load files you trust."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    import torch  # noqa: F401
    import src.geometry_aware_encoder  # noqa: F401  (registers classes for unpickling)
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def _clip_features(image_path: str, device: str) -> np.ndarray:  # pragma: no cover - needs CLIP
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inp = proc(images=[img], return_tensors="pt").to(device)
        out = model.vision_model(pixel_values=inp["pixel_values"])
        return out.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 1024)


def detect_from_image(
    image_path: str, *, model_path: str, repo_path: str | None = None,
    device: str = "cpu", trusted: bool = False,
) -> DetectionResult:  # pragma: no cover - needs CLIP + model
    brain = _load_encoder(model_path, repo_path, trusted)
    preds = brain.predict(_clip_features(image_path, device))  # {roi: (1, n_vox)}
    visual = float(np.mean([float(np.mean(v)) for v in preds.values()]))
    res = detect_from_networks(
        {"Visual": visual},
        source=f"Digital Brain encoder (visual cortex; {os.path.basename(model_path)})",
    )
    res.notes.insert(
        0,
        "Digital Brain encoder measures VISUAL cortex only; reward/emotion/conflict constructs "
        "are NOT covered by this input — use whole-brain fMRI (detect_from_parcels) for those.",
    )
    return res
