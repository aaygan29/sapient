"""Real engine: the true Sapient-1 forward pass.

Imports the private `sapient1` package + frozen encoders, loads the checkpoint,
runs inference, and post-processes to ROI/parcel summaries. All heavy imports are
guarded so that merely importing this module never breaks mock mode; constructing
RealEngine without the deps/checkpoint raises a clear, actionable error.

Prerequisites (see README):
  * pip install ".[real]" and the sapient1 package on the GPU box
  * a real checkpoint at SAPIENT_MODEL_ID on HuggingFace (per the repo README this
    is "not yet uploaded" — until then, run in mock mode)
  * the canonical Schaefer-1000 -> ROI mapping to populate roi_scores faithfully
"""
from __future__ import annotations

import numpy as np

from ..schemas import Prediction
from ..settings import settings
from . import plotting
from . import postprocess as pp
from .base import Stimulus


def _load_state(ckpt_path: str, torch):
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file  # type: ignore
        return load_file(ckpt_path)
    return torch.load(ckpt_path, map_location="cpu")


class RealEngine:
    model_version = settings.model_version

    def __init__(self) -> None:
        try:
            import torch  # type: ignore
            from sapient1 import SapientConfig, SapientModel, load_parcellation  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "RealEngine requires the [real] extra (torch, sapient1, scipy, nilearn) and a "
                "checkpoint. Install on the GPU box and set SAPIENT_ENGINE=real, or use "
                f"SAPIENT_ENGINE=mock locally. Underlying import error: {exc}"
            ) from exc

        self._torch = torch
        self._SapientModel = SapientModel
        self._SapientConfig = SapientConfig
        self._model = self._load_checkpoint()
        try:
            self._parcel_matrix = load_parcellation()
        except Exception:
            self._parcel_matrix = None  # ROI/parcel projection degrades gracefully
        from .features import RealFeatureExtractor
        self._features = RealFeatureExtractor()

        # Canonical Schaefer-1000 -> Yeo-7 network mapping for the buy/sell readout.
        # Loaded lazily from nilearn on first predict (postprocess.load_schaefer_network_ids).
        self._network_ids = None

    def _load_checkpoint(self):
        from huggingface_hub import hf_hub_download  # type: ignore
        torch = self._torch
        ckpt_path = hf_hub_download(settings.model_id, filename="model.safetensors")
        cfg = self._SapientConfig()  # TODO: load released config.yaml for exact dims/n_subjects
        model = self._SapientModel(cfg).eval()
        model.load_state_dict(_load_state(ckpt_path, torch), strict=True)
        return model

    def predict(self, stim: Stimulus, *, include_full_parcels: bool = False) -> Prediction:
        torch = self._torch
        feats = self._features.extract(
            video_bytes=stim.video_bytes, audio_bytes=stim.audio_bytes, transcript=stim.transcript
        )
        with torch.no_grad():
            video = torch.from_numpy(feats.video).float().unsqueeze(0)
            audio = torch.from_numpy(feats.audio).float().unsqueeze(0)
            text = torch.from_numpy(feats.text).float().unsqueeze(0)
            subject = torch.tensor([stim.subject_idx], dtype=torch.long)
            vert = self._model(video, audio, text, subject)  # (1, 100, 20484)
        vert_np = vert.squeeze(0).cpu().numpy()

        from . import signals as sig

        n = int(vert_np.shape[0])
        seconds = np.arange(n, dtype=float)

        if self._parcel_matrix is not None:
            parcels = np.asarray(pp.vertices_to_parcels(vert_np, self._parcel_matrix))  # (T, 1000)
            parcel_mean = parcels.mean(axis=0)
            if self._network_ids is None:
                self._network_ids = pp.load_schaefer_network_ids(parcels.shape[1])
            construct_ts = sig.construct_ts_from_parcels(parcels, self._network_ids)  # paper-grounded readout
            network_ts = sig.network_ts_from_parcels(parcels, self._network_ids)      # per-second live brain
            roi_scores = pp.parcels_to_rois(parcel_mean, network_ids=self._network_ids)
        else:
            parcel_mean = np.zeros(pp.N_PARCELS)
            construct_ts = {c.key: np.full(n, 0.5) for c in sig.CONSTRUCTS}
            network_ts = None
            roi_scores = {}

        signals = sig.build_signals(construct_ts, seconds, network_ts=network_ts)
        summary = pp.summarize_parcels(parcel_mean, include_full=include_full_parcels)
        try:
            image = plotting.surface_image(vert_np)
        except NotImplementedError:
            image = plotting.roi_bar_image(roi_scores or {"(mapping pending)": float(parcel_mean.mean())})

        return Prediction(
            roi_scores=roi_scores,
            parcels=summary,
            signals=signals,
            image_data_uri=image,
            n_timepoints=n,
        )
