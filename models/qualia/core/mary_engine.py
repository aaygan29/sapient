"""qualia.core.mary_engine — load Mary once and run it; the Qualia<->Mary boundary.

Mary is treated as a frozen engine loaded from a versioned checkpoint. Because the
model config travels INSIDE the checkpoint, pointing the registry at a newer/better
Mary upgrades every Qualia capability with no code change here.

Reuses the load logic proven in Mary-Papers/paper-C-cross-modal/code/mary_encode.py
(checkpoint carries config; filter mismatched/unused-stream keys; freeze).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Streams Mary actively uses today (audio + text). Video streams are inactive in
# current checkpoints (CONTRACTS.md §0) and are simply omitted -> Mary skips them.
ACTIVE_STREAMS = ("beats", "whisper", "qwen_ctx")
N_VERTICES = 20484  # fsaverage5, locked (CONTRACTS.md §1)


def _mary_root() -> str:
    """Locate the sibling `mary/` package (sapient-models/mary)."""
    env = os.environ.get("QUALIA_MARY_ROOT")
    if env:
        return env
    if Path("/root/maryrepo").exists():       # Modal mount convention
        return "/root/maryrepo"
    # qualia/core/mary_engine.py -> parents[2] == sapient-models ; mary is the sibling
    return str(Path(__file__).resolve().parents[2] / "mary")


class MaryEngine:
    """Frozen Mary, loaded once. Provides encode() (-> vertices) and embed()
    (-> subject-independent penultimate embedding)."""

    def __init__(self, ckpt_path: str | Path, device: str = "cpu",
                 channel: str | None = None, version: str | None = None):
        root = _mary_root()
        if root not in sys.path:
            sys.path.insert(0, root)
        from mary.model import MaryConfig, MaryModel  # noqa: E402

        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        self.config = ck["config"]
        model = MaryModel(MaryConfig.from_yaml(self.config))
        # Filter checkpoint keys to those matching current architecture (only the
        # unused video-stream encoders differ across ckpts; we never feed them).
        msd, src = model.state_dict(), ck["state_dict"]
        filtered = {k: v for k, v in src.items()
                    if k in msd and tuple(v.shape) == tuple(msd[k].shape)}
        dropped = [k for k in src if k not in filtered]
        bad = [k for k in dropped
               if any(f"stream_encoders.{s}." in k for s in ACTIVE_STREAMS)]
        if bad:
            raise RuntimeError(f"Mary load mismatch on ACTIVE streams: {bad[:5]}")
        model.load_state_dict(filtered, strict=False)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.device = device
        self.channel = channel
        self.version = version
        self.d_model = int(model.cfg.d_model)
        self.n_vertices = N_VERTICES

    @classmethod
    def from_channel(cls, channel: str = "improving", device: str = "cpu") -> "MaryEngine":
        """Load the checkpoint for a registry channel (improving=rides upgrades)."""
        from .registry import resolve
        r = resolve(channel)
        return cls(r["ckpt"], device=device, channel=r["channel"], version=r["version"])

    @staticmethod
    def assemble_features(feat_dir: str | Path, streams=ACTIVE_STREAMS,
                          device: str = "cpu") -> dict[str, torch.Tensor]:
        """Load cached `{feat_dir}/{stream}.npy` -> {stream: (1, T_2Hz, D_m)}.
        Present streams trimmed to a common T; missing streams omitted."""
        feat_dir = Path(feat_dir)
        arrays = {}
        for s in streams:
            p = feat_dir / f"{s}.npy"
            if p.exists():
                arrays[s] = np.load(p).astype(np.float32)
        if not arrays:
            raise FileNotFoundError(f"no stream features under {feat_dir} for {list(streams)}")
        T = min(a.shape[0] for a in arrays.values())
        return {s: torch.from_numpy(a[:T]).float()[None].to(device) for s, a in arrays.items()}

    @torch.no_grad()
    def encode(self, features: dict[str, torch.Tensor], subject_idx: int = 0) -> np.ndarray:
        """Mary forward -> predicted fMRI (T_TR, 20484) float32. Needs T_2Hz >= 2."""
        subj = torch.tensor([int(subject_idx)], dtype=torch.long, device=self.device)
        return self.model(features, subj)[0].float().cpu().numpy()

    @torch.no_grad()
    def embed(self, features: dict[str, torch.Tensor]) -> np.ndarray:
        """Subject-independent brain representation: mean over time of the
        penultimate (pre-vertex-head) features. Shape (d_model,)."""
        x = self.model.penultimate(features)          # (1, T_TR, d_model)
        return x[0].mean(dim=0).float().cpu().numpy()

    def provenance(self) -> dict:
        return {"channel": self.channel, "version": self.version,
                "d_model": self.d_model, "vertices": self.n_vertices, "space": "fsaverage5"}
