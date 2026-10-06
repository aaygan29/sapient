"""MaryModel — faithful 6-stream ORCLE-Nano brain encoder. Per CONTRACTS.md §2/§6.

Maps cached frozen-backbone features → predicted fMRI BOLD over 20,484 fsaverage5
vertices. The trainable adapter (~35M params) fuses the 6 streams, aligns them in
time, and predicts the cortical response.

Forward signature (CONTRACTS.md §6):
    pred = model(features: dict[str, Tensor(B, T_2Hz, D_m)],
                 subject_idx: Tensor(B,))            # -> (B, T_TR, 20484)

Architecture lock (ORCLE-Nano whitepaper §3, 8-step adapter — cite the step if you
ever need to deviate; see
`/Users/robertgutierrez/Desktop/sapient-research/Mary-Model-Insights/01-orcle-nano-architecture.md`):

    1. Per-stream FFN          D_m → 768 (GELU + LayerNorm), one per stream
    2. Per-stream HRF conv      depthwise Conv1d k=5
    3. Modality dropout         p=0.15, zero entire streams, TRAINING ONLY
    4. Fusion Transformer       1-layer pre-norm, learned modality embeds, cross-attn
    5. Attentive temporal pool  learned queries, 2 Hz → T_TR
    6. Demographic slot         additive (age, sex)
    7. Prediction Transformer   2-layer, RoPE on the time axis
    8. Vertex head              group linear + per-subject linear (head.py)

Frozen backbones are loaded ONLY in `data/extract_*` (lineage guarantee, CONTRACTS.md
§7) — never here. This module only sees the cached `(T_2Hz, D_m)` features. Streams may
be MISSING from the features dict (datasets lack some modalities); absent streams are
skipped and excluded from fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from .adapter import (
    STREAM_DIMS,
    STREAM_ORDER,
    AttentiveTemporalPool,
    DemographicSlot,
    FusionTransformer,
    PredictionTransformer,
    StreamEncoder,
)
from .head import VertexHead


@dataclass
class MaryConfig:
    # Core dims (CONTRACTS.md §5, LOCKED).
    d_model: int = 768
    n_fusion_layers: int = 1          # fusion Transformer
    n_prediction_layers: int = 2      # prediction Transformer (RoPE)
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1

    # ORCLE-specific knobs.
    hrf_kernel_size: int = 5          # per-stream depthwise HRF conv
    modality_dropout: float = 0.15    # zero entire streams, training only
    n_vertices: int = 20484           # fsaverage5 (LH+RH)
    rope: bool = True
    use_demographic_slot: bool = True

    # Sequence (2 Hz feature grid → 1 Hz fMRI grid).
    max_stim_len: int = 200           # T_2Hz
    max_fmri_len: int = 100           # T_TR per window

    # Subjects (override per data slice).
    n_subjects: int = 9
    per_subject_rank: int = 16        # low-rank per-subject head (0 = full dense)
    group_rank: int = 128             # low-rank shared group head (0 = full dense)

    # Demographic slot cardinalities.
    n_age_buckets: int = 8
    n_sex: int = 3

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "MaryConfig":
        """Construct from a loaded YAML config (matches sapient1's from_yaml pattern).

        Accepts either the full config dict (with a `model:` block) or the model block
        directly. Unknown keys (streams:, train:, eval:, loss:, ...) are ignored.
        """
        m = cfg.get("model", cfg)
        kwargs = {
            "d_model": m.get("d_model", cls.d_model),
            "n_fusion_layers": m.get("n_fusion_layers", cls.n_fusion_layers),
            "n_prediction_layers": m.get("n_prediction_layers", cls.n_prediction_layers),
            "n_heads": m.get("n_heads", cls.n_heads),
            "ff_mult": m.get("ff_mult", cls.ff_mult),
            "dropout": m.get("dropout", cls.dropout),
            "hrf_kernel_size": m.get("hrf_kernel_size", cls.hrf_kernel_size),
            "modality_dropout": m.get("modality_dropout", cls.modality_dropout),
            "n_vertices": m.get("n_vertices", cls.n_vertices),
            "rope": m.get("rope", cls.rope),
            "use_demographic_slot": m.get("use_demographic_slot", cls.use_demographic_slot),
            "max_stim_len": m.get("max_stim_len", cls.max_stim_len),
            "max_fmri_len": m.get("max_fmri_len", cls.max_fmri_len),
            "n_subjects": m.get("n_subjects", cls.n_subjects),
            "per_subject_rank": m.get("per_subject_rank", cls.per_subject_rank),
            "group_rank": m.get("group_rank", cls.group_rank),
            "n_age_buckets": m.get("n_age_buckets", cls.n_age_buckets),
            "n_sex": m.get("n_sex", cls.n_sex),
        }
        return cls(**kwargs)


class MaryModel(nn.Module):
    """6-stream ORCLE-Nano brain encoder. (B, T_2Hz, D_m) features → (B, T_TR, 20484)."""

    def __init__(self, cfg: MaryConfig):
        super().__init__()
        self.cfg = cfg
        self.stream_order = STREAM_ORDER  # canonical, fixed order

        # ---- Step 1+2: per-stream FFN + HRF conv (one StreamEncoder per stream) ----
        self.stream_encoders = nn.ModuleDict(
            {
                name: StreamEncoder(STREAM_DIMS[name], cfg.d_model, cfg.hrf_kernel_size)
                for name in self.stream_order
            }
        )

        # ---- Step 4: fusion Transformer (modality dropout = step 3, applied in forward) ----
        self.fusion = FusionTransformer(
            d_model=cfg.d_model,
            n_streams=len(self.stream_order),
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
            n_layers=cfg.n_fusion_layers,
        )

        # ---- Step 5: attentive temporal pooling (2 Hz → T_TR) ----
        self.temporal_pool = AttentiveTemporalPool(
            d_model=cfg.d_model,
            max_fmri_len=cfg.max_fmri_len,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )

        # ---- Step 6: demographic slot ----
        if cfg.use_demographic_slot:
            self.demographic = DemographicSlot(cfg.d_model, cfg.n_age_buckets, cfg.n_sex)
        else:
            self.demographic = None

        # ---- Step 7: prediction Transformer (RoPE) ----
        self.prediction = PredictionTransformer(
            d_model=cfg.d_model,
            n_layers=cfg.n_prediction_layers,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
        )

        # ---- Step 8: vertex head (group + per-subject) ----
        self.head = VertexHead(
            cfg.d_model,
            cfg.n_vertices,
            cfg.n_subjects,
            cfg.per_subject_rank,
            cfg.group_rank,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],     # {stream: (B, T_2Hz, D_m)}; streams may be absent
        subject_idx: torch.Tensor,             # (B,) long
        age_bucket: Optional[torch.Tensor] = None,  # (B,) long or None
        sex: Optional[torch.Tensor] = None,         # (B,) long or None
    ) -> torch.Tensor:
        if not features:
            raise ValueError("MaryModel.forward received no stream features.")

        # Infer batch / time from any present stream.
        any_stream = next(iter(features.values()))
        B, T_2Hz = any_stream.shape[0], any_stream.shape[1]
        device = any_stream.device
        S = len(self.stream_order)
        D = self.cfg.d_model

        # ---- Steps 1+2 per present stream; absent streams encoded as zeros + masked ----
        encoded = torch.zeros(B, T_2Hz, S, D, device=device, dtype=any_stream.dtype)
        present = torch.zeros(B, S, dtype=torch.bool, device=device)
        for s_i, name in enumerate(self.stream_order):
            if name in features and features[name] is not None:
                encoded[:, :, s_i, :] = self.stream_encoders[name](features[name])
                present[:, s_i] = True

        # ---- Step 3: modality dropout (training only) — zero present streams w/ p ----
        if self.training and self.cfg.modality_dropout > 0:
            drop = (
                torch.rand(B, S, device=device) < self.cfg.modality_dropout
            ) & present
            keep = present & ~drop
            # Guard: never drop the last remaining stream for a sample.
            empty = ~keep.any(dim=1)
            if empty.any():
                # Restore the first present stream for samples that lost everything.
                first_present = present.float().argmax(dim=1)
                keep[empty, first_present[empty]] = True
            present = keep
            encoded = encoded * present.float().view(B, 1, S, 1)

        # ---- Step 4: fusion Transformer (cross-attn over present streams) ----
        fused = self.fusion(encoded, present)             # (B, T_2Hz, D)

        # ---- Step 5: attentive temporal pooling → T_TR ----
        t_tr = min(self.cfg.max_fmri_len, T_2Hz // 2)
        x = self.temporal_pool(fused, t_tr)               # (B, T_TR, D)

        # ---- Step 6: demographic slot ----
        if self.demographic is not None:
            x = self.demographic(x, age_bucket, sex)

        # ---- Step 7: prediction Transformer (RoPE) ----
        x = self.prediction(x)                            # (B, T_TR, D)

        # ---- Step 8: vertex head (group + per-subject) ----
        return self.head(x, subject_idx)                  # (B, T_TR, n_vertices)

    def penultimate(
        self,
        features: dict[str, torch.Tensor],
        age_bucket: Optional[torch.Tensor] = None,
        sex: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run steps 1-7 only and return the penultimate features `x` (B, T_TR, d_model)
        that the vertex head (step 8) consumes. Subject-independent (the per-subject term
        lives entirely in `self.head`). Used by the per-subject probe fine-tune to cache
        frozen-backbone features once, then train only the head on top.

        Modality dropout (step 3) is governed by `self.training` exactly as in `forward`,
        so call under `model.eval()` for deterministic features.
        """
        if not features:
            raise ValueError("MaryModel.penultimate received no stream features.")
        any_stream = next(iter(features.values()))
        B, T_2Hz = any_stream.shape[0], any_stream.shape[1]
        device = any_stream.device
        S = len(self.stream_order)
        D = self.cfg.d_model

        encoded = torch.zeros(B, T_2Hz, S, D, device=device, dtype=any_stream.dtype)
        present = torch.zeros(B, S, dtype=torch.bool, device=device)
        for s_i, name in enumerate(self.stream_order):
            if name in features and features[name] is not None:
                encoded[:, :, s_i, :] = self.stream_encoders[name](features[name])
                present[:, s_i] = True

        if self.training and self.cfg.modality_dropout > 0:
            drop = (
                torch.rand(B, S, device=device) < self.cfg.modality_dropout
            ) & present
            keep = present & ~drop
            empty = ~keep.any(dim=1)
            if empty.any():
                first_present = present.float().argmax(dim=1)
                keep[empty, first_present[empty]] = True
            present = keep
            encoded = encoded * present.float().view(B, 1, S, 1)

        fused = self.fusion(encoded, present)
        t_tr = min(self.cfg.max_fmri_len, T_2Hz // 2)
        x = self.temporal_pool(fused, t_tr)
        if self.demographic is not None:
            x = self.demographic(x, age_bucket, sex)
        x = self.prediction(x)
        return x

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
