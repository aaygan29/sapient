"""SapientModel — multimodal brain encoder. Per spec §3.

Architecture lock (non-negotiable, cite the decision number if you ever need
to deviate; see engineering-outline/00_HANDOFF_README.md):

  Order (decision #5):
    project → modality_dropout → SUM → pool → +pos → +subj → transformer → head

  1.  modality_dropout p=0.3 AFTER projections, BEFORE sum.
  2.  subject_dropout p=0.1 at training time only.
  3.  pos_embed: nn.Parameter(1, 100, 1152) × 0.02, LEARNABLE, max_len=100.
  4.  pos_embed + subject_embed added AFTER the 2 Hz → 1 Hz mean-pool.
  6.  subject_embed broadcast: subject_embed[idx].unsqueeze(1) over 100 timesteps.
  7.  Two-stage head: low_rank Linear(1152→2048, bias=False), brain_head
      Linear(2048→20484).
  13. n_subjects: 4 for sapient-1; ~75 for sapient-2.

This is the ONLY file in the codebase that may import from `transformers` for
type hints. The frozen encoders themselves are loaded in data/, never here —
that's the lineage guarantee (CODEBASE_MAP §4 audit point).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class SapientConfig:
    # Modality feature dims (set by upstream encoder).
    d_video: int = 1280
    d_audio: int = 1024
    d_text:  int = 3072        # Llama-3.2-3B (sole text encoder for v1)

    # Transformer body
    hidden: int = 1152
    n_layers: int = 8
    n_heads: int = 8
    ff_mult: int = 4           # ff = 4 × hidden = 4608
    dropout: float = 0.1

    # Output head (decision #7)
    low_rank_dim: int = 2048
    n_vertices: int = 20484    # fsaverage5 × 2 hemispheres

    # Regularization (decisions #1, #2)
    modality_dropout: float = 0.3
    subject_dropout: float = 0.1

    # Sequence (decisions #3, #4, #5)
    max_stim_len: int = 200    # 100 s × 2 Hz
    max_fmri_len: int = 100    # 100 s × 1 Hz

    # Subjects (decision #13) — override per dataset
    n_subjects: int = 4

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "SapientConfig":
        """Construct from a loaded YAML config (utils.load_config(...))."""
        m = cfg.get("model", cfg)
        # Only known fields; YAML may carry extras (data:, train:, eval:, etc.)
        kwargs = {
            "d_video": m.get("d_video", cls.d_video),
            "d_audio": m.get("d_audio", cls.d_audio),
            "d_text":  m.get("d_text",  cls.d_text),
            "hidden":  m.get("hidden",  cls.hidden),
            "n_layers": m.get("n_layers", cls.n_layers),
            "n_heads":  m.get("n_heads",  cls.n_heads),
            "ff_mult":  m.get("ff_mult",  cls.ff_mult),
            "dropout":  m.get("dropout",  cls.dropout),
            "low_rank_dim": m.get("low_rank_dim", cls.low_rank_dim),
            "n_vertices":   m.get("n_vertices",   cls.n_vertices),
            "modality_dropout": m.get("modality_dropout", cls.modality_dropout),
            "subject_dropout":  m.get("subject_dropout",  cls.subject_dropout),
            "max_stim_len": m.get("max_stim_len", cls.max_stim_len),
            "max_fmri_len": m.get("max_fmri_len", cls.max_fmri_len),
            "n_subjects":   m.get("n_subjects",   cls.n_subjects),
        }
        return cls(**kwargs)


class SapientModel(nn.Module):
    """Trimodal brain encoder. Maps (B, 200, *) stimulus features → (B, 100, 20484) fMRI."""

    def __init__(self, cfg: SapientConfig):
        super().__init__()
        self.cfg = cfg

        # ---- Per-modality projections ----
        self.proj_video = nn.Linear(cfg.d_video, cfg.hidden)
        self.proj_audio = nn.Linear(cfg.d_audio, cfg.hidden)
        self.proj_text  = nn.Linear(cfg.d_text,  cfg.hidden)

        # ---- Embeddings (post-pool, decision #4) ----
        # Learnable positional embedding initialized at std=0.02 (decision #3).
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.max_fmri_len, cfg.hidden) * 0.02
        )
        self.subject_embed = nn.Embedding(cfg.n_subjects, cfg.hidden)

        # ---- Transformer body ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,                  # pre-norm; more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # ---- Two-stage low-rank head (decision #7) ----
        self.low_rank   = nn.Linear(cfg.hidden, cfg.low_rank_dim, bias=False)
        self.brain_head = nn.Linear(cfg.low_rank_dim, cfg.n_vertices)

    def forward(
        self,
        video: torch.Tensor,   # (B, 200, d_video)
        audio: torch.Tensor,   # (B, 200, d_audio)
        text:  torch.Tensor,   # (B, 200, d_text)
        subject_idx: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        B = video.size(0)

        # 1. Project each modality to shared `hidden` space.
        v = self.proj_video(video)       # (B, 200, hidden)
        a = self.proj_audio(audio)
        t = self.proj_text(text)

        # 2. Modality dropout — training only, AFTER projections, BEFORE sum.
        if self.training:
            v, a, t = self._modality_dropout(v, a, t)

        # 3. Sum modalities.
        x = v + a + t                    # (B, 200, hidden)

        # 4. Downsample 2 Hz → 1 Hz via mean-pool over consecutive pairs.
        x = x.view(B, -1, 2, self.cfg.hidden).mean(dim=2)  # (B, 100, hidden)

        # 5. Add positional + subject embeddings (post-pool, decision #4).
        x = x + self.pos_embed                              # broadcast (1, 100, H)
        subj = self.subject_embed(subject_idx).unsqueeze(1) # (B, 1, H) — decision #6
        if self.training and torch.rand(1).item() < self.cfg.subject_dropout:
            subj = torch.zeros_like(subj)
        x = x + subj

        # 6. Transformer body.
        x = self.transformer(x)          # (B, 100, hidden)

        # 7. Two-stage brain head.
        x = self.low_rank(x)             # (B, 100, low_rank_dim)
        x = self.brain_head(x)           # (B, 100, n_vertices)
        return x

    def _modality_dropout(
        self,
        v: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Independently zero each modality with probability `modality_dropout`.

        Decision #1: applied AFTER projections, BEFORE sum. Each modality is
        an independent Bernoulli draw — this matches the spec §3.3 reference
        implementation and the TRIBE v1 paper.
        """
        p = self.cfg.modality_dropout
        if torch.rand(1).item() < p:
            v = torch.zeros_like(v)
        if torch.rand(1).item() < p:
            a = torch.zeros_like(a)
        if torch.rand(1).item() < p:
            t = torch.zeros_like(t)
        return v, a, t

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
