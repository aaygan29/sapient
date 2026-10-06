"""Mary adapter — the 8-step ORCLE-Nano trainable adapter (~35M params).

Faithful to the ORCLE-Nano whitepaper. See
`/Users/robertgutierrez/Desktop/sapient-research/Mary-Model-Insights/01-orcle-nano-architecture.md`
§3 (8-step adapter table) and `04-reimplementation-guide.md` §3 (module specs, defaults
for ambiguous bits flagged with ⚠️). Interface boundary is `mary/CONTRACTS.md` §2/§6.

The 8 steps (operating on per-stream tensors `(B, T_2Hz, D_m)`):

  1. Per-stream FFN          D_m → 768 (GELU + LayerNorm). One per stream.
  2. Per-stream HRF conv      depthwise Conv1d(768, 768, k=5, groups=768), padding=2.
  3. Modality dropout         zero entire streams, p=0.15, TRAINING ONLY.
  4. Fusion Transformer       1-layer pre-norm + learned modality embeddings + cross-attn.
  5. Attentive temporal pool  learned queries (one per TR) cross-attend → 2 Hz → 1 Hz / T_TR.
  6. Demographic slot         additive (age bucket, sex) embedding, per TR token.
  7. Prediction Transformer   2-layer pre-norm + RoPE on the time axis.
  8. Vertex head              group linear + per-subject linear (see `head.py`).

Frozen backbones are NEVER loaded here — that is the lineage guarantee
(CONTRACTS.md §7). This module only ever sees the cached `(T_2Hz, D_m)` features.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

# Canonical stream ordering and native feature dims (CONTRACTS.md §2, LOCKED).
STREAM_DIMS: dict[str, int] = {
    "slowfast": 2304,
    "qwen_vl": 4096,   # Qwen3-VL-8B real hidden size (paper's 3584 = Qwen2-VL-7B); extractor outputs 4096
    "beats": 768,
    "whisper": 1280,
    "qwen_ctx": 4096,
    "got_ocr": 768,
}
STREAM_ORDER: tuple[str, ...] = tuple(STREAM_DIMS.keys())


# ---------------------------------------------------------------------------
# Step 1 + 2 — per-stream FFN and depthwise HRF conv
# ---------------------------------------------------------------------------
class StreamEncoder(nn.Module):
    """Per-stream FFN (step 1) + depthwise HRF conv (step 2) for ONE stream.

    Step 1: `Linear(D_m, d_model) → GELU → LayerNorm`.
    Step 2: `Conv1d(d_model, d_model, kernel_size=5, groups=d_model, padding=2)` over
    the time axis — depthwise, one learned ~5 s hemodynamic kernel per channel,
    length-preserving (padding=2). Flagged default per guide §3 step 2.
    """

    def __init__(self, d_in: int, d_model: int, hrf_kernel_size: int = 5):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.hrf_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=hrf_kernel_size,
            groups=d_model,
            padding=hrf_kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_2Hz, D_m)
        x = self.norm(self.act(self.proj(x)))  # (B, T_2Hz, d_model)
        # depthwise conv over time → channels-first
        x = x.transpose(1, 2)                   # (B, d_model, T_2Hz)
        x = self.hrf_conv(x)                    # (B, d_model, T_2Hz)
        x = x.transpose(1, 2)                   # (B, T_2Hz, d_model)
        return x


# ---------------------------------------------------------------------------
# Step 4 — Fusion Transformer (1-layer pre-norm, learned modality embeds, cross-attn)
# ---------------------------------------------------------------------------
class FusionTransformer(nn.Module):
    """Fuse the per-stream tensors at each 2 Hz timestep via cross-attention.

    A learned modality embedding is added to each stream. Streams are stacked along
    a "modality" axis (S) and a pre-norm Transformer encoder layer cross-attends over
    that axis at every timestep. We then mean-pool over present streams to recombine
    into one fused `(B, T_2Hz, d_model)` sequence.

    Modality dropout (step 3) is realized via `key_padding_mask`: dropped/absent streams
    are masked out of attention AND excluded from the recombination mean (so attention is
    renormalized over remaining streams, per guide §3 step 3). #heads default 8 (flagged).
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        n_layers: int = 1,
    ):
        super().__init__()
        self.modality_embed = nn.Parameter(torch.randn(n_streams, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        streams: torch.Tensor,          # (B, T, S, d_model) — stacked along modality axis
        stream_mask: torch.Tensor,      # (B, S) bool — True = present/active stream
    ) -> torch.Tensor:
        B, T, S, D = streams.shape
        # Add learned modality embeddings (broadcast over batch + time).
        x = streams + self.modality_embed.view(1, 1, S, D)
        # Fold (B, T) into the batch dim so attention is over the S (modality) axis only.
        x = x.reshape(B * T, S, D)                         # (B*T, S, D)
        # key_padding_mask: True = IGNORE. Expand stream_mask over time.
        kpm = (~stream_mask).unsqueeze(1).expand(B, T, S).reshape(B * T, S)
        # Guard fully-masked rows (no active streams) — keep them unmasked to avoid NaNs;
        # those rows are zeroed out by the recombination weights below anyway.
        all_masked = kpm.all(dim=1)
        kpm = kpm.clone()
        kpm[all_masked] = False
        x = self.encoder(x, src_key_padding_mask=kpm)      # (B*T, S, D)
        x = x.reshape(B, T, S, D)
        # Recombine: mean over PRESENT streams only (renormalized).
        w = stream_mask.float().view(B, 1, S, 1)           # (B, 1, S, 1)
        denom = w.sum(dim=2).clamp(min=1.0)                # (B, 1, 1)
        fused = (x * w).sum(dim=2) / denom                 # (B, T, D)
        return fused


# ---------------------------------------------------------------------------
# Step 5 — Attentive temporal pooling (learned queries, 2 Hz → T_TR)
# ---------------------------------------------------------------------------
class AttentiveTemporalPool(nn.Module):
    """Resample the 2 Hz fused sequence to the fMRI TR grid via learned queries.

    `T_TR` learned query vectors cross-attend to the fused `(B, T_2Hz, d_model)` sequence,
    producing `(B, T_TR, d_model)`. This is the ORCLE-faithful replacement for the family's
    mean-pool (CONTRACTS.md §2). #heads default 8.
    """

    def __init__(self, d_model: int, max_fmri_len: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(max_fmri_len, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, t_tr: int) -> torch.Tensor:
        # x: (B, T_2Hz, d_model)
        B = x.size(0)
        q = self.queries[:t_tr].unsqueeze(0).expand(B, t_tr, -1)  # (B, T_TR, d_model)
        out, _ = self.attn(q, x, x)                               # (B, T_TR, d_model)
        return self.norm(out)


# ---------------------------------------------------------------------------
# Step 6 — Demographic slot (additive conditioning on age bucket + sex)
# ---------------------------------------------------------------------------
class DemographicSlot(nn.Module):
    """Small additive embedding of (age bucket, sex), added to every TR token.

    age_bucket ∈ [0, n_age_buckets), sex ∈ [0, n_sex). When demographics are unknown
    the caller passes bucket 0 / sex 0 (a learned "unknown" slot). ~7K params.
    """

    def __init__(self, d_model: int, n_age_buckets: int = 8, n_sex: int = 3):
        super().__init__()
        self.age_embed = nn.Embedding(n_age_buckets, d_model)
        self.sex_embed = nn.Embedding(n_sex, d_model)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T_TR, d_model)
        age_bucket: Optional[torch.Tensor],    # (B,) long or None
        sex: Optional[torch.Tensor],           # (B,) long or None
    ) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        if age_bucket is None:
            age_bucket = torch.zeros(B, dtype=torch.long, device=device)
        if sex is None:
            sex = torch.zeros(B, dtype=torch.long, device=device)
        slot = self.age_embed(age_bucket) + self.sex_embed(sex)  # (B, d_model)
        return x + slot.unsqueeze(1)                              # broadcast over T_TR


# ---------------------------------------------------------------------------
# Step 7 — Prediction Transformer with RoPE
# ---------------------------------------------------------------------------
def _build_rope_cache(t: int, head_dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embedding.

    Returns (cos, sin) each of shape (t, head_dim). Standard RoPE: rotate pairs of
    channels by position-dependent angles. head_dim must be even.
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)            # (t, half)
    emb = torch.cat([freqs, freqs], dim=-1)       # (t, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embedding on the time axis."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = _build_rope_cache(T, self.head_dim, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )                                              # (B, n_heads, T, head_dim)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        return self.out(attn)


class PredictionLayer(nn.Module):
    """Pre-norm Transformer encoder layer with RoPE self-attention + GELU FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class PredictionTransformer(nn.Module):
    """2-layer pre-norm Transformer with RoPE for temporal dependencies across TRs."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [PredictionLayer(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
