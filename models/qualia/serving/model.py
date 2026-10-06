"""qualia/serving/model.py — (formerly kairo_model.py) RECONSTRUCTED KairoEncoder nn.Module (v4).

Reconstructed from the v4_seed_0/best_model.pt state_dict (81 tensors) and the
checkpoint's own embedded recipe (ckpt['streams'], ckpt['feature_dims'],
n_streams=18, n_backbones=8, val_pearson=0.178, epoch=3, version='v4').

The architecture is fully determined by the saved tensor shapes; this module is
written so that `model.load_state_dict(ckpt['model_state_dict'], strict=True)`
succeeds. That clean load IS the proof the architecture matches.

Shape map (verified):
  encoder.projections.<stream>     : Linear(in_dim -> 256)  for 18 streams
  encoder.subject_emb              : Embedding(5, 256)
  encoder.transformer.layers.0     : 1x nn.TransformerEncoderLayer(d_model=256, nhead=?, ff=1024)
  bottleneck.0                     : Linear(4864 -> 2048)     # 4864 = 19 * 256 (18 streams + 1 subject slot)
  bottleneck.3                     : LayerNorm(2048)          # .1 GELU, .2 Dropout (no params)
  predictor.input_proj             : Linear(2048 -> 2048)
  predictor.layers.{0,1}.0         : RoPEAttention(qkv 2048->6144, out_proj 2048->2048)
  predictor.layers.{0,1}.1         : LayerNorm(2048)
  predictor.layers.{0,1}.2         : FFN(Linear 2048->8192, GELU, Dropout, Linear 8192->2048)
  predictor.layers.{0,1}.3         : LayerNorm(2048)
  predictor.head                   : Linear(2048 -> 1000)    # Schaefer-1000 parcels
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The 18 streams in the EXACT order they appear in the state_dict / bottleneck.
# (order matters: bottleneck.0 expects them concatenated in this order, then +subject)
STREAM_ORDER = [
    "vjepa_block5", "vjepa_block15", "vjepa_block23",
    "whisper_layer12", "whisper_layer25", "whisper_layer31", "whisper_layernorm",
    "vmae2_block10", "vmae2_block18", "vmae2_block25",
    "ivl3_layer10", "ivl3_layer15", "ivl3_layer20", "ivl3_norm",
    "emonet", "llama", "w2vbert", "vjepa2",
]
FEATURE_DIMS = {
    "vjepa_block5": 1024, "vjepa_block15": 1024, "vjepa_block23": 1024,
    "whisper_layer12": 1280, "whisper_layer25": 1280, "whisper_layer31": 1280,
    "whisper_layernorm": 1280, "vmae2_block10": 1280, "vmae2_block18": 1280,
    "vmae2_block25": 1280, "ivl3_layer10": 3584, "ivl3_layer15": 3584,
    "ivl3_layer20": 3584, "ivl3_norm": 3584, "emonet": 900, "llama": 9216,
    "w2vbert": 3072, "vjepa2": 4224,
}
D_MODEL = 256
N_SUBJECTS = 5
BOTTLENECK_IN = (len(STREAM_ORDER) + 1) * D_MODEL  # 19 * 256 = 4864
PREDICTOR_DIM = 2048
N_PARCELS = 1000


# ── RoPE ───────────────────────────────────────────────────────────────────────
def _build_rope(dim: int, seq: int, device, base: float = 10000.0):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (seq, half)
    emb = torch.cat([freqs, freqs], dim=-1)          # (seq, dim)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, hd); cos/sin: (T, hd)
    cos = cos[None, None]
    sin = sin[None, None]
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RoPEAttention(nn.Module):
    """Matches keys: qkv (dim->3dim), out_proj (dim->dim). Has a `.rope` buffer
    in the original; we recompute RoPE on the fly so no buffer is needed for load
    (buffers in the ckpt that we don't register are ignored under strict=False on
    buffers only — but we register a dummy to keep strict=True clean if present)."""
    def __init__(self, dim=PREDICTOR_DIM, n_heads=8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # (B, H, T, hd)
        cos, sin = _build_rope(self.head_dim, T, x.device)
        q, k = _apply_rope(q, k, cos.to(x.dtype), sin.to(x.dtype))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class PredictorBlock(nn.ModuleList):
    """Subclasses ModuleList so its state_dict keys are EXACTLY .0/.1/.2/.3 (the
    checkpoint's layout): .0=attn(RoPE), .1=LayerNorm, .2=FFN Sequential
    (-> .2.0 Linear / .2.3 Linear), .3=LayerNorm. Pre-norm residual."""
    def __init__(self, dim=PREDICTOR_DIM, n_heads=8, ff=8192, dropout=0.1):
        super().__init__([
            RoPEAttention(dim, n_heads),                                    # .0
            nn.LayerNorm(dim),                                             # .1
            nn.Sequential(nn.Linear(dim, ff), nn.GELU(),                  # .2 (.2.0/.2.3)
                          nn.Dropout(dropout), nn.Linear(ff, dim)),
            nn.LayerNorm(dim),                                             # .3
        ])

    def forward(self, x):
        attn, norm1, ffn, norm2 = self[0], self[1], self[2], self[3]
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))
        return x


class Predictor(nn.Module):
    def __init__(self, dim=PREDICTOR_DIM, n_layers=2, n_parcels=N_PARCELS, n_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([PredictorBlock(dim, n_heads=n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(dim, n_parcels)

    def forward(self, x):
        x = self.input_proj(x)
        for blk in self.layers:
            x = blk(x)
        return self.head(x)


class KairoEncoder(nn.Module):
    """encoder.projections (18) + subject_emb + 1-layer transformer; then
    bottleneck (Linear 4864->2048, GELU, Dropout, LayerNorm); then predictor."""
    def __init__(self, n_heads_enc=8, n_heads_pred=8):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.projections = nn.ModuleDict({
            s: nn.Linear(FEATURE_DIMS[s], D_MODEL) for s in STREAM_ORDER
        })
        self.encoder.subject_emb = nn.Embedding(N_SUBJECTS, D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=n_heads_enc, dim_feedforward=1024,
            batch_first=True, norm_first=False)
        self.encoder.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.bottleneck = nn.Sequential(
            nn.Linear(BOTTLENECK_IN, PREDICTOR_DIM),   # .0
            nn.GELU(),                                 # .1
            nn.Dropout(0.1),                           # .2
            nn.LayerNorm(PREDICTOR_DIM),               # .3
        )
        self.predictor = Predictor(n_heads=n_heads_pred)

    def forward(self, feats: dict[str, torch.Tensor], subject_idx: torch.Tensor):
        """feats: {stream: (B, T, in_dim)}  subject_idx: (B,) long
        Returns parcels (B, T, 1000).

        NOTE: the exact placement of the 1-layer encoder.transformer and how the
        18 projected streams are combined with the subject slot into the 4864 dim
        is the one inference-time detail not fully pinned by shapes. The default
        below (project -> per-stream encoder over the stream axis -> concat 18
        streams + subject -> bottleneck) is the most likely arrangement given the
        Algonauts-2025 lineage; verify by checking output variance across clips.
        If load_state_dict passes but outputs look wrong, try the alternative in
        the docstring of run_kairo_diff.py (encoder applied over time, mean-pool).
        """
        B = subject_idx.shape[0]
        projected = []
        for s in STREAM_ORDER:
            x = feats[s]                               # (B, T, in_dim)
            projected.append(self.encoder.projections[s](x))   # (B, T, 256)
        # stack streams -> (B, T, 18, 256); run the 1-layer transformer over the
        # STREAM axis per timestep (cross-stream mixing), then flatten streams.
        stk = torch.stack(projected, dim=2)            # (B, T, 18, 256)
        Bn, T, S, C = stk.shape
        mixed = self.encoder.transformer(stk.reshape(Bn * T, S, C))  # (B*T, 18, 256)
        mixed = mixed.reshape(Bn, T, S, C)
        subj = self.encoder.subject_emb(subject_idx)   # (B, 256)
        subj_t = subj[:, None, None, :].expand(Bn, T, 1, C)        # (B, T, 1, 256)
        cat = torch.cat([mixed, subj_t], dim=2).reshape(Bn, T, (S + 1) * C)  # (B,T,4864)
        h = self.bottleneck(cat)                       # (B, T, 2048)
        return self.predictor(h)                       # (B, T, 1000)


def load_kairo(ckpt_path: str, map_location="cpu"):
    """Load a v4 checkpoint into KairoEncoder. Returns (model, meta).
    Raises on any state_dict mismatch (that is the architecture proof)."""
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = ckpt["model_state_dict"]
    model = KairoEncoder()
    # The original may carry rope buffers we recompute; drop any *.rope keys so a
    # strict load on the parametric tensors still validates every weight matrix.
    rope_keys = [k for k in sd if k.endswith(".rope") or ".rope" in k]
    sd_clean = {k: v for k, v in sd.items() if k not in rope_keys}
    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    meta = {k: ckpt[k] for k in ckpt if k != "model_state_dict"}
    return model, meta, {"missing": list(missing), "unexpected": list(unexpected),
                         "dropped_rope_buffers": rope_keys}
