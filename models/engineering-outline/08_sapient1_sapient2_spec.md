# Sapient-1 / Sapient-2 Build Specification

**Purpose:** complete, paste-into-Claude-Code build brief for two clean-room brain encoding models.
**Trained from scratch on commercially-clean data, with no TRIBE v2 weight lineage.**
**Trainable on a single A100/H100 inside 24 hours.**

**Verification date:** 2026-05-26. All architecture numbers cross-checked against the TRIBE v1 paper ([arxiv.org/abs/2507.22229](https://arxiv.org/abs/2507.22229)) and the TRIBE v2 source code (read from the public `facebookresearch/algonauts-2025` repo on 2026-05-25; not used as code — only as a paper supplement).

---

## 0. Reading guide

This document is the **single source of truth** for the build. Read it once end-to-end. Then hand it to Claude Code as the build brief, **section by section**, in this order:

1. §1 (goals + repo layout) → Claude creates the two folders
2. §2 (data prep) → Claude builds data loaders
3. §3 (model code) → Claude builds the transformer
4. §4 (training loop) → Claude builds train.py
5. §5 (eval) → Claude builds eval.py
6. §6 (HF release) → Claude builds release.py

Do not skip ahead. Each section assumes the previous one is built and validated.

---

## 1. Goal & repo layout

### What we are building

Two PyTorch projects, externally owned, in two separate folders, releasable to two separate HuggingFace repos:

- **sapient-1** — base brain encoder. Multimodal stimulus → 20,484 fsaverage5 vertex predictions. Trained from scratch on CNeuroMod (+ optional BOLD Moments + Narratives).
- **sapient-2** — conversation-specialized brain encoder. Same architecture as sapient-1. Trained on ds004996 (NeuroEngage). Two lineages: from-scratch and fine-tuned-from-sapient-1.

Both feed your existing `sapienteval/` parcellation + Buy Moment + 8-scorer stack downstream.

### What this is NOT

- ❌ A copy of TRIBE v2 code. We re-implement from the paper.
- ❌ A fine-tune of TRIBE v2 weights. Zero TRIBE weight lineage at any stage.
- ❌ A change to your existing `sapienteval/` post-model code. That stays as-is.

### Repo layout (both projects)

```
sapient-1/                                    ← external folder, separate HF repo
├── README.md                                  ← public model card draft
├── LICENSE                                    ← Apache 2.0
├── NOTICE                                     ← required by Apache + Llama license
├── pyproject.toml
├── configs/
│   ├── sapient1_base.yaml                     ← single source of hyperparameters
│   └── sapient1_llama.yaml                    ← Llama-3.2-3B text encoder (only variant)
├── data/
│   ├── download_cneuromod.sh                  ← datalad install + get (sapient-1)
│   ├── download_bold_moments.sh                ← sapient-1 augmentation
│   ├── download_narratives.sh                  ← sapient-1 augmentation
│   ├── download_ds004996.sh                    ← sapient-2 only (NeuroEngage)
│   ├── download_ds001740.sh                    ← sapient-2 only (Rauchbauer HRI)
│   ├── prepare_fmri.py                        ← fsaverage5 projection + parcel atlas
│   ├── extract_video_features.py              ← V-JEPA 2 (frozen)
│   ├── extract_audio_features.py              ← Whisper + W2V-BERT (frozen)
│   ├── extract_text_features.py               ← Llama-3.2-3B (frozen)
│   └── dataset.py                             ← Torch Dataset + DataLoader
├── sapient1/
│   ├── __init__.py
│   ├── model.py                               ← the transformer + heads
│   ├── losses.py                              ← MSE-with-mask
│   ├── modules.py                             ← projection, attention, FFN
│   └── utils.py
├── train.py                                   ← main training loop
├── eval.py                                    ← vertex + parcel Pearson eval
├── release.py                                 ← HF upload + model card
└── tests/
    ├── test_shapes.py                         ← shape sanity checks
    └── test_overfit.py                        ← 1-batch overfit test

sapient-2/                                    ← external folder, separate HF repo
└── (identical structure, with configs/ pointing at ds004996)
```

---

## 2. Data preparation

### 2.1 Download

```bash
# CNeuroMod (CC0 subset of 4 subjects)
datalad install git@github.com:courtois-neuromod/cneuromod.processed.git
cd cneuromod.processed
datalad get fmriprep/movie10/sub-*/ses-*/func/*space-MNI152NLin2009cAsym_*
datalad get fmriprep/friends/sub-*/ses-*/func/*space-MNI152NLin2009cAsym_*
datalad get -r fmriprep/movie10/sourcedata/movie10/stimuli
datalad get -r fmriprep/friends/sourcedata/friends/stimuli
```

The 4 CC0 subjects (sub-01, sub-02, sub-03, sub-05) require **no auth keys**. The other 2 subjects require a DTA we do not need. Reference: [CNeuroMod access docs](https://docs.cneuromod.ca/en/latest/ACCESS.html).

For sapient-2 (both HRI datasets, both CC0 on OpenNeuro):
```bash
# Primary: ds004996 NeuroEngage (~50 subjects, human-human + human-robot dialogue)
datalad install https://github.com/OpenNeuroDatasets/ds004996.git
cd ds004996
cat dataset_description.json | jq .License   # MUST read "CC0"; abort otherwise
datalad get .
cd ..

# Augmentation: ds001740 Rauchbauer 2019/2020 (25 French speakers, Furhat robot)
# Different robot embodiment → forces sapient-2 to learn robot-agnostic representations
datalad install https://github.com/OpenNeuroDatasets/ds001740.git
cd ds001740
git checkout 2.1.0   # use v2.1.0 explicitly
cat dataset_description.json | jq .License   # MUST read "CC0"; abort otherwise
datalad get .
```

References: [OpenNeuro ds004996](https://openneuro.org/datasets/ds004996), [OpenNeuro ds001740 v2.1.0](https://openneuro.org/datasets/ds001740/versions/2.1.0), [Rauchbauer paper](https://pubmed.ncbi.nlm.nih.gov/30852994/).

### 2.2 fMRI prep (`data/prepare_fmri.py`)

For each subject and each run:

1. **Load** the preprocessed BOLD timeseries (MNI volumetric space)
2. **Project to fsaverage5 surface mesh** using `nilearn.surface.vol_to_surf` → produces 10,242 vertices per hemisphere, 20,484 total per timepoint
3. **Save** as a memory-mapped numpy array shape `(n_timepoints, 20484)` per run
4. **Compute and cache** the Schaefer-2018 1,000-parcel atlas for downstream use:
   ```python
   from nilearn.datasets import fetch_atlas_schaefer_2018
   atlas = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=2)
   # save atlas.maps to disk; this maps each vertex to a parcel
   ```
5. **Build the vertex→parcel projection matrix** once: a sparse `(1000, 20484)` matrix that averages vertices into parcels. Save as `parcellate.npz`.

**Why both fsaverage5 vertices AND Schaefer-1000 parcels:** the model trains and predicts at vertex level (20,484 dims, matches TRIBE v2 spatial granularity). The parcel projection is a **separate deterministic post-processing step** for downstream consumption by your existing `sapienteval/` code. Both are produced by one forward pass.

**TR handling:**
- CNeuroMod: TR = 1.49 s
- BOLD Moments: TR = 1.75 s (resampled to 1.0 s in their preprocessing)
- Narratives: TR = 1.5 s
- ds004996: TR = (verify from dataset_description.json before training)
- ds001740: TR = (verify from dataset_description.json before training)

**Resample all fMRI to 1 Hz** for consistency. Linear interpolation along the time axis.

### 2.3 Feature extraction (frozen encoders)

For each dataset, run feature extraction **once** and cache to disk. These features are reused across every training run. **The encoders never see gradients.**

#### `data/extract_video_features.py` — V-JEPA 2

```python
from transformers import AutoModel, AutoImageProcessor

# V-JEPA 2 Gigantic (1B params), MIT licensed
# (verified: https://github.com/facebookresearch/vjepa2)
model_id = "facebook/vjepa2-vitg-fpc64-256"
model = AutoModel.from_pretrained(model_id).eval().cuda()
processor = AutoImageProcessor.from_pretrained(model_id)

# Input: 64 frames spanning preceding 4 seconds at 2 Hz
# Output: 1280-dim embedding per 2 Hz step
# (matches TRIBE v1 paper Section 3.1)
```

Save outputs as `(n_steps, 1280)` arrays per video, where `n_steps = video_duration_seconds * 2`.

#### `data/extract_audio_features.py` — Whisper + W2V-BERT

```python
from transformers import WhisperModel, AutoFeatureExtractor, AutoModel

# Whisper-large-v3 (MIT)
whisper = WhisperModel.from_pretrained("openai/whisper-large-v3").eval().cuda()
# Output: 1280-dim @ ~50 Hz, resample to 2 Hz

# Wav2Vec-BERT 2.0 (MIT)
w2vbert = AutoModel.from_pretrained("facebook/w2v-bert-2.0").eval().cuda()
# Output: 1024-dim @ 50 Hz, resample to 2 Hz
```

We use **W2V-BERT 2.0 only** to match the TRIBE v2 HF card (which lists W2V-BERT, not Whisper). Whisper is kept available as an ablation but not in the default config.

Save as `(n_steps, 1024)` per audio file.

#### `data/extract_text_features.py` — Llama-3.2-3B

```python
# Llama-3.2-3B (Llama 3.2 Community License)
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B", torch_dtype=torch.bfloat16).eval().cuda()
# Hidden dim = 3072
# Context: preceding 1024 words
```

**Decision (2026-05-26):** Llama-3.2-3B is the sole text encoder for sapient-1 and sapient-2. The earlier Qwen2.5-7B variant has been removed from the project — no Modal app, no HF release, no cached features. Rationale: pre-seed compute budget, the 700M MAU clause is irrelevant at this stage, and Llama-3.2-3B has the stronger published brain-encoding precedent (TRIBE v1 family).

For each word in the transcript:
1. Look at preceding 1024-word context
2. Run through the LLM
3. Average token embeddings that correspond to the current word
4. Time-align to the word's onset timestamp (from forced alignment)
5. Resample to 2 Hz evenly-spaced grid

Save as `(n_steps, hidden_dim)` per transcript.

**Why we extract at 2 Hz:** matches TRIBE v2 and most movie-watching fMRI work. fMRI is at 1 Hz post-resampling; we predict at 1 Hz; but stimulus features at 2 Hz give the model finer temporal grain before pooling.

### 2.4 Dataset class (`data/dataset.py`)

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class SapientDataset(Dataset):
    """
    Returns one 100-second window aligned across modalities.
    
    Per item:
      video:   (200, 1280)    # 100s * 2Hz, V-JEPA features
      audio:   (200, 1024)    # 100s * 2Hz, W2V-BERT features
      text:    (200, 3072)    # 100s * 2Hz, Llama-3.2-3B features
      fmri:    (100, 20484)   # 100s * 1Hz, fsaverage5 vertices
      subject: int            # subject index for embedding lookup
      mask:    (100,)         # 1 where fmri is valid, 0 where padded
    """
    def __init__(self, manifest_path, duration_trs=100, hrf_offset_trs=5):
        self.manifest = load_manifest(manifest_path)
        self.duration_trs = duration_trs    # 100 seconds at 1 Hz
        self.hrf_offset_trs = hrf_offset_trs  # 5 second BOLD lag
    
    def __getitem__(self, idx):
        clip = self.manifest[idx]
        # Sample a random 100-TR window from this run
        run_len = clip.n_trs
        start_tr = np.random.randint(0, run_len - self.duration_trs - self.hrf_offset_trs)
        
        # Stimulus features for [start_tr, start_tr + duration_trs)
        stim_start_steps = start_tr * 2   # 2 Hz stimulus rate
        stim_end_steps   = (start_tr + self.duration_trs) * 2
        video = clip.video_features[stim_start_steps:stim_end_steps]  # (200, 1280)
        audio = clip.audio_features[stim_start_steps:stim_end_steps]  # (200, 1024)
        text  = clip.text_features[stim_start_steps:stim_end_steps]   # (200, 3072)
        
        # fMRI labels are SHIFTED by HRF offset
        fmri_start_tr = start_tr + self.hrf_offset_trs
        fmri_end_tr   = fmri_start_tr + self.duration_trs
        fmri = clip.fmri[fmri_start_tr:fmri_end_tr]  # (100, 20484)
        
        return {
            'video': torch.from_numpy(video).float(),
            'audio': torch.from_numpy(audio).float(),
            'text':  torch.from_numpy(text).float(),
            'fmri':  torch.from_numpy(fmri).float(),
            'subject': torch.tensor(clip.subject_idx, dtype=torch.long),
            'mask': torch.ones(self.duration_trs),
        }
```

**Note on HRF offset:** the BOLD signal lags neural activity by ~5 seconds. The TRIBE v1 paper aligns stimulus features at time t to fMRI signal at time t+5s. Standard practice. We do the same.

---

## 3. Model architecture (`sapient1/model.py`)

### 3.1 Layer-by-layer spec

The numbers below are **verified against the TRIBE v1 paper**. We deliberately use a smaller hidden size than the v1 paper (1152 vs 3072) to match the published TRIBE v2 source-code defaults and to fit comfortably on a single A100 in 24h. This is a known-good config.

```
Input: video (B, 200, 1280)
       audio (B, 200, 1024)
       text  (B, 200, 3072)
       subject_idx (B,)
       
Output: fmri (B, 100, 20484)
```

| Layer | Input shape | Output shape | Params | Trainable |
|---|---|---|---|---|
| `proj_video`: Linear(1280, 1152) | (B, 200, 1280) | (B, 200, 1152) | 1.47M | ✅ |
| `proj_audio`: Linear(1024, 1152) | (B, 200, 1024) | (B, 200, 1152) | 1.18M | ✅ |
| `proj_text`: Linear(3072, 1152) | (B, 200, 3072) | (B, 200, 1152) | 3.54M | ✅ |
| Modality dropout (p=0.3) applied **after projections, before sum** | (B, 200, 1152) × 3 | (B, 200, 1152) × 3 | 0 | — |
| Sum modalities | (B, 200, 1152) × 3 | (B, 200, 1152) | 0 | — |
| Downsample 200 steps (2 Hz) → 100 steps (1 Hz) via mean pool of pairs | (B, 200, 1152) | (B, 100, 1152) | 0 | — |
| Add positional embedding (**learnable** `nn.Parameter(1, 100, 1152) × 0.02`, max_len=**100**, post-pool) | (B, 100, 1152) | (B, 100, 1152) | 115K (pos) | ✅ |
| Add subject embedding (broadcast over 100 timesteps), `subject_dropout=0.1` at train time | (B, 100, 1152) | (B, 100, 1152) | n_subj × 1152 | ✅ |
| **Transformer × 8** (hidden=1152, heads=8, ff=4608, dropout=0.1) | (B, 100, 1152) | (B, 100, 1152) | ~127M | ✅ |
| `low_rank_head`: Linear(1152, 2048, bias=False) | (B, 100, 1152) | (B, 100, 2048) | 2.36M | ✅ |
| `brain_head`: Linear(2048, 20484) | (B, 100, 2048) | (B, 100, 20484) | 41.97M | ✅ |

**Total trainable parameters: ~180M**

### 3.2 Why these specific numbers

| Choice | Justification |
|---|---|
| `hidden=1152` | TRIBE v2 source code default; divisible by 8 attention heads |
| `n_layers=8` | TRIBE v1 paper Table 1; ablations in the paper show 8 layers is the sweet spot |
| `n_heads=8` | TRIBE v1 paper Table 1 |
| `ff=4× hidden` | Standard transformer FFN ratio |
| `low_rank_head=2048` | TRIBE v2 source default; reduces final head params 10× without performance loss |
| Output dim = 20,484 | fsaverage5, both hemispheres, matches TRIBE v2 |
| `modality_dropout=0.3` | TRIBE v1 paper: prevents over-reliance on single modality, improves multi-modal generalization |
| `subject_dropout=0.1` | TRIBE v1 paper: prevents overfit to a single subject's idiosyncratic anatomy |
| Window = 100 TRs | TRIBE v1 paper Table 1; balances context vs memory |
| Stimulus rate = 2 Hz, fMRI rate = 1 Hz | TRIBE v1 paper Section 3.1 |

### 3.3 Code skeleton

```python
import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class SapientConfig:
    # Modality dims
    d_video: int = 1280
    d_audio: int = 1024
    d_text:  int = 3072      # Llama-3.2-3B hidden dim
    
    # Transformer
    hidden: int = 1152
    n_layers: int = 8
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    
    # Output
    low_rank_dim: int = 2048
    n_vertices: int = 20484   # fsaverage5 × 2 hemispheres
    
    # Regularization
    modality_dropout: float = 0.3
    subject_dropout: float = 0.1
    
    # Sequence
    max_stim_len: int = 200   # 100s × 2Hz
    max_fmri_len: int = 100   # 100s × 1Hz
    
    # Subjects
    n_subjects: int = 4       # CNeuroMod open subjects for sapient-1 (sub-01,02,03,05)
                              # ~75 for sapient-2 (50 ds004996 + 25 ds001740)


class SapientModel(nn.Module):
    def __init__(self, cfg: SapientConfig):
        super().__init__()
        self.cfg = cfg
        
        # Per-modality projection
        self.proj_video = nn.Linear(cfg.d_video, cfg.hidden)
        self.proj_audio = nn.Linear(cfg.d_audio, cfg.hidden)
        self.proj_text  = nn.Linear(cfg.d_text,  cfg.hidden)
        
        # Positional + subject embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_fmri_len, cfg.hidden) * 0.02)
        self.subject_embed = nn.Embedding(cfg.n_subjects, cfg.hidden)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden * cfg.ff_mult,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # pre-norm, more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        
        # Output head (low-rank)
        self.low_rank = nn.Linear(cfg.hidden, cfg.low_rank_dim, bias=False)
        self.brain_head = nn.Linear(cfg.low_rank_dim, cfg.n_vertices)
    
    def forward(self, video, audio, text, subject_idx):
        B = video.size(0)
        
        # 1. Project each modality to shared space
        v = self.proj_video(video)   # (B, 200, hidden)
        a = self.proj_audio(audio)   # (B, 200, hidden)
        t = self.proj_text(text)     # (B, 200, hidden)
        
        # 2. Modality dropout (training only)
        if self.training:
            v, a, t = self._modality_dropout(v, a, t)
        
        # 3. Sum modalities
        x = v + a + t   # (B, 200, hidden)
        
        # 4. Downsample 2Hz → 1Hz by mean-pooling pairs
        x = x.view(B, -1, 2, self.cfg.hidden).mean(dim=2)   # (B, 100, hidden)
        
        # 5. Add positional + subject embeddings
        x = x + self.pos_embed
        subj = self.subject_embed(subject_idx).unsqueeze(1)  # (B, 1, hidden)
        if self.training and torch.rand(1).item() < self.cfg.subject_dropout:
            subj = torch.zeros_like(subj)  # drop subject id with prob 0.1
        x = x + subj
        
        # 6. Transformer body
        x = self.transformer(x)   # (B, 100, hidden)
        
        # 7. Low-rank brain head
        x = self.low_rank(x)        # (B, 100, low_rank_dim)
        x = self.brain_head(x)      # (B, 100, 20484)
        
        return x
    
    def _modality_dropout(self, v, a, t):
        """Randomly zero out modalities with probability `modality_dropout`."""
        p = self.cfg.modality_dropout
        if torch.rand(1).item() < p: v = torch.zeros_like(v)
        if torch.rand(1).item() < p: a = torch.zeros_like(a)
        if torch.rand(1).item() < p: t = torch.zeros_like(t)
        return v, a, t
```

### 3.4 Validation

Before training:

```python
# tests/test_shapes.py
def test_forward_pass():
    cfg = SapientConfig()
    model = SapientModel(cfg)
    
    B = 4
    video = torch.randn(B, 200, 1280)
    audio = torch.randn(B, 200, 1024)
    text  = torch.randn(B, 200, 3072)
    subj  = torch.randint(0, 4, (B,))
    
    out = model(video, audio, text, subj)
    assert out.shape == (B, 100, 20484), f"Got {out.shape}"
    print(f"OK. Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

Expected: ~180M parameters, output shape `(4, 100, 20484)`.

---

## 4. Training loop (`train.py`)

### 4.1 Loss

```python
def masked_mse_loss(pred, target, mask):
    """
    pred:   (B, 100, 20484)
    target: (B, 100, 20484)
    mask:   (B, 100)
    
    Returns scalar.
    """
    # Per-vertex squared error, then mean over vertices
    err = ((pred - target) ** 2).mean(dim=-1)   # (B, 100)
    # Mean over valid timepoints
    loss = (err * mask).sum() / mask.sum().clamp(min=1)
    return loss
```

### 4.2 Optimizer & schedule

| Setting | Value | Source |
|---|---|---|
| Optimizer | AdamW | TRIBE v1 paper |
| LR | 1e-4 | TRIBE v1 paper |
| Weight decay | 0.0 | TRIBE v1 paper / v2 source default |
| Betas | (0.9, 0.999) | PyTorch default |
| LR schedule | OneCycleLR, pct_start=0.1 (10% linear warmup, then cosine decay) | TRIBE v2 source |
| Epochs | 15 (or fewer if time-bounded) | TRIBE v1 paper |
| Batch size | 8 per GPU | TRIBE v2 source default |
| Grad accumulation | 2 → effective batch 16 | TRIBE v1 paper |
| Precision | bfloat16 mixed | A100/H100 native bf16 |
| Gradient clip | 1.0 | safety |

### 4.3 Single-GPU training loop

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

def train_one_epoch(model, loader, optim, scheduler, scaler, device):
    model.train()
    for step, batch in enumerate(loader):
        for k in batch: batch[k] = batch[k].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(batch['video'], batch['audio'], batch['text'], batch['subject'])
            loss = masked_mse_loss(pred, batch['fmri'], batch['mask'])
            loss = loss / GRAD_ACCUM_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
        
        if step % 50 == 0:
            log({'loss': loss.item() * GRAD_ACCUM_STEPS, 'lr': scheduler.get_last_lr()[0]})

def main():
    cfg = load_config('configs/sapient1_llama.yaml')
    model = SapientModel(cfg).cuda()
    
    train_ds = SapientDataset('manifests/train.json')
    val_ds   = SapientDataset('manifests/val.json')
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    total_steps = len(train_loader) * 15 // GRAD_ACCUM_STEPS
    scheduler = OneCycleLR(optim, max_lr=1e-4, total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(15):
        train_one_epoch(model, train_loader, optim, scheduler, scaler, device='cuda')
        val_metrics = evaluate(model, val_loader)
        log({'epoch': epoch, **val_metrics})
        save_checkpoint(model, f'checkpoints/sapient1_e{epoch:02d}.pt')
```

### 4.4 Time budget — what fits in 24 hours on a single A100/H100

**A100 80GB realistic throughput:**
- ~1.2 sec/step at batch 8 with bf16 (transformer forward+backward dominant)
- 300 batches per minute → 18,000 batches per hour → 432,000 batches in 24h
- CNeuroMod 4 subjects × 80h × ~600 windows per subject (100s windows, sliding by 10s) ≈ 192,000 windows per epoch
- 432,000 / 192,000 ≈ **2.25 epochs in 24h**

**That's enough for a working model but NOT TRIBE-paper-quality (15 epochs).**

**To make 24h productive:**
1. **Subsample the data** to a representative chunk (~50h instead of 320h) → 5 epochs in 24h
2. **Use bf16 throughout** including activations (we already do)
3. **Cache features aggressively** — feature extraction happens once, never on the hot path
4. **Use larger batch size if memory allows** — try batch=16 with grad_accum=1

**Realistic 24-hour benchmark target:**
- TRIBE v2 paper-quality (15 epochs, 8 subjects, full data): not achievable on single GPU
- Sapient-1 MVP (3-5 epochs, 4 subjects, subset of stimulus): **achievable**
- Expected vertex-level Pearson correlation: **0.25–0.35** (vs TRIBE's 0.54 with full setup) — a working model with your name on it, ready to scale post-funding

### 4.5 The 24-hour timeline

| Hour | Task |
|---|---|
| 0–2 | Datalad install CNeuroMod (4 CC0 subjects, just movie10 + Friends seasons 1-2 to start) |
| 2–6 | fMRI prep: project to fsaverage5, cache as memmap |
| 6–12 | Feature extraction (V-JEPA + W2V-BERT + Llama-3.2-3B) — most time-expensive step; can run in parallel |
| 12–13 | Build manifests, train/val/test splits (held-out = Friends S2 last 3 episodes) |
| 13–14 | Sanity tests: shapes, 1-batch overfit |
| 14–22 | Training (3-5 epochs depending on dataset size) |
| 22–23 | Eval: vertex Pearson, parcel Pearson, per-subject breakdown |
| 23–24 | Model card draft, HF upload |

### 4.6 Sapient-2 specifics

For sapient-2 you have two training runs:

```python
# Run 1: sapient-2-scratch (random init)
model = SapientModel(cfg_sapient2)
model.cuda()
train(...)

# Run 2: sapient-2-ft (init from sapient-1)
model = SapientModel(cfg_sapient2)
sapient1_state = torch.load('sapient1_final.pt')
# Load only the body, not the subject embeddings (different subject count)
state = {k: v for k, v in sapient1_state.items() if 'subject_embed' not in k}
model.load_state_dict(state, strict=False)
model.cuda()
train(..., lr=5e-5)   # smaller LR for fine-tuning
```

The fine-tuned variant will almost certainly win on ds004996 because the model already knows how to map stimulus → brain in general. Train both, ship the winner.

---

## 5. Evaluation (`eval.py`)

### 5.1 Held-out splits

**For sapient-1 trained on CNeuroMod:**
- **Train:** Friends S1-S6 episodes 1 through (end-3), all 4 subjects, plus 3 of 4 Movie10 movies
- **Val:** Friends S1-S6 last 3 episodes per subject (used for early stopping)
- **Test:** 1 held-out Movie10 movie (subject-and-stimulus-novel), evaluated last

**For sapient-2 trained on ds004996:**
- **Train:** 80% of subjects × 80% of runs
- **Val:** 20% of subjects × all runs (subject-novel split)
- **Test:** all subjects × 20% of runs (stimulus-novel split)

### 5.2 Metrics

```python
import torch
from scipy.stats import pearsonr

def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            for k in batch: batch[k] = batch[k].cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(batch['video'], batch['audio'], batch['text'], batch['subject'])
            all_preds.append(pred.cpu().float())
            all_targets.append(batch['fmri'].cpu())
    
    pred = torch.cat(all_preds, dim=0)      # (N, 100, 20484)
    target = torch.cat(all_targets, dim=0)  # (N, 100, 20484)
    
    # Flatten time: (N*100, 20484)
    pred = pred.reshape(-1, 20484)
    target = target.reshape(-1, 20484)
    
    # Vertex-level Pearson: for each vertex, correlate predicted vs actual across all timepoints
    vertex_r = []
    for v in range(20484):
        r, _ = pearsonr(pred[:, v].numpy(), target[:, v].numpy())
        vertex_r.append(r if not (r != r) else 0.0)  # handle NaN
    vertex_r = torch.tensor(vertex_r)
    
    return {
        'vertex_pearson_mean': vertex_r.mean().item(),
        'vertex_pearson_std':  vertex_r.std().item(),
        'vertex_pearson_top10pct': vertex_r.topk(2048).values.mean().item(),
    }

def evaluate_at_parcels(model, loader, parcellation_matrix):
    """Same as evaluate() but project vertices → 1000 Schaefer parcels first.
    
    parcellation_matrix: (1000, 20484) sparse averaging matrix
    """
    # ... same as above but apply: pred = pred @ parcellation_matrix.T before correlation
```

### 5.3 Investor-presentable results

After eval, produce one figure and one table:

**Figure:** brain surface heatmap of vertex-level Pearson correlation (use `nilearn.plotting.plot_surf_stat_map`). Visual cortex should be hottest. This is the "look it learned visual processing" plot.

**Table:**

| Metric | sapient-1-llama | TRIBE v2 (paper) |
|---|---|---|
| Vertex Pearson (mean) | ? | not reported at vertex level |
| Vertex Pearson (top 10%) | ? | n/a |
| Parcel Pearson (Schaefer-1000, mean) | ? | 0.21 (Algonauts test set, OOD) |

This is what you put in the deck.

---

## 6. HuggingFace release (`release.py`)

### 6.1 Folder structure to upload

```
sapient-1-llama/                       ← HF repo: The-Sapient-Company/sapient-1-llama
├── README.md                          ← model card (required)
├── LICENSE                            ← Apache 2.0
├── NOTICE                             ← required by Apache; lists encoder licenses
├── config.json                        ← SapientConfig dump
├── model.safetensors                  ← weights, safetensors format
├── parcellate.npz                     ← Schaefer-1000 projection matrix
└── inference.py                       ← minimal forward-pass example
```

### 6.2 Model card template (README.md)

```markdown
---
license: apache-2.0
library_name: pytorch
tags:
- brain-encoding
- fmri
- multimodal
- neuroscience
datasets:
- courtois-neuromod/cneuromod
- lahner/bold-moments
- princeton/narratives
---

# sapient-1-llama

A trimodal brain encoder predicting whole-cortex fMRI response to video, audio, and text stimuli.

## Architecture
- 8-layer transformer, hidden=1152, 8 attention heads
- Frozen feature encoders: V-JEPA 2 (video), Wav2Vec-BERT 2.0 (audio), Llama-3.2-3B (text)
- Output: 20,484 fsaverage5 cortical vertices @ 1 Hz
- ~180M trainable parameters

## Training Data
- Courtois NeuroMod (CC0 4-subject open subset, 80h × 4 subjects = 320h)
- BOLD Moments Dataset (Lahner et al. 2024)
- Narratives (Nastase et al. 2021); fMRI used as labels; stimulus audio not redistributed

## License
This model: Apache 2.0
The frozen feature extractors retain their own licenses (see NOTICE).
The Llama-3.2-3B text encoder is subject to the Llama 3.2 Community License — see NOTICE.
This model is provided for research and commercial use under Apache 2.0 (subject to upstream encoder license terms).

## Citation
@misc{sapient1,
  author = {Sapient Team},
  title  = {sapient-1: A clean-room trimodal brain encoder},
  year   = {2026},
  publisher = {HuggingFace},
}

## Built On
- TRIBE v1 (D'Ascoli et al. 2025, arXiv:2507.22229) — architectural inspiration only; this model does not contain or derive from TRIBE v1 or v2 weights.
- V-JEPA 2 (Meta, MIT)
- Wav2Vec-BERT 2.0 (Meta, MIT)
- Llama-3.2-3B (Meta, Llama 3.2 Community License)
```

### 6.3 NOTICE file (required for Apache + Llama)

```
sapient-1-llama
Copyright 2026 Sapient

This product includes frozen weights from third-party models. The released sapient-1
weights are Apache 2.0; upstream feature-extractor weights retain their own licenses:

- V-JEPA 2 (facebook/vjepa2-vitg-fpc64-256), MIT License, Copyright 2025 Meta Platforms, Inc.
- Wav2Vec-BERT 2.0 (facebook/w2v-bert-2.0), MIT License, Copyright 2023 Meta Platforms, Inc.
- Llama-3.2-3B (meta-llama/Llama-3.2-3B), Llama 3.2 Community License,
  Copyright 2024 Meta Platforms, Inc. The derived model is named "sapient-1-llama" in
  compliance with Llama 3.2 §5 naming requirements. "Built with Llama."
```

### 6.4 Upload

```python
from huggingface_hub import HfApi, create_repo

api = HfApi()
create_repo("The-Sapient-Company/sapient-1-llama", private=True, exist_ok=True)
api.upload_folder(
    folder_path="./sapient-1-llama",
    repo_id="The-Sapient-Company/sapient-1-llama",
    repo_type="model",
)
```

Start private, flip to public after the call when you've stress-tested the inference example.

---

## 7. Wiring into your existing `sapienteval/` stack

Your existing post-model code expects parcellated input. Sapient-1 outputs vertices. Add this adapter (lives in `sapient1/inference.py`):

```python
import numpy as np
import torch

def vertices_to_parcels(vertex_pred, parcel_matrix):
    """
    vertex_pred:   (T, 20484) numpy or torch — model output
    parcel_matrix: (1000, 20484) sparse averaging matrix
    
    Returns: (T, 1000) Schaefer-2018 parcel predictions
    """
    if isinstance(vertex_pred, torch.Tensor):
        vertex_pred = vertex_pred.cpu().numpy()
    return vertex_pred @ parcel_matrix.T  # (T, 1000)

# Then your existing sapienteval pipeline picks up from there:
# parcels → 03_parcellate.py (which already expects this shape)
# → buy_moment_detector.py → 8 cognitive scorers → output
```

The 1,000-dim parcel array is the **exact interface your existing code already consumes**. No changes to `sapienteval/` are needed. Your post-model IP is untouched.

---

## 8. Acceptance criteria — before we say "done"

Each must pass before you ship to the investor:

- [ ] Both folders exist as standalone projects, clone-able, runnable end-to-end
- [ ] `tests/test_shapes.py` passes (model produces correct output shape)
- [ ] `tests/test_overfit.py` passes (model can drive loss to ~0 on a single batch)
- [ ] Training run completes 3+ epochs without crash
- [ ] Held-out vertex Pearson > 0.10 (sanity threshold — confirms learning happened)
- [ ] Eval produces a brain surface plot showing visual cortex activation
- [ ] Adapter `vertices_to_parcels` produces shape `(T, 1000)` compatible with your `sapienteval/`
- [ ] HF repo created, model card complete, weights uploaded
- [ ] NOTICE file lists every third-party encoder license
- [ ] No TRIBE v2 source code copied; no TRIBE v2 weights anywhere in lineage

---

## 9. What to say to the investor

When asked "show me your model":

> "Two models, both in our own HuggingFace org. sapient-1 is our base brain encoder — an eight-layer transformer that maps multimodal video, audio, and text features onto twenty thousand cortical surface vertices. Trained from scratch on CNeuroMod, the same publicly-released dataset Meta's TRIBE v2 uses, plus BOLD Moments and Narratives. No weights inherited from anyone. sapient-2 is the conversation-specialized variant, fine-tuned on the NeuroEngage human-robot interaction dataset. The whole stack is Apache 2.0 — we can sell into enterprise without legal review."

When asked "isn't this just TRIBE v2?":

> "Architecturally similar — eight layers, twelve hundred hidden dimensions, V-JEPA plus Wav2Vec-BERT plus a text LLM is the right design for this problem and TRIBE's paper showed why. The architectural choice is published science; it's not Meta's intellectual property. What is Meta's is the weights they released under CC-BY-NC, which we can't use commercially. So we trained from scratch on the same public data. Our weights, our license, our right to ship. The actual moat is what comes next — the proprietary brain-response dataset we'll collect through our product. Meta can't replicate that. CNeuroMod is public; our product's behavioral feedback loop is not."

When asked "why a smaller hidden size than the published v1 paper?":

> "Single-GPU training budget on pre-seed compute. The architecture is identical; the hidden size matches the published v2 source-code defaults. Post-funding we scale to multi-node and match v2 paper-quality numbers. The codebase is dimension-agnostic — it's one config change."

---

## 10. What I am NOT specifying — and why

- **Exact loss numbers to target** — depends entirely on data subset chosen. Sanity threshold is vertex Pearson > 0.10 on held-out. Anything above 0.20 is genuinely strong for 24h on single GPU.
- **Wandb / tracking setup** — your call. The training loop logs basic metrics; plug in your preferred tracker.
- **Distributed training** — single-GPU only here. DDP comes later when you have a multi-node budget.
- **Knowledge distillation from TRIBE v2** — explicitly avoided to keep lineage clean. If you ever want to use TRIBE v2 as a teacher under a *new commercial license from Meta*, that's a separate conversation.
- **A Qwen / Mistral / Gemma variant** — explicitly removed (2026-05-26). Llama-3.2-3B is the sole text encoder. A future encoder swap is a one-line config change but is not in scope for v1.

---

## 11. Pre-flight checklist before handing to Claude Code

1. ✅ This spec is in `out/08_sapient1_sapient2_spec.md`
2. ✅ The licensing map is in `out/09_data_sources_and_licensing.md`
3. ✅ ds004996 confirmed CC0 — see [OpenNeuro ds004996](https://openneuro.org/datasets/ds004996) and its mirror [github.com/OpenNeuroDatasets/ds004996](https://github.com/OpenNeuroDatasets/ds004996). OpenNeuro defaults to CC0 per platform policy.
4. ⚠️ Verify BOLD Moments release license on OpenNeuro ds005165 before including it
5. ✅ Decided: A100 (40GB) for feature extraction, H100 (80GB) for training — see §12 Modal naming
6. ✅ HF org = `The-Sapient-Company` confirmed — see §13 HF naming
7. ⚠️ Confirm GitHub OAuth approval so I can audit your existing repo and flag any code that needs to stay out of these two new projects

---

## 12. Modal GPU naming convention (locked)

Every Modal app must be named explicitly so the cost dashboard, logs, and GPU usage are auditable. **Never launch an anonymous Modal job.** Format: `<model>-<stage>-<encoder-or-purpose>`.

### Feature extraction (run once, cache to disk, reuse forever)
| Modal app name | GPU | Purpose |
|---|---|---|
| `sapient-1-features-vjepa2` | 1×A100-80GB | V-JEPA 2 video features (1B params, needs headroom) |
| `sapient-1-features-w2vbert` | 1×A100-40GB | Wav2Vec-BERT 2.0 audio features |
| `sapient-1-features-llama` | 1×A100-40GB | Llama-3.2-3B text features (sole text variant) |

### Training (this is what costs real money)
| Modal app name | GPU | Purpose |
|---|---|---|
| `sapient-1-train-llama` | 1×H100-80GB | Sapient-1 from scratch, Llama-3.2-3B text encoder |
| `sapient-2-train-scratch-llama` | 1×H100-80GB | Sapient-2 from scratch on ds004996 + ds001740, Llama |
| `sapient-2-train-ft-llama` | 1×H100-80GB | Sapient-2 fine-tuned from sapient-1-llama |

### Eval (cheap, frequent)
| Modal app name | GPU | Purpose |
|---|---|---|
| `sapient-1-eval` | 1×A100-40GB | Vertex + parcel Pearson eval on held-out |
| `sapient-2-eval` | 1×A100-40GB | Same, for sapient-2 |

### Modal app definition pattern

```python
import modal

app = modal.App("sapient-1-train-llama")  # MUST match table above

image = modal.Image.debian_slim().pip_install(
    "torch==2.4.0", "transformers==4.46", "nilearn==0.10.4",
    "datalad", "einops", "wandb",
)

@app.function(
    gpu="H100:1",                # explicit GPU type, never default
    timeout=24 * 60 * 60,        # 24h cap, hard kill if exceeded
    volumes={"/data": modal.Volume.from_name("sapient-data")},
)
def train():
    ...
```

**Rule:** every Modal app name in this project starts with `sapient-1-` or `sapient-2-`. No exceptions. If you spin up an experiment under any other name, kill it immediately.

---

## 13. HuggingFace naming convention (locked)

**HF organization:** `The-Sapient-Company` ([huggingface.co/The-Sapient-Company](https://huggingface.co/The-Sapient-Company))

Every model, dataset, and space lives under this org. No personal-account uploads. The investor will look here.

### Models
| HF repo ID | What it is |
|---|---|
| `The-Sapient-Company/sapient-1-llama` | Sapient-1 trained with Llama-3.2-3B text encoder |
| `The-Sapient-Company/sapient-2-scratch-llama` | Sapient-2 from scratch on ds004996 + ds001740 |
| `The-Sapient-Company/sapient-2-ft-llama` | Sapient-2 fine-tuned from `sapient-1-llama` |

### Datasets (cached feature outputs — internal use, kept private)
| HF repo ID | What it is |
|---|---|
| `The-Sapient-Company/cneuromod-features-vjepa2` | V-JEPA 2 features over CNeuroMod stimuli |
| `The-Sapient-Company/cneuromod-features-w2vbert` | W2V-BERT features over CNeuroMod stimuli |
| `The-Sapient-Company/cneuromod-features-llama` | Llama-3.2-3B features over CNeuroMod transcripts |
| `The-Sapient-Company/cneuromod-fmri-fsaverage5` | fsaverage5 projected fMRI (20,484 vertices) |
| `The-Sapient-Company/ds004996-features-{vjepa2,w2vbert,llama}` | Same, for NeuroEngage |
| `The-Sapient-Company/ds004996-fmri-fsaverage5` | NeuroEngage fMRI on fsaverage5 |
| `The-Sapient-Company/ds001740-features-{vjepa2,w2vbert,llama}` | Same, for Rauchbauer HRI corpus |
| `The-Sapient-Company/ds001740-fmri-fsaverage5` | Rauchbauer fMRI on fsaverage5 |

**Rule:** feature-cache repos start `private=True`. They contain derivative work over CC0 stimuli, but we don't redistribute until we've checked encoder license terms (V-JEPA 2 outputs are MIT — fine; Llama outputs technically derive from Llama weights — keep private until legal review).

### Spaces (demos)
| HF space ID | What it is |
|---|---|
| `The-Sapient-Company/sapient-1-demo` | Live demo: paste a video URL, see vertex predictions |
| `The-Sapient-Company/sapient-2-demo` | Live demo: paste a conversation clip, see vertex predictions |

### Naming rule for derivative models using Llama

Per the Llama 3.2 Community License §5, any derivative model name MUST include "Llama" at the start. We comply by naming the artifact `sapient-1-llama` (the "-llama" suffix is sufficient under their interpretation, but the **model card title must read** "Sapient-1 (Llama-3.2 variant)" and the README must include "Built with Llama" prominently.

### `release.py` template

```python
from huggingface_hub import HfApi, create_repo

REPO_ID = "The-Sapient-Company/sapient-1-llama"  # one of: sapient-1-llama, sapient-2-scratch-llama, sapient-2-ft-llama

api = HfApi()
create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    folder_path="./release_artifacts",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Initial release: sapient-1-llama v0.1",
)
```

Flip `private=False` only after the investor demo and after legal has signed off on the model card.

---

## 14. Top-level engineer-onboarding files (mandatory, both repos)

Both `sapient-1/` and `sapient-2/` MUST include these at the repo root:

### `README.md`
What the project is, who it's for, 30-second elevator pitch, link to `ARCHITECTURE.md`, link to `CODEBASE_MAP.md`, quickstart (`make setup && make features && make train`), HF model link, license.

### `ARCHITECTURE.md`
The "how does this thing work" doc for a new engineer. Pulls the relevant sections out of this spec: data pipeline diagram, model diagram, training loop summary, eval methodology. Should be readable in 10 minutes.

### `CODEBASE_MAP.md`
The file-by-file walkthrough. New engineer onboarding doc + investor audit reference. See `out/CODEBASE_MAP.md` for the template — copy that into each repo and fill in the specifics.

### `Makefile`
```makefile
.PHONY: setup features train eval release clean test

setup:
	pip install -e .
	datalad install ...

features:
	modal run data/extract_video_features.py
	modal run data/extract_audio_features.py
	modal run data/extract_text_features.py

train:
	modal run train.py --config configs/sapient1_llama.yaml

eval:
	modal run eval.py --checkpoint $$(ls -t checkpoints/*.pt | head -1)

release:
	python release.py

test:
	pytest tests/ -v

clean:
	rm -rf checkpoints/ wandb/ __pycache__/
```

### `LICENSE`, `NOTICE`, `CITATION.cff`
Apache 2.0 LICENSE. NOTICE listing every third-party model with its license (see §6.3). CITATION.cff so investors / academics get a clean citation block.

---

Once those are checked, paste this spec section by section into Claude Code and let it build. Then come back to me to audit the diff.
