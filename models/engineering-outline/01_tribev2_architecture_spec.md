# TRIBE v2 — Architecture Spec (Layer-by-Layer)

Reverse-engineered from [facebookresearch/tribev2](https://github.com/facebookresearch/tribev2) (commit `main`, May 2026), the [HF model card](https://huggingface.co/facebook/tribev2), the [Meta blog post](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/), and the v1 paper [arXiv 2507.22229](https://arxiv.org/html/2507.22229v1).

Numbers in **bold** are pulled from `tribev2/grids/defaults.py` and `tribev2/model.py` — they are the actual values you'd reproduce by running `python -m tribev2.grids.test_run`.

---

## 1. Big picture

TRIBE v2 is a **multimodal-to-fMRI encoder**. Given time-aligned (video, audio, text) of a naturalistic stimulus, it predicts the fMRI BOLD response on the **fsaverage5 cortical surface (~20,484 vertices per hemisphere → ~20k cortical vertices total when using both hemispheres at fsaverage5 resolution)** at 1 Hz (1 sample / TR).

```
  video ─► V-JEPA2-ViT-G  ─┐
  audio ─► Wav2VecBERT     ├─► per-modality MLP projector ─► concat ─► 8-layer Transformer ─► low-rank head ─► SubjectLayers ─► AdaptiveAvgPool1d ─► fMRI[B, V, T']
  text  ─► Llama-3.2-3B    ─┘                                                                       │
                                                                                                    └── +time_pos_embed (+ optional subject_embed)
```

The **trainable core** is everything from the projector onward. The three foundation encoders are **frozen feature extractors** — TRIBE v2 caches their layer activations to disk and only ever trains the fusion-transformer + heads.

---

## 2. Inputs and temporal alignment

| Stream | Encoder | Checkpoint | Sampling rate | Layers used | Native dim |
|---|---|---|---|---|---|
| Video | V-JEPA 2 ViT-Gigantic | `facebook/vjepa2-vitg-fpc64-256` | **2 Hz**, 4-sec clips | `[0.75, 1.0]` (2 layers, top two quarters) | ~1408 |
| Image (still) | DINOv2-Large | `facebook/dinov2-large` | 2 Hz | `2/3` of depth | 1024 |
| Audio | Wav2Vec-BERT (Seamless) | `Wav2VecBert` (registered in `neuralset`) | 2 Hz | `[0.75, 1.0]` | 1024 |
| Text | Llama-3.2-3B (gated) | `meta-llama/Llama-3.2-3B` | 2 Hz (events at word onset) | `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` (6 layer groups) | 3072 |
| fMRI target | — | — | **1 Hz** (TR ≈ 1.0–1.49 s depending on study) | — | ~20k vertices |

All modality features are resampled to **2 Hz**. fMRI is at **1 Hz** with a **5-TR hemodynamic offset** (`offset: 5` in the `neuro_extractor` config) — the model predicts the BOLD signal 5 TRs after the stimulus, matching the well-known HRF peak around 5 s.

**Layer aggregation** (`Data.layer_aggregation = "group_mean"`): The chosen pretrained layers are partitioned into groups, mean-pooled within each group, then concatenated. With `layers_to_use=[0.5, 0.75, 1.0]` (the default in `defaults.py`), each modality contributes 3 layer-group features per timestep, concatenated.

**Why this matters for the pitch:** Layer-group averaging is the v1 trick that survived into v2. Early/middle/late transformer layers carry different cortical correlates; concatenating gives the brain encoder access to all of them at once, instead of you guessing which layer matches which cortical area.

---

## 3. Per-modality encoders — frozen, not fine-tuned

`tribev2` does not unfreeze the foundation encoders. They run **once** in a feature-extraction pass (see `main.py:Data.get_loaders` → `extractor.prepare(events)` → `_free_extractor_model` which deletes the model from GPU after caching), then the cached features are loaded from disk during training.

Practical consequences:
- **Compute decoupling.** v1 paper: feature extraction was 128×V100 × 24h; model training was 1×V100 × 24h. v2 carries the same split.
- **Gradients only flow into the projector + combiner + transformer + low-rank head + SubjectLayers**, not into V-JEPA, Wav2Vec, or Llama.
- **License risk surfaces here.** TRIBE v2 weights are **CC-BY-NC-4.0** (non-commercial), but the cached Llama-3.2-3B activations also inherit Llama's community license terms. Anything you ship commercially has to re-extract features under an unencumbered text encoder, or use Llama under its own commercial terms — not under TRIBE's CC-BY-NC umbrella.

---

## 4. The trainable core — `FmriEncoder` (in `tribev2/model.py`)

This is the only thing TRIBE v2 actually trains. Five sequential blocks:

### 4.1 Per-modality projector (`self.projectors`, an `nn.ModuleDict`)
- For each modality: an **MLP** (`norm_layer="layer"`, `activation_layer="gelu"`), input dim = `feature_dim × num_layers` (because `layer_aggregation="cat"`), output dim = `hidden // n_modalities`.
- With `hidden=1152` and 3 modalities (text, audio, video), each projector outputs **384-dim**. With image added as a 4th, it's 288-dim.
- Modality dropout: during training, with probability **0.3** an entire modality vector is zeroed for a given batch sample. This is what gives v2 robustness to missing modalities at inference time.

### 4.2 Aggregation (`extractor_aggregation = "cat"`)
- Concatenate the per-modality projector outputs along the channel dim → **hidden=1152** per timestep.
- Alternative configs (`"stack"`, `"sum"`) exist but the released default is concat.

### 4.3 Combiner (`config.combiner = None` in defaults)
- The default config sets `combiner: None`, which makes the combiner an `nn.Identity()` — the concatenated 1152-dim vector goes straight into the transformer. (A 2-layer Mlp combiner is an alternative.)

### 4.4 Time positional embedding (`self.time_pos_embed`)
- Learned, **shape (1, max_seq_len=1024, 1152)**.
- Added (not concatenated) to the input. No RoPE, no sinusoidal — just a learned table.

### 4.5 Subject embedding (`self.subject_embed`) — **OFF by default**
- `subject_embedding: False` in defaults. The subject conditioning happens in `SubjectLayers` at the output, not as an additive token at the input.
- If turned on: an `nn.Embedding(n_subjects, hidden)` whose lookup is added to every timestep.

### 4.6 Transformer encoder (`self.encoder = config.encoder.build(dim=hidden)`)
- **8 layers** (`encoder.depth: 8` in defaults).
- Heads, MLP-ratio, attn-dropout, ff-dropout, layer-dropout: not set explicitly in `defaults.py` for v2 — they take the `neuraltrain.models.transformer.TransformerEncoder` defaults. The handoff document mentions v1 used **8 heads, hidden 3072**; v2 uses **hidden 1152** and likely the same head count.
- **Bidirectional** (encoder-only, no causal mask). Attention is over the 100-TR × 2 Hz = **200-step** time window.
- `attn_dropout = ff_dropout = layer_dropout = config.dropout` (set to 0.0 in defaults, but the plumbing exists).

### 4.7 Low-rank head (`self.low_rank_head`)
- `low_rank_head: 2048` → an `nn.Linear(1152, 2048, bias=False)` between transformer output and the subject-conditional predictor.
- **Why this exists:** the predictor maps to ~20k vertices × n_subjects. Without the bottleneck, the parameter count of the output head would dominate the model. The 2048-dim factorization is what makes per-subject heads tractable.

### 4.8 SubjectLayers predictor (`self.predictor`)
- `SubjectLayers` from `neuraltrain.models.common`. Effectively a **bank of per-subject linear projections** from the 2048-dim shared representation to the per-subject output (cortical vertices in the cortical grid, or subcortical voxels in the subcortical grid).
- `subject_dropout: 0.1` → 10% of the time, the subject-specific weights are replaced with the "average subject" — this is the mechanism for **zero-shot generalization to unseen subjects** at inference (you just route through the average-subject path).
- This is the mechanism for the moat claim: more subjects → wider tensor → richer shared bottleneck. Sapient's edge is putting *thousands* of subjects through here, not 720.

### 4.9 Temporal smoothing (optional, not on by default)
- `TemporalSmoothing` is a **depth-wise Gaussian Conv1d** (kernel 9, optionally fixed sigma) applied **before** the transformer. Off in the released default config.

### 4.10 Output pooling (`self.pooler`)
- `nn.AdaptiveAvgPool1d(n_output_timesteps)`.
- The transformer runs at the input rate (2 Hz). The pooler downsamples to the fMRI rate (1 Hz). Output shape: `(B, n_vertices, T')` where `T' = duration_trs = 100`.

---

## 5. Output space

| Setting | Value | Source |
|---|---|---|
| Surface | **fsaverage5** | `neuro_extractor.projection.mesh` |
| Projection kind | **ball** (radius 3 mm) | `neuro_extractor.projection` |
| Cortical vertices | **20,484** (fsaverage5 standard, both hemispheres) | HF card says "~20k" |
| Subcortical variant | Mask-based volumetric, `fwhm: 6.0` Gaussian smoothing | `grids/run_subcortical.py` |
| fMRI hemodynamic offset | 5 TRs | `neuro_extractor.offset: 5` |
| Allow missing | True (subjects/runs without fMRI just contribute features) | `neuro_extractor.allow_missing` |

The model outputs one value per cortical vertex per TR — it predicts the **mean BOLD signal** at the vertex, not voxel-level, not high-resolution.

**Important nuance for the pitch:** "20k vertices" is a marketing-friendly description. The real number is 20,484 (fsaverage5 = 10,242 per hemisphere × 2). If anyone presses, *that's the number*.

---

## 6. Loss and optimization (from `grids/defaults.py`)

| Knob | Value |
|---|---|
| Loss | **`MSELoss(reduction="none")`**, then `.mean()` in `pl_module.py` |
| Optimizer | **Adam**, lr **1e-4**, weight_decay **0.0** |
| Scheduler | **OneCycleLR**, max_lr 1e-4, pct_start 0.1 |
| Epochs | **15** |
| Batch size | **8** (per GPU) |
| Duration per sample | **100 TRs** ≈ 100 s |
| Train/val split | val_ratio **0.1** by time within timelines (held-out tail of each story/episode) |
| Modality dropout | **0.3** |
| Subject dropout | **0.1** |
| Seed | 33 |

**Critical detail about the loss:** It's a **plain MSE across all cortical vertices simultaneously**, no per-vertex weighting, no noise-ceiling normalization in the loss itself. Noise ceiling is applied at *evaluation* time as a normalization on Pearson r. The v1 paper also reports `Pearson`, `SmoothL1`, `Huber` as ensemble variants, but the released v2 default is straight MSE.

**Bad-sample handling (`pl_module._run_step`):** When `stride_drop_incomplete=False`, rows where all vertex targets are zero are masked out before loss computation. This is how missing/padded frames are excluded without changing the dataloader.

---

## 7. Metrics

Three online metrics, computed every step:

| Metric (`log_name`) | Class | What it measures |
|---|---|---|
| `pearson` | `OnlinePearsonCorr` (dim=0) | Vertex-wise Pearson correlation across all subjects pooled |
| `subj_pearson` | `GroupedMetric(OnlinePearsonCorr, kwargs={dim:0})` grouped by subject | Per-subject Pearson — for inter-subject generalization analysis |
| `retrieval_top1` | `TopkAcc(topk=1)` | Stimulus retrieval: given a predicted brain pattern, can you pick the matching stimulus segment vs. distractors |

**Monitor:** `val/pearson`. Best checkpoint = highest val Pearson.

**For the pitch:** Pearson r is the de-facto standard for brain-encoding evaluation since at least Naselaris/Gallant 2011. Reporting MSE alone would be a tell that you don't read the literature; reporting Pearson + noise-ceiling-normalized Pearson is the standard adults use.

---

## 8. Training dataset composition

`tribev2/grids/defaults.py` lists 4 studies as the default training corpus:

| Study | Stimulus type | Subjects | Hrs/subject (from `utils.RECORDING_DURATIONS`) |
|---|---|---|---|
| **Algonauts2025Bold** | Movies + Friends sitcom | 4 used (sub-01, 02, 03, 05) | 66.4 hrs each |
| **Lahner2024Bold** | Short video clips | 10 | 6.2 hrs each |
| **Lebel2023Bold** | Spoken stories (text/audio) | 8 | 6.2–18.1 hrs each |
| **Wen2017** | Naturalistic video | 3 | 11.7 hrs each |

Recording-duration totals from `RECORDING_DURATIONS` (constants in `utils.py`):
- Algonauts2025Bold: 4 subj × 66.4 hrs = **265.6 hrs**
- Lahner2024Bold: 10 subj × 6.2 hrs = **62 hrs**
- Lebel2023Bold: ≈ **87 hrs** total (mixed)
- Wen2017: 3 subj × 11.7 hrs = **35.1 hrs**
- **Total ≈ 450 hours of subject-time** documented in the repo constants.

The Meta blog and HF card say **"1,000+ hours of fMRI across >700 subjects"** — that is the **full training corpus they used for the publication**, which extends well beyond the 4 studies enumerated in the public release config. The repo ships the *recipe*, not the full dataset. This is normal for Meta releases (compare BrainMagick, Llama-3 pretrain data).

**`MultiStudyLoader` (in `utils.py`) handles multi-study composition:** Each study is loaded as a `Chain`, the same set of transforms (audio extraction, word transcription, sentence/context attachment, chunking, query, split) is applied uniformly, and they're concatenated into one DataFrame.

**Subject weighting (`get_subject_weights` in `utils.py`):**  
`weigh_by ∈ {"n_subjects", "speech", "video", "recording_time"}` — controls whether the loss weighs subjects equally, weighs by recording time, or restricts to speech-only or video-only studies. The released default is `n_subjects` (equal weight per study, normalized so each study contributes equally).

---

## 9. Training compute

The repo is built for SLURM. From `grids/defaults.py`:

- `gpus_per_node: 1`, `mem_gb: 128`, `timeout_min: 60*24*3` (3-day soft cap).
- Feature extraction is the expensive part — `max_jobs: 1024` for fMRI extraction, `max_jobs: 1024` for video (V-JEPA inference on lots of clips).
- v1 paper baseline: **128 × V100 × 24 h for feature extraction**, **1 × V100 × 24 h for model training**.
- For v2, expect ~3–10× the feature-extraction compute (more studies, more videos) but model training itself remains a **single-GPU job** because everything past the projector fits on one GPU.

**This is the single most important architectural choice you should internalize:** **TRIBE is a single-GPU model on top of a 100+-GPU feature-extraction farm.** The expensive thing isn't the brain encoder — it's the labeled brain data and the foundation-model feature cache. The model is one tab in the spreadsheet; the dataset is the rest of the spreadsheet.

---

## 10. Evaluation protocol

From `main.py` and `pl_module.py`:

- **Splits**: by-time within each timeline. The `SplitEvents` transform (val_ratio=0.1) holds out the **last 10% of each story/episode**. Optional `split_segments_by_time` mode also exists for held-out timelines.
- **Held-out subjects** are handled via `average_subjects: True` and `resize_subject_layer: True`. At test time the model can either (a) use the trained subject-specific weights or (b) route through the average-subject path enabled by `subject_dropout=0.1`.
- **Held-out tasks/languages** (e.g. the zero-shot claims): tested by training on Algonauts+Lahner+Wen and evaluating on Lebel2023 (speech), or by training on English studies and evaluating on non-English audio.
- **Metric of record**: vertex-wise Pearson, optionally noise-ceiling-normalized (paper-style: `r / r_ceiling`).

---

## 11. Inference path (from `demo_utils.py:TribeModel.predict`)

```python
from tribev2 import TribeModel
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
df    = model.get_events_dataframe(video_path="clip.mp4")   # builds Audio+Video+Word events
preds, segments = model.predict(events=df)                  # (n_timesteps, n_vertices)
```

What `get_events_dataframe` does:
1. Extracts audio from video → `ExtractAudioFromVideo`.
2. Chunks audio and video to 30–60-second windows.
3. Transcribes audio → word-onset events via `ExtractWordsFromAudio`.
4. Attaches sentence and context to each word event (`AddText`, `AddSentenceToWords`, `AddContextToWords` with `max_context_len=1024`).
5. Returns the standardized event DataFrame.

What `predict` does:
1. Builds a one-shot DataLoader with the same extractors used in training.
2. Runs the model in eval mode over **100-TR strided segments**.
3. Returns `(preds, segments)` — predictions in shape `(n_timesteps, n_vertices)` and the segment list for indexing back to stimulus time.

**The TTS detour for text-only input (`TextToEvents`):** If you only have text, `TribeModel` synthesizes it to audio with gTTS, then transcribes it back to get word-level timing. This is how the "text" modality is given temporal alignment even when no audio is provided.

---

## 12. Architectural decisions that look like minor configs but aren't

These are the easy-to-miss choices that make v2 work:

1. **Modality dropout 0.3, not 0.0.** Trains the model to handle missing modalities. Lets it ingest text-only stories (Lebel) and silent-movie clips alike, and makes inference robust to one-modality use.
2. **Subject dropout 0.1.** Reserves an "average subject" slot — this is the zero-shot-to-new-subjects mechanism.
3. **Low-rank head with bottleneck 2048.** Without this, the per-subject output layer would balloon to ~20k × n_subjects × hidden parameters. The 2048-dim factorization is what makes scaling to hundreds of subjects feasible on one GPU.
4. **Layer aggregation `[0.5, 0.75, 1.0]` with `group_mean`.** Picks middle, late, and top layers of each frozen encoder — well-established finding in encoding-model literature that middle layers correlate with mid-level cortex, late layers with high-level cortex.
5. **Adaptive pooling, not transformer downsampling.** The transformer runs at 2 Hz; the pooler does the rate conversion to 1 Hz fMRI at the very end. This decouples the temporal model from the fMRI TR.
6. **Studies cite a 5-TR HRF offset, not a learned HRF.** TRIBE v2 does *not* try to learn the hemodynamic response function. It assumes the canonical 5-TR delay and trains the model to predict the BOLD at t+5 directly. A learned HRF would be a defensible Sapient differentiator if you've done it.

---

## 13. What's NOT in TRIBE v2 (the negative spaces — these are your wedge)

- **No real-time inference.** TRIBE is batch-only. There is no streaming inference loop, no <100ms latency story.
- **No closed-loop / decoding direction.** TRIBE is strictly stimulus→brain (encoding). It is *not* a brain→stimulus decoder. Sapient can be both.
- **No conversational / interactive stimuli.** Every training study is a passive-viewing or passive-listening paradigm (movies, podcasts, stories). NeuroEngage is literally the only public dataset with active 10-minute back-and-forth conversation in the scanner. This is your moat.
- **No physiological signals (HR, GSR, eye, pupil) fused in.** Pure stimulus → fMRI. The agent/robotics use case needs multi-signal.
- **No fine-tuning recipe for production users.** The repo is research-grade — there's `resize_subject_layer` to extend to new subjects, but no "fine-tune on your 5 hours and ship" path.
- **License: CC-BY-NC-4.0.** Cannot ship commercially as-is. You either re-train weights or use it for benchmarking only.
- **No cognitive-state labels.** TRIBE predicts BOLD signal, not engagement / attention / confusion / valence. The whole jump from "predicting fMRI" to "predicting cognitive state" is downstream of TRIBE and is where Sapient's product lives.

---

*End of architecture spec.*
