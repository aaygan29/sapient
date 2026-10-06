# Mary — Interface Contracts (WS0)

> Every workstream codes against THIS file. It is the stable boundary so agents can work in parallel.
> Mary is the faithful 6-stream **ORCLE** brain encoder, joining the family at `sapient-models/`.
> Build spec: `/Users/robertgutierrez/Desktop/sapient-research/Mary-Model-Insights/` (esp. `01-` and `04-`).

## 0. Data-availability decision (from the Modal check, H0)

Modal profile: `robert-16572`. Volume **`sapient-data`** already contains `raw/` with:
- `raw/cneuromod/fmriprep/{friends,movie10,...}` — CNeuroMod fMRIPrep derivatives **already materialized**,
  incl. `task-bourne05` (= the Bourne Ultimatum OOD stimulus for Paper B). Spaces present per run:
  `space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` (volumetric), `space-fsLR_den-91k_bold.dtseries.nii`
  (CIFTI grayordinate), `space-T1w`. **No fsaverage5 yet → must project.**
- `raw/ds002345` (Huth Narratives), `raw/ds005165` (BOLD Moments), `raw/ds001740`, `raw/ds004996`.

**VERIFIED (H0 stat check):** CNeuroMod BOLD is materialized real bytes (bourne03/04/05 preproc ≈ 1 GB MNI
each; fsLR CIFTI 148 MB) — NOT annex pointers. BUT **CNeuroMod movie stimuli are NOT on the volume and are
copyright-gated** (Bourne/Friends) → cannot drive feature extraction. The cneuromod superdataset only has
`fmriprep`+`smriprep` subdatasets (no raw BIDS, no stimuli).

**SPRINT DATASET DECISION — Huth Narratives (ds002345).** It is one of the papers' own training datasets
("Huth Narratives, audio+text, no video, CC0") and on the volume it has **29 open `.wav` stimuli
(`raw/ds002345/stimuli/*_audio.wav`) + multi-subject fMRI, materialized**. This is the legal, feasible,
paper-faithful workhorse for the 24h sprint:
- Active streams: **whisper, beats, qwen_ctx** (speech / audio events / narrative text). Video streams
  (slowfast, qwen_vl, got_ocr) have NO open stimulus media this sprint → recorded as missing per entry;
  Mary's missing-stream + modality-dropout logic handles this (exactly as the paper does for audio-only
  datasets). This is a logged deviation, not a silent gap.
- OOD eval (Papers A & B): hold out one narrative story (present across many subjects) as the OOD stimulus;
  compute the noise ceiling across subjects on it; run Mary (per-subject vs average) and TRIBE v2 (average)
  on it. Our numbers on Huth are the deliverable ("our own data, our own numbers").

**Routing consequences:**
- **WS-D**: skip CNeuroMod-stimulus work. For ds002345: fMRI is BIDS raw (`sub-*/func/*_bold.nii.gz`,
  volumetric) → project to **fsaverage5** (nilearn `vol_to_surf`, reuse `sapient1/data/prepare_fmri.py`),
  resample to 1 Hz, build manifest. Stimuli are the `.wav` files (one per story). Map story↔runs via BIDS
  `task-`/events. **Verify ds002345 `*_bold.nii.gz` are real bytes before processing.**
- **WS-F**: extract the 3 audio/text streams (whisper, beats, qwen_ctx) from the `.wav` stimuli (+ forced
  alignment for text/qwen_ctx). Author the 3 video extractors but mark them inactive (no stimuli) this sprint.
- **WS-B**: run TRIBE v2 (`facebook/tribev2`) zero-shot on a held-out ds002345 story (audio→trimodal with
  video zeroed) vs the multi-subject fMRI; same harness Mary uses.
- **If CNeuroMod stimulus access exists** (Robert/Yahvin have the movie files or a CNeuroMod DUA token):
  point WS-F at them to add video streams + the exact Bourne OOD — drop-in, no architecture change.

## 1. Output space (LOCKED, matches the whole family + papers)
- **20,484 fsaverage5 vertices** (10,242/hemisphere, LH then RH concatenated). Never change.

## 2. Tensor-shape contract (the spine)

Per-stream cached features → adapter → prediction:

| Symbol | Shape | Notes |
|--------|-------|-------|
| stream feature `m` | `(T_2Hz, D_m)` float16 | one `.npy` per (dataset,subject,run,stream) |
| batch input per stream | `(B, T_2Hz, D_m)` | B=batch, T_2Hz = stim_seconds×2 |
| after per-stream FFN | `(B, T_2Hz, 768)` | d_model = **768** |
| after fusion + temporal pool | `(B, T_TR, 768)` | T_TR = fMRI timepoints in the window |
| prediction | `(B, T_TR, 20484)` | the output |
| fMRI target | `(B, T_TR, 20484)` | z-scored per vertex per run |
| subject_idx | `(B,)` long | index into per-subject head/embedding |

**Stream dims `D_m` (LOCKED, from ORCLE paper):**
| stream key | backbone | D_m |
|------------|----------|-----|
| `slowfast` | SlowFast R101 | 2304 |
| `qwen_vl`  | Qwen3-VL-8B-Instruct | 3584 |
| `beats`    | BEATs | 768 |
| `whisper`  | Whisper-large-v3-turbo | 1280 |
| `qwen_ctx` | Qwen3-8B (128K) | 4096 |
| `got_ocr`  | GOT-OCR 2.0 | 768 |

> **Family alignment:** the family caches features at **2 Hz** and fMRI at **1 Hz**, windows of 100 TRs,
> HRF offset 5 (`sapient1/data/dataset.py`). Mary reuses this temporal convention: stream features at 2 Hz,
> fMRI resampled to 1 Hz, `T_2Hz = 200`, `T_TR = 100` per window. Mary's **attentive temporal pooling**
> (learned queries) replaces the family's mean-pool to map 2 Hz→1 Hz — this is the ORCLE-faithful part.

## 3. Feature cache path convention (Modal volume `sapient-data`)
```
/data/features/mary/{dataset}/{subject}/{run}/{stream}.npy     # (T_2Hz, D_m) float16
/data/fmri/mary/{dataset}/{subject}/{run}.npy                  # (T_1Hz, 20484) float32, z-scored
/data/manifest_mary.json                                       # see §4
```
`dataset ∈ {cneuromod, lebel2023, had, huth, wen2017}`; `stream ∈ {slowfast,qwen_vl,beats,whisper,qwen_ctx,got_ocr}`.

## 4. Manifest schema (`manifest_mary.json`) — consumed verbatim by `dataset.py`
```json
{
  "n_subjects": 9,
  "streams": ["slowfast","qwen_vl","beats","whisper","qwen_ctx","got_ocr"],
  "entries": [
    {
      "dataset": "cneuromod", "subject": "sub-01", "run": "movie10_bourne05",
      "subject_idx": 0, "n_trs": 100, "split": "train",
      "feature_paths": {"slowfast": "/data/.../slowfast.npy", "...": "..."},
      "fmri_path": "/data/fmri/mary/cneuromod/sub-01/movie10_bourne05.npy"
    }
  ]
}
```
Splits: `train` / `val` / `test`. Subjects with a missing stream (e.g. Wen2017 video-only) record only the
streams they have; **modality dropout (p=0.15) + missing-stream handling** zero the absent streams.

## 5. Config keys (`configs/mary_base.yaml`) — LOCKED hyperparameters (ORCLE paper)
```
model.d_model=768, n_fusion_layers=1, n_prediction_layers=2, n_vertices=20484,
modality_dropout=0.15, hrf_kernel_size=5, dropout=0.1, n_heads=8 (default), rope=true
loss: mse_weight=1.0, negcorr_weight=0.5, infonce_weight=0.1, infonce_temperature=0.07
train: optimizer=AdamW, lr=3e-4, weight_decay=0.01, scheduler=cosine_with_warmup,
       warmup_steps=1000, max_epochs=30, batch_size=16, sequence_length=200 (TR window),
       early_stopping_patience=5, early_stopping_metric=val/pearson_mean, grad_clip=1.0
seeds: [13]
```

## 6. Module API (so WS-A/eval can call Mary without reading internals)
```python
# mary/model.py
from mary.model import MaryModel, MaryConfig
cfg = MaryConfig.from_yaml(loaded_yaml)
model = MaryModel(cfg)
pred = model(features: dict[str, Tensor(B,T_2Hz,D_m)], subject_idx: Tensor(B,))  # -> (B, T_TR, 20484)

# mary/losses.py
from mary.losses import composite_loss   # composite_loss(pred, target, mask) -> (scalar, dict_of_terms)

# shared eval harness (Mary-Papers/shared/eval_harness.py)
pearson_r(pred, bold)                     # per-vertex r, returns mean/median/top10pct + per-vertex vector
noise_ceiling(bold_by_subject)            # mean pairwise inter-subject r  (sanity ≈ 0.1177 on bourne)
yeo7_decomposition(r_per_parcel)          # Schaefer-1000 -> Yeo-7 {network: (r, ceiling, %ceiling)}
bootstrap_ci(vec, n=10000, seed=42)       # (lo, hi)
prediction_structure_check(pred)          # asserts non-broadcast; returns lag-1 autocorr
```

## 7. Family conventions to MATCH (from sapient1)
- One-file importable `train.py` with `main_train(config_path, data_root, local)` + Modal `train_remote`.
- `eval.py` produces metrics JSON + brain-surface PNG. `release.py` pushes to `The-Sapient-Company/mary-*`.
- Modal: GPU H100-80GB train (24h timeout), A100 for extraction; secrets `hf-token`, `wandb`;
  W&B project `sapient`; volume `sapient-data` mounted at `/data`.
- Frozen backbones loaded ONLY in `data/extract_*` (lineage guarantee) — never in `mary/model.py`.
- Modal app naming: `mary-features-{slowfast,qwen_vl,beats,whisper,qwen_ctx,got_ocr}`,
  `mary-train`, `mary-eval`.

## 8. Deviations from the paper (logged honestly for the write-up)
- Temporal grid uses the family's 2 Hz-feature / 1 Hz-fMRI / 100-TR-window convention (vs ORCLE's native
  per-dataset TR). Faithful in spirit; attentive temporal pool still learned.
- POC scale: subset of subjects/hours per the 24h clock (ORCLE-Nano is itself a 15% POC — paper-consistent).
- Any backbone that fails to integrate in time is logged and its stream dropped (papers ablate stream count).
