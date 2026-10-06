# sapient-1 Codebase Map

Repo-local copy of the workspace codebase map. The canonical version lives in
[`../engineering-outline/CODEBASE_MAP.md`](../engineering-outline/CODEBASE_MAP.md).
When the layout below diverges between sapient-1 and sapient-2, this copy is
authoritative for sapient-1; the sibling copy in `sapient2/CODEBASE_MAP.md` is
authoritative for sapient-2.

---


**Purpose:** the file-by-file walkthrough of the `sapient-1/` and `sapient-2/` repos. Two audiences:

1. **New engineers** — read this in 15 minutes and know where everything lives before opening a single file.
2. **Investor technical due-diligence** — read this and verify the codebase is structured, auditable, and license-clean.

**Version:** v1, dated 2026-05-26. Pair with [`08_sapient1_sapient2_spec.md`](./08_sapient1_sapient2_spec.md) (the build spec) and [`09_data_sources_and_licensing.md`](./09_data_sources_and_licensing.md) (the licensing map).

---

## 0. The 30-second version

Two PyTorch projects. Same architecture, different training data. Each:

- Downloads CC0 fMRI from OpenNeuro / CNeuroMod
- Extracts frozen features through V-JEPA 2 (video), W2V-BERT (audio), Llama-3.2-3B (text)
- Trains an 8-layer transformer (hidden=1152, ~180M params trainable) that predicts 20,484 fsaverage5 cortical-surface vertices per second
- Evaluates with vertex-wise + Schaefer-1000-parcel Pearson correlation
- Ships to HuggingFace under Apache 2.0

Everything runs on Modal. Every Modal app, HF repo, and feature cache is named under the convention in `08_sapient1_sapient2_spec.md` §12–13. No anonymous resources.

---

## 1. Repo-root files (both `sapient-1/` and `sapient-2/`)

| File | What it does | Read it when |
|---|---|---|
| `README.md` | Public face. 30-sec pitch, quickstart, HF link, license. | First touch with the repo. |
| `ARCHITECTURE.md` | The "how this thing works" doc. Data flow + model diagram + training loop summary. | Before reading any code. |
| `CODEBASE_MAP.md` | This file (copied into each repo). | While navigating the repo. |
| `LICENSE` | Apache 2.0, full text. | Legal review. |
| `NOTICE` | Required by Apache + Llama license. Lists every third-party model with its license and copyright holder. See §6.3 of `08_sapient1_sapient2_spec.md`. | Legal review. |
| `CITATION.cff` | Citation File Format. Auto-generates BibTeX for citing the model. | Academic / investor follow-ups. |
| `pyproject.toml` | Python package config. Declares deps with pinned versions. | Setup, version bumps. |
| `Makefile` | One-line commands: `make setup`, `make features`, `make train`, `make eval`, `make release`, `make test`. | Every day. |
| `.gitignore` | Excludes `checkpoints/`, `wandb/`, `data/raw/`, etc. | When something keeps getting committed by accident. |

---

## 2. `configs/` — all hyperparameters in one place

**Design rule:** no hyperparameter is allowed to be hardcoded in `*.py`. Every knob lives in YAML.

| File | What it configures |
|---|---|
| `sapient1_base.yaml` (or `sapient2_base.yaml`) | Architecture defaults: hidden=1152, heads=8, layers=8, ff=4608, low_rank_head=2048, dropout=0.1. Shared across variants. |
| `sapient1_llama.yaml` | Inherits from base. Sets `text_encoder: meta-llama/Llama-3.2-3B`, `text_dim: 3072`. **Sole text variant.** |
| `sapient2_scratch_llama.yaml` | Inherits from base. `train_data: ds004996+ds001740`. Random init. |
| `sapient2_ft_llama.yaml` | Inherits from base. `train_data: ds004996+ds001740`. `init_checkpoint: The-Sapient-Company/sapient-1-llama`, drops `subject_embed` on load, `lr: 5e-5`. |

**Note (2026-05-26):** Qwen2.5-7B was removed from the project. `proj_text` is dimension-agnostic in code (one config line to swap to Qwen/Mistral/Gemma later), but no Qwen YAML, Modal app, or HF release exists in v1.

Why this matters for audit: a reviewer can diff two YAMLs and see exactly what changed between two trained models. No hidden state.

---

## 3. `data/` — data pipeline

Runs once per dataset, caches everything to disk + HF dataset repos. **Read order = execution order.**

| File | What it does | Modal app name | Output |
|---|---|---|---|
| `download_cneuromod.sh` | `datalad install` + `datalad get` for CC0 subjects only (sub-01, 02, 03, 05). | n/a (CPU) | `data/raw/cneuromod/` |
| `download_bold_moments.sh` | OpenNeuro ds005165 download via `aws s3 cp` or datalad. | n/a | `data/raw/bold_moments/` |
| `download_narratives.sh` | OpenNeuro ds002345 download. | n/a | `data/raw/narratives/` |
| `download_ds004996.sh` (sapient-2 only) | NeuroEngage. CC0 check before proceeding. | n/a | `data/raw/ds004996/` |
| `download_ds001740.sh` (sapient-2 only) | Rauchbauer 2019/2020 HRI corpus (Furhat robot, French). Pinned to v2.1.0. CC0 check. | n/a | `data/raw/ds001740/` |
| `prepare_fmri.py` | Vol → fsaverage5 surface projection (20,484 vertices). Builds the Schaefer-1000 vertex→parcel projection matrix. Resamples to 1 Hz. | n/a (CPU OK, but heavy) | `data/fmri/{dataset}/sub-XX_run-YY.npy` shape `(T, 20484)` + `data/parcellate.npz` |
| `extract_video_features.py` | V-JEPA 2 Gigantic frozen forward pass over all video stimuli, 2 Hz. | `sapient-1-features-vjepa2` (1×A100-80GB) | `data/features/vjepa2/...` shape `(T*2, 1280)` |
| `extract_audio_features.py` | W2V-BERT 2.0 frozen forward pass, 2 Hz. | `sapient-1-features-w2vbert` (1×A100-40GB) | `data/features/w2vbert/...` shape `(T*2, 1024)` |
| `extract_text_features.py` | Llama-3.2-3B frozen forward pass over transcripts. Pooled to per-word, then aligned to 2 Hz. | `sapient-1-features-llama` (1×A100-40GB) | `data/features/llama/...` shape `(T*2, 3072)` |
| `dataset.py` | `SapientDataset` class. Samples 100-second windows, applies 5-TR HRF offset. Returns dict of video/audio/text/fmri/subject/mask tensors. | n/a | importable Python class |
| `manifest.py` | Builds the JSON manifest mapping (subject, run, clip) → file paths + length metadata. | n/a | `data/manifest.json` |

**License audit point:** `download_*.sh` scripts MUST refuse to proceed if the dataset's `dataset_description.json` doesn't list a CC0 license. Hardcode the check.

---

## 4. `sapient1/` or `sapient2/` — the model

The model package. Same structure in both repos; renamed at the package level.

| File | What it does |
|---|---|
| `__init__.py` | Exports `SapientModel`, `SapientConfig`, version. |
| `model.py` | The top-level `SapientModel(nn.Module)`. Builds: 3 input projections (video/audio/text → 1152d), subject embedding (added post-pool with `subject_dropout=0.1`), **learnable** positional embedding (`nn.Parameter(1, 100, 1152) × 0.02`, max_len=100, added post-pool), 8-layer transformer encoder, low-rank head (1152 → 2048 → 20484, bias=False on the low-rank projection). |
| `modules.py` | `InputProjection`, `TransformerBlock`, `LowRankHead`. Each is a single small `nn.Module`. |
| `losses.py` | `masked_mse_loss(pred, target, mask)`. Per-vertex MSE, then mask-aware mean. |
| `metrics.py` | `vertex_pearson(pred, target)`, `parcel_pearson(pred, target, parcel_matrix)`. Both vectorized. |
| `inference.py` | `vertices_to_parcels(vertex_pred, parcel_matrix)` — the adapter that bridges to `sapienteval/`. |
| `utils.py` | Config loading, seed setting, device helpers. |

**Audit point:** `model.py` is the **only** file that imports from `transformers` (for type hints — encoders are loaded in `data/`, not here). The training code never touches third-party model weights at runtime. That's the lineage guarantee.

---

## 5. `train.py` — single-file training entry point

One file, ~300 lines. Read top to bottom.

Flow:
1. Parse `--config <path>` arg.
2. Load YAML → `SapientConfig`.
3. Set seed.
4. Build `SapientModel(config)`.
5. Build `SapientDataset(manifest_path, ...)` × {train, val}.
6. Build `AdamW(lr=1e-4)` + `OneCycleLR(pct_start=0.1)`.
7. For each epoch:
    - Train loop (bf16, gradient accumulation, grad clip 1.0)
    - Validation Pearson
    - Save checkpoint if best
    - W&B log
8. Final eval on held-out test set.

Designed so a new engineer can read this file and know everything the training does. **No clever indirection.**

---

## 6. `eval.py` — held-out evaluation

| Function | What it does |
|---|---|
| `evaluate_vertex_pearson(model, loader)` | Run model on held-out data, compute per-vertex Pearson r averaged over time. |
| `evaluate_parcel_pearson(model, loader, parcel_matrix)` | Project predictions to Schaefer-1000, compute per-parcel Pearson. |
| `plot_brain_surface(pearson_map, output_path)` | nilearn plot of Pearson on fsaverage5. Investor demo asset. |
| `dump_metrics_json(...)` | Writes `eval/results.json` for reproducibility. |

Modal app: `sapient-1-eval` (1×A100-40GB).

---

## 7. `release.py` — HuggingFace publishing

Single script. Creates the HF repo (private), writes model card from template, uploads weights + config + tokenizer references.

Hardcoded constants at top:
```python
HF_ORG = "The-Sapient-Company"
REPO_ID = f"{HF_ORG}/sapient-1-llama"  # sole text variant in v1
```

Outputs to `release_artifacts/` first for review. **Never auto-uploads.** Engineer runs `python release.py --confirm` to actually push.

---

## 8. `tests/` — minimum viable tests

| File | What it tests |
|---|---|
| `test_shapes.py` | Forward pass produces `(B, 100, 20484)` for `(B, 200, *)` inputs. |
| `test_overfit.py` | Train on a single batch for 50 steps. Loss must drop below 0.01. |
| `test_parcellation.py` | `vertices_to_parcels` produces `(T, 1000)` and is deterministic. |
| `test_license_guard.py` | `download_*.sh` scripts refuse non-CC0 datasets. |

CI runs all four on every push.

---

## 9. The lineage guarantee (read this twice)

This is what an investor's technical advisor will ask about. Memorize it.

**Claim:** Sapient-1 and Sapient-2 contain zero TRIBE v2 weight lineage and zero TRIBE v2 source code.

**Evidence in the codebase:**

1. `model.py` is < 400 lines, written from the TRIBE v1 paper architecture description. No file copied from `facebookresearch/algonauts-2025`.
2. `git log` shows commits from the Sapient team only. No squash of an upstream repo.
3. `NOTICE` lists every third-party weight: V-JEPA 2 (MIT), W2V-BERT (MIT), Llama-3.2-3B (Llama 3.2 Community License). No TRIBE v2.
4. Training data is CC0 from McGill (CNeuroMod) and OpenNeuro. No data inherited from a non-CC0 source.
5. The published `The-Sapient-Company/sapient-*` checkpoints are signed with a deterministic hash of the training config + dataset manifest. Reproducible end-to-end.

**Verbal version to the investor:** "Our architecture is published-paper science from the TRIBE v1 paper. Our weights are trained from scratch by us, on CC0 data, on our compute. Our license is Apache 2.0. We have written documentation of every line of provenance."

---

## 10. Onboarding checklist for a new engineer

Day 1 (4 hours):
- [ ] Clone repo, run `make setup`
- [ ] Read `README.md`, `ARCHITECTURE.md`, this file
- [ ] Read `08_sapient1_sapient2_spec.md` §1–5
- [ ] Run `make test` — all green

Day 2 (full day):
- [ ] Read `09_data_sources_and_licensing.md` end-to-end
- [ ] Run `bash data/download_cneuromod.sh` and watch it work
- [ ] Run `data/prepare_fmri.py` on one subject
- [ ] Open one fMRI numpy array and verify shape is `(T, 20484)`

Day 3 (full day):
- [ ] Spin up `sapient-1-features-vjepa2` on Modal, extract V-JEPA features for one CNeuroMod clip
- [ ] Open `sapient1/model.py` and walk through one forward pass with `pdb`
- [ ] Read `train.py` end-to-end

Day 4–5:
- [ ] Launch a 1-epoch training run on `sapient-1-train-llama`
- [ ] Run `eval.py` on the resulting checkpoint
- [ ] Render brain surface plot

By end of week 1 the engineer should be able to make a one-line config change and ship a new training run.

---

## 11. Audit checklist for technical due diligence

Investor's CTO-friend asks for a 30-min walkthrough. In this order:

1. **Open `NOTICE`** — verify it lists every third-party model + license. (90 seconds)
2. **Open `LICENSE`** — verify Apache 2.0. (10 seconds)
3. **Open `data/download_*.sh`** — verify every dataset is CC0. (2 minutes)
4. **Open `sapient1/model.py`** — verify it's clean PyTorch with no copied code. Walk through forward pass. (10 minutes)
5. **Open `configs/sapient1_llama.yaml`** — verify hyperparameters match the spec doc. (2 minutes)
6. **Open `train.py`** — verify the training loop is auditable. (10 minutes)
7. **Open `tests/test_overfit.py`** + run it — verify model actually learns. (5 minutes)
8. **Show HF repo** [`The-Sapient-Company/sapient-1-llama`](https://huggingface.co/The-Sapient-Company/sapient-1-llama) — verify model card, weights, license tag. (1 minute)
9. **Show eval results** — held-out vertex Pearson + brain surface plot. (5 minutes)

If anyone asks "isn't this just TRIBE v2?", answer with §9 above.

---

## 12. What this file does not cover

- **Hyperparameter rationale** — that's in `08_sapient1_sapient2_spec.md` §3–4.
- **Why each dataset was chosen** — that's in `09_data_sources_and_licensing.md`.
- **The downstream `sapienteval/` stack (Buy Moment + 8 scorers)** — that's a separate repo owned by Sapient, deliberately not in this codebase. See `02_sapient_codebase_audit.md`.
- **Production inference** — out of scope for this build. The trained checkpoints are the deliverable; productionization comes post-funding.

---

*Last updated: 2026-05-26. If this file is older than 30 days, ask whoever last touched the repo to refresh it.*
