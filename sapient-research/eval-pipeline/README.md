# sapienteval — Sapient Cognitive Eval (Track B: fine-tune)

A brain-grounded scoring system for AI agent `(prompt, response)` pairs. This folder holds the **Python + Modal** side: data download, fMRIPrep, parcellation, fine-tune, eval, and inference. The user-facing TypeScript scorer + UI (Track A) lives in [`src/sapienteval/`](../src/sapienteval/) and [`src/lib/sapienteval/`](../src/lib/sapienteval/).

## Scope

- **Track B (this folder)** — fine-tune a Sapient brain-encoding model on NeuroEngage (OpenNeuro ds004996) fMRI data to produce `sapient_v1.pt`. Wire it behind `SAPIENT_SCORER=v1`; only flip the flag when pre-registered acceptance criteria are met.
- **Track A (elsewhere)** — deterministic, literature-grounded TS scorer + GPT-4o-mini judge baseline + radar-chart UI. Ships first, gated to allowlisted emails.

## Isolation

This pipeline is **fully isolated** from the existing video-scan Modal pipeline (`Sapient/modal_pipeline/app.py` in the sibling repo). They share the same Modal workspace and Supabase project but:

- Separate Modal app names: `sapienteval-fmriprep`, `sapienteval-finetune`, `sapienteval-inference`.
- Separate Modal volumes: `sapienteval-data`, `sapienteval-checkpoints`. The base brain-encoding checkpoint is **re-downloaded** into the new volume — no shared state with the production `sapient-v2` inference.
- Separate Supabase tables (`sapienteval_*` prefix).
- Separate route + UI tab in the Sapient app at `/sapienteval`.
- The existing `/api/sapient-scan/*` flow is not touched.

## Folder map

```
sapienteval/
├── README.md                      # this file
├── SCIENCE.md                     # the "paper" — written incrementally during build
├── ACCEPTANCE_CRITERIA.md         # pre-registered v1 ship gates
├── splits.json                    # pre-registered train/val/test (committed before any training)
├── pyproject.toml                 # Python project metadata + deps
├── .gitignore                     # ignores data/, checkpoints, caches
├── 01_download_neuroengage.py     # datalad pull of ds004996 into data/
├── 02_run_fmriprep_modal.py       # (TBD) launch fMRIPrep containers on Modal
├── 03_parcellate.py               # (TBD) Schaefer-400 via nilearn
├── 04_align_stimuli.py            # (TBD) transcripts → TR-binned (text, BOLD) pairs
├── 05_finetune_sapient.py         # (TBD) local sanity loop
├── 06_train_mapping_head.py       # (TBD) 400 parcels → 8 cognitive dims
├── 07_eval_held_out.py            # (TBD) Pearson R on held-out subjects, touched ONCE
├── 08_make_model_card.py          # (TBD) emits src/data/sapient_v1_model_card.ts
├── data/                          # gitignored — raw BIDS + derivatives
└── modal/
    ├── README.md                  # Modal apps + volumes reference
    ├── __init__.py
    ├── fmriprep_runner.py         # (TBD) Modal app: sapienteval-fmriprep
    ├── finetune_sapient.py        # (TBD) Modal app: sapienteval-finetune
    └── inference_sapient.py       # (TBD) Modal app: sapienteval-inference
```

## Links

- [ACCEPTANCE_CRITERIA.md](./ACCEPTANCE_CRITERIA.md) — pre-registered v1 ship gates (read before training).
- `SCIENCE.md` — written during build as code lands; covers architecture, data pipeline, splits, validation methodology, and limitations.

## Naming

The fine-tuned model is called **Sapient brain-encoding architecture** descriptively: an 8-layer transformer that predicts BOLD signals across 400 cortical parcels. We do not cite external architectures in this folder, the model card, or `SCIENCE.md`.

## Track B preprocessing pipeline

Run these in order after standing up the `sapienteval` Python env (`pip install -e sapienteval`). Each step is idempotent and resumable; see each script's docstring for full CLI args.

1. **`01_download_neuroengage.py`** — DataLad pull of OpenNeuro `ds004996` into `data/ds004996/`. Pilot defaults to subjects from `splits.json` (~3 GB).
2. **`02_run_fmriprep_modal.py`** (a.k.a. `modal/fmriprep_runner.py`) — Run fMRIPrep on Modal with `--fs-no-reconall` and `--output-spaces MNI152NLin2009cAsym:res-2`. Writes preprocessed BOLD to `data/derivatives/fmriprep/sub-XX/func/`.
3. **`03_parcellate.py`** — Schaefer-400 7-Networks volumetric parcellation via `nilearn`. Reads fMRIPrep BOLD, emits `data/parcellated/sub-XX_task-{task}_run-{run}.npy` of shape `(T, 400)` float32. Standardized, detrended, bandpassed 0.01–0.1 Hz at TR=2.0s.
4. **`04_align_stimuli.py`** — TR-bin ds004996 `events.tsv` transcripts (text-only path). Emits `data/aligned/sub-XX_task-{task}_run-{run}.json` with per-TR text and event records, HRF-shifted by 4s.
5. **`05_finetune_sapient.py`** / `modal/finetune_sapient.py` — A100-40GB Modal fine-tune of the Sapient brain-encoding architecture on the `(parcellated, aligned)` pairs. Initializes from a base 8-layer transformer checkpoint (~700MB), freezes text/audio/video encoders, unfreezes the top 2 transformer layers + final LayerNorm + per-subject linear heads, then optimizes `MSE + 0.1 * temporal_smoothness` with AdamW (lr 1e-4 transformer / 1e-3 heads, cosine schedule, 500 warmup steps, batch 8 TRs, 5 epochs, early-stop patience 2 on within-subject val Pearson R). Output: `sapient_v0.5_pilot.pt` on the `sapienteval-checkpoints` volume. The thin Python wrapper defaults to **dry-run**; pass `--no-dry-run` to actually launch. Expected pilot cost: ~15 GPU-h ≈ $40. **TODOs before this can train:** (a) populate `BASE_CHECKPOINT_HF_REPO` in `modal/finetune_sapient.py` (or upload the base `.pt` to `/base/sapient_base.pt` on the checkpoints volume), and (b) wire the four stubs `build_brain_encoding_model`, `load_training_data`, `iter_batches`, `evaluate_pearson_r` once the base architecture spec and `04_align_stimuli.py` output format are finalized.
