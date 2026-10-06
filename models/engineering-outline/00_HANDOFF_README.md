# Sapient — Claude Code Handoff Package

**Last updated:** 2026-05-26
**Status:** Llama-only, audited and aligned across all docs.

---

## What to give Claude Code (drop these into the project)

Paste this prompt into Claude Code first, then attach the files below:

> Reload specs — sapient is now **Llama-only** per 2026-05-26 decision. The full architecture, training plan, and data licensing live in `out/`. Read `00_HANDOFF_README.md` first, then `08_sapient1_sapient2_spec.md` (the build doc), then `CODEBASE_MAP.md` (file-by-file repo layout). Use `09_data_sources_and_licensing.md` for any licensing questions. Do not introduce Qwen, Mistral, Gemma, or any text encoder other than Llama-3.2-3B. The HF org is `The-Sapient-Company` — never `simplrads`.

---

## File index — what each doc is and when to use it

### Tier 1 — Claude Code reads these every session

| # | File | What it is | When Claude needs it |
|---|---|---|---|
| **00** | `00_HANDOFF_README.md` | This file. Orientation + read order. | First thing every session. |
| **08** | `08_sapient1_sapient2_spec.md` | **The build doc.** Full architecture, code skeletons, training config, repo layout, HF/Modal naming, week-by-week plan. 1,008 lines. | Whenever writing model, dataset, training, or eval code. |
| **CODEBASE_MAP** | `CODEBASE_MAP.md` | File-by-file map of the `sapient1/` and `sapient2/` repos. What lives where, what each file does, demo-day checklist. | Whenever creating or editing any file in the repo. |
| **09** | `09_data_sources_and_licensing.md` | Per-dataset license, source URLs, download instructions, encoder licenses, compliance table. | Before downloading data, releasing weights, or answering license questions. |

### Tier 2 — reference docs (read on demand)

| # | File | What it is | When Claude needs it |
|---|---|---|---|
| 01 | `01_tribev2_architecture_spec.md` | TRIBE v2 architecture reverse-engineered from Meta's code + paper. Source-of-truth for what we are diverging from. | When comparing our code to TRIBE, debugging shape mismatches, or defending architectural choices. |
| 02 | `02_sapient_codebase_audit.md` | Side-by-side TRIBE vs Sapient design choices with investor-facing talking points. | When fielding "how is this different from Meta?" |
| CODEBASE_MAP §11 | (inside CODEBASE_MAP.md) | Demo-day script. | Day-of investor demo. |

### Tier 3 — investor / narrative materials (you, not Claude Code)

| # | File | What it is | When you need it |
|---|---|---|---|
| 03 | `03_sapient_vs_tribe_onepager.md` | One-page Sapient vs TRIBE comparison. | Send to investors before meetings. |
| 04 | `04_top3_defenses.md` | Three hardest investor objections + crisp answers. | 5 min before any pitch. |
| 05 | `05_neuroengage_finetune_walkthrough.md` | Step-by-step ds004996 + ds001740 fine-tune runbook for sapient-2. | When kicking off the sapient-2 training week. |
| 06 | `06_headline_cheatsheet.md` | One-page numbers cheatsheet (177M params, 26.5h public data, etc.). | Quick reference during calls. |
| 07 | `07_investor_mock_drill_30q.md` | 30 hardest questions an investor will ask + your scripted answers. | Day before pitch — rehearse out loud. |

### Diagrams (visual reference — share these with investors AND Claude Code)

| File | What it shows |
|---|---|
| `sapient1_architecture.png` / `.svg` | Sapient-1: V-JEPA 2 + W2V-BERT + Llama-3.2-3B → 8-layer transformer → 20,484 fsaverage5 vertices. Trained on CNeuroMod (4 CC0 subjects) + BOLD Moments + Narratives. |
| `sapient2_architecture.png` / `.svg` | Sapient-2: same architecture, two training paths (scratch + fine-tune from sapient-1), trained on ds004996 NeuroEngage (~50 subj) + ds001740 Rauchbauer HRI (25 subj). Includes §9 lineage panel. |
| `diagrams/sapient_diagrams.py` | The renderer. Run `python diagrams/sapient_diagrams.py` to regenerate PNG + SVG after any architecture edit. |

---

## What "Llama-only" means in practice

**Decision date:** 2026-05-26
**Decision:** Llama-3.2-3B is the **sole** text encoder. Qwen2.5-7B and all other variants are removed — no Modal app, no HF release, no cached features. The `proj_text` Linear is dimension-agnostic in code, so a future encoder swap is one config change away, but it is explicitly out of scope for v1.

**Concrete consequences:**

- HF model releases: **3 only** — `The-Sapient-Company/sapient-1-llama`, `The-Sapient-Company/sapient-2-scratch-llama`, `The-Sapient-Company/sapient-2-ft-llama`
- Feature caches per dataset: **3 only** — `{vjepa2, w2vbert, llama}` (no qwen variant)
- Training Modal apps: **3 only** — `sapient-1-train-llama`, `sapient-2-train-scratch-llama`, `sapient-2-train-ft-llama`
- Feature-extraction Modal apps: 3 per model — `sapient-{1,2}-features-{vjepa2,w2vbert,llama}`
- Compute + storage savings vs the old dual-encoder plan: **~40-50%**

---

## Architecture lock — do not deviate without explicit instruction

These 15 details are the spec §3 architecture corrections. Verified in `out/08_sapient1_sapient2_spec.md` §3.

1. **Modality dropout** `p=0.3` applied **after projections, before sum**.
2. **Subject dropout** `p=0.1` at training time only.
3. **Pos embed:** `nn.Parameter(1, 100, 1152) × 0.02`, **learnable**, `max_len=100`. Not sinusoidal, not 200.
4. **Pos embed + subject embed** added **after** the 2 Hz → 1 Hz mean-pool.
5. **Order:** project → modality_dropout → SUM → pool → +pos → +subj → transformer.
6. **Subject embed** broadcast: `subject_embed[idx].unsqueeze(1)` over 100 timesteps.
7. **Two-stage head:** `low_rank: Linear(1152 → 2048, bias=False)` then `brain_head: Linear(2048 → 20484)`.
8. **HRF offset:** fMRI window = `[start_tr+5, start_tr+105)`, stim window = `[start_tr*2, (start_tr+100)*2)`.
9. **Whisper-large-v3** is an ablation only (dashed border in diagrams). W2V-BERT 2.0 is the primary audio encoder.
10. **Schaefer-1000 atlas** is built once offline, used only in §7 post-processing — not at training time.
11. **sapient-2-ft** uses `lr=5e-5`, **not** `1e-5`.
12. **sapient-2-ft** drops `subject_embed` when loading sapient-1 weights: `state = {k: v for k, v in sapient1_state.items() if 'subject_embed' not in k}`.
13. **n_subjects:** 4 for sapient-1 (CC0 CNeuroMod sub-01, 02, 03, 05); ~75 for sapient-2 (50 ds004996 + 25 ds001740).
14. **BOLD Moments + Narratives = augmentation only** (week 2). **CNeuroMod = primary** (week 1).
15. **Loss:** per-vertex squared error → mean over 20,484 vertices → mask-weighted mean over time.

---

## Data lock

**sapient-1 (week 1 primary, week 2 augmentation):**

- CNeuroMod (primary): 4 CC0 subjects (sub-01, 02, 03, 05) × ~80 h ≈ **320 h**. Friends S1-6 + Movie10. TR = 1.49 s. Released CC0 via CONP.
- BOLD Moments (week 2 aug): 10 subjects, 1,102 short videos. OpenNeuro `ds005165`. TR = 1.75 s.
- Narratives (week 2 aug, features-only): 300+ subjects listening to spoken stories. TR = 1.5 s.

**sapient-2:**

- ds004996 NeuroEngage (primary): ~50 subjects × 11 min × 3 runs ≈ **16.5 h**. CC0 on OpenNeuro.
- ds001740 Rauchbauer (HRI augmentation): 25 French speakers × ~24 min ≈ **10 h**. Furhat robot. Pin v2.1.0. CC0 on OpenNeuro.
- **Pooled sapient-2 total: ~26.5 h, ~75 subjects.**

---

## HF + Modal naming lock

**HF org:** `The-Sapient-Company` ([huggingface.co/The-Sapient-Company](https://huggingface.co/The-Sapient-Company)). **Never** `simplrads`.

**Model releases (3):**

- `The-Sapient-Company/sapient-1-llama`
- `The-Sapient-Company/sapient-2-scratch-llama`
- `The-Sapient-Company/sapient-2-ft-llama`

**Feature cache repos (per dataset):**

- `The-Sapient-Company/{cneuromod,ds004996,ds001740}-features-{vjepa2,w2vbert,llama}` (9 total)
- `The-Sapient-Company/{cneuromod,ds004996,ds001740}-fmri-fsaverage5` (3 total)

**Modal apps:**

- Feature extraction: `sapient-{1,2}-features-{vjepa2,w2vbert,llama}` — A100-80GB for vjepa2, A100-40GB for w2vbert and llama
- Training: `sapient-1-train-llama`, `sapient-2-train-scratch-llama`, `sapient-2-train-ft-llama` — H100-80GB
- Eval: `sapient-{1,2}-eval` — A100-40GB

---

## What you keep as reference (do NOT paste into Claude Code)

- `04_top3_defenses.md`, `06_headline_cheatsheet.md`, `07_investor_mock_drill_30q.md` — your own pitch prep. Not engineering inputs.
- `03_sapient_vs_tribe_onepager.md` — send to investors, not to Claude.
- `sapient_cuban_prep_handoff.md` (workspace root) — Mark Cuban call prep, superseded by tier-1/2 docs.
- `diagrams/sapient_architecture.png/.svg/.py` — the **old** combined diagram from before sapient-1/sapient-2 split. Keep as historical reference, do not use.

---

## Suggested Claude Code session opener (copy-paste ready)

```
You are working on Sapient — pre-seed AI startup building brain encoder
models (sapient-1, sapient-2). Read these in order before any code:

  1. out/00_HANDOFF_README.md
  2. out/08_sapient1_sapient2_spec.md
  3. out/CODEBASE_MAP.md
  4. out/09_data_sources_and_licensing.md

Hard rules:
  - Llama-3.2-3B is the only text encoder. No Qwen, Mistral, Gemma.
  - HF org is `The-Sapient-Company`. Never `simplrads`.
  - Every release name ends in `-llama`.
  - The 15 architecture decisions in 00_HANDOFF_README §"Architecture lock"
    are non-negotiable. Cite the number if you ever need to deviate.

Start by confirming the four files above are loaded. Then ask me what
you should build first.
```
