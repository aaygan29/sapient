# Sapient Cognitive Eval — Handoff Doc

Written 2026-05-24 at the end of a long build session. Read this before touching anything in `sapienteval/` or `src/sapienteval/`.

---

## TL;DR (read first, 30 seconds)

**Track A (the demo) is shippable today.** Live at `/sapienteval` ("Eval" tab in main nav, between Library and API). Restyled to match `AnalyzeTab` visual language. v0 deterministic scorer passes 10/10 directional invariants on 8 showcase examples. Side-by-side A/B response comparison with brain vs LLM-judge scoring + 4 preset comparison pairs (Sycophantic vs Substantive, Refusal vs Helpful, Jargon vs Plain, ToM-blind vs ToM-aware).

**Track B (the real brain model) is built end-to-end but the trained checkpoint doesn't meet the pre-registered ship gate.** Pipeline runs cleanly: 19 NeuroEngage subjects preprocessed with fmriprep, parcellated to Schaefer-400, transcripts aligned per TR, model fine-tuned. But val_r = 0.020 vs gate of 0.20. Per the acceptance criteria you committed to, `SAPIENT_SCORER=v1` stays OFF in production. The trained checkpoint exists (`sapient_v1_scaleup.pt`) but is not in the inference path.

**~$10 of $300 budget spent.** Nothing is currently running. 0 active Modal containers, 0 local processes.

**One decision pending from you**: pick path forward (accept and ship v0 only / diagnose val_r plateau / deploy v1 endpoint as experimental). See §6.

---

## 1. What was built — full inventory

### Track A — the user-facing demo

| File | Purpose |
|---|---|
| `src/sapienteval/CognitiveEvalDemo.tsx` | Main /sapienteval view. Two tabs: Compare (A/B response comparison) + Validation (8 examples through PASS/FAIL invariants). |
| `src/sapienteval/ResponseCard.tsx` | Side-by-side response panel (Response A vs Response B with brain + judge score orbs). |
| `src/sapienteval/ScoreOrbs.tsx` | 8-orb circular score grid, 96px each, brain + judge ghost arc. |
| `src/sapienteval/CompareBar.tsx` | Per-dim A-vs-B bar chart with winner color. |
| `src/sapienteval/WinnerPanel.tsx` | Trophy + verdict + wedge story + divergence chips. |
| `src/sapienteval/ValidationGallery.tsx` | 8 examples through API in parallel → PASS/FAIL pills. |
| `src/sapienteval/CorrelationPanel.tsx` | Pearson r per dim between brain v0 + hand annotations (n=0 until annotations seeded). |
| `src/sapienteval/RadarChart.tsx` | Used only in Validation gallery for per-example mini-radars. |
| `src/sapienteval/FeatureTable.tsx` | Collapsible feature breakdown. |
| `src/sapienteval/comparisonPairs.ts` | 4 preset A/B pairs derived from the 8 single-response examples. |
| `src/sapienteval/allowlist.ts` | `SAPIENTEVAL_EMAILS` = your 2 emails. Imported by both client and server. |
| `src/lib/sapienteval/scorers.ts` | 8 deterministic cognitive-dim scorers, each with cited prior. |
| `src/lib/sapienteval/features.ts` | Hand-rolled text feature extractor, zero new deps. |
| `src/lib/sapienteval/examples.ts` | 8 showcase examples with expected directional invariants. |
| `src/lib/sapienteval/divergence.ts` | Cosine distance + deterministic feature-attribution explanation. |
| `src/lib/sapienteval/wordlist-top10k.ts` | Bundled top-10k English words for jargon ratio. |
| `src/lib/sapienteval/types.ts` | TypeScript types for cognitive scores, examples, results. |
| `api/sapienteval/score.ts` | POST endpoint. v0 deterministic by default; falls back from v1 to v0 if v1 endpoint unreachable. |
| `api/sapienteval/judge.ts` | GPT-4o-mini LLM judge baseline. |
| `api/sapienteval/annotate.ts` | Human annotation submit endpoint. |
| `api/sapienteval/correlations.ts` | Pearson r computation for CorrelationPanel. |
| `api/sapienteval/_lib.ts` | Shared helpers (Clerk auth, allowlist check). |
| `scripts/validate-sapienteval.ts` | CLI: brain-only deterministic scoring on 8 examples, asserts directional invariants, exits 0 on pass. |
| `scripts/seed-sapienteval-annotations.ts` | TSV bulk loader: insert `(cache_key, dim, human_score, annotator_email, notes)` rows into `sapienteval_annotations`. |
| `src/App.tsx` | Added: `import { SAPIENTEVAL_EMAILS } from './sapienteval/allowlist'` + lazy `CognitiveEvalDemo` + 'sapienteval' view in union + path check in `getInitialView()` + render block inside main shell + nav tab between Library and API (gated to allowlisted emails). |
| `server.ts` | Added 3 route blocks for `/api/sapienteval/{score,annotate,correlations}`. |
| `supabase/migrations/20260524000000_sapienteval.sql` | 3 tables: `sapienteval_cache`, `sapienteval_annotations`, `sapienteval_v1_inferences`. |

### Track B — Modal pipeline (in `sapienteval/`)

| File | Purpose |
|---|---|
| `sapienteval/README.md` | Folder scope + isolation note + folder map. |
| `sapienteval/SCIENCE.md` | "Paper" — descriptive architecture, data pipeline, v0/v1 scorer methodology, validation methodology, honest limitations. |
| `sapienteval/ACCEPTANCE_CRITERIA.md` | Pre-registered ship gates for v1. Including the §"Evaluation space" clarification that loss is computed in 400-parcel space (see §6 of this doc for context). |
| `sapienteval/splits.json` | Pre-registered train/val/test stratified by partner group (human/robot). Pilot = sub-01, 02, 03, 05 (sub-04 had no transcripts). |
| `sapienteval/pyproject.toml`, `.gitignore` | Python project metadata. |
| `sapienteval/01_download_neuroengage.py` | Local idempotent NeuroEngage downloader (also see `modal/data_download.py`). |
| `sapienteval/02_run_fmriprep_modal.py` | Thin client to invoke Modal fmriprep app. |
| `sapienteval/03_parcellate.py` | Local Schaefer-400 parcellation (Modal version in `modal/postprocess.py`). |
| `sapienteval/04_align_stimuli.py` | Local stimulus alignment (Modal version in `modal/postprocess.py`). |
| `sapienteval/05_finetune_sapient.py` | Local sanity loop (real training is `modal/finetune_sapient.py`). |
| `sapienteval/modal/data_download.py` | Modal CPU app `sapienteval-data-download`. Pulls ds004996 BIDS + OsLjung annotation repo via openneuro-py + git. Has `download_subjects` and `download_annotations` and `list_dataset` functions. |
| `sapienteval/modal/fmriprep_runner.py` | Modal CPU app `sapienteval-fmriprep`. Runs fmriprep with `--fs-no-reconall --output-spaces MNI152NLin2009cAsym:res-2`. Requires `freesurfer-license` Modal secret. Use `--wait` (blocking) — `.spawn()` is flaky on Modal for long runs. |
| `sapienteval/modal/repair_nifti.py` | Modal CPU app `sapienteval-repair-nifti`. Fixes ds004996 NIfTI headers that claim more TRs than the file actually contains. **MUST run before fmriprep** on each new subject batch. |
| `sapienteval/modal/postprocess.py` | Modal app `sapienteval-postprocess`. Two functions: `parcellate_subject` (Schaefer-400 7-Networks volumetric, no bandpass), `align_subject` (TR-bins events.tsv + merges OsLjung transcripts with HRF shift). |
| `sapienteval/modal/finetune_sapient.py` | Modal A100 app `sapienteval-finetune`. Loads base SapientModel + finetunes top 2 layers + readout. Critical refactor: bypasses Sapient's Fmri surface-projection extractor (requires FreeSurfer mesh artifacts we don't have); injects parcellated (B,400,T) targets per-segment via `_inject_parcel_targets_into_batch`. Loss computed in 400-parcel space. 4-check dry-run (model instantiates, state_dict zero diff, fwd+bwd finite, grad isolation). |
| `sapienteval/modal/inference_sapient.py` | Modal A100 app `sapienteval-inference` — **WRITTEN BUT NOT YET DEPLOYED**. Loads trained checkpoint + (literature-prior or learned) mapping head. Exposes `@modal.fastapi_endpoint POST /score`. Falls back from learned mapping to literature-prior if `mapping_head.npz` missing. |

### What's deployed on Modal right now

```
sapienteval-data-download   deployed
sapienteval-fmriprep        deployed
sapienteval-postprocess     deployed
sapienteval-finetune        deployed
sapienteval-repair-nifti    deployed
sapienteval-inference       NOT deployed (next-step decision)
```

### What's on the Modal volumes

```
sapient-base-encoder-weights  (RO) /best.ckpt (~700MB) + /config.yaml
                              Renamed from legacy simplr2-tribev2-weights.

sapienteval-data              /ds004996/sub-XX/anat,func,fmap/  for 20 subjects
                              /annotations/{human-human,human-robot}/  (OsLjung)
                              /derivatives/sub-XX/func/*_desc-preproc_bold.nii.gz
                              /parcellated/sub-XX_task-conversation_run-NN.npy
                              /aligned/sub-XX_task-conversation_run-NN.json

sapienteval-checkpoints       /sapient_v0.5_pilot.pt       (N=4 pilot run, val_r=0.015)
                              /sapient_v0.5_pilot_history.json
                              /sapient_v1_scaleup.pt        (N=19 scale-up, val_r=0.020)
                              /sapient_v1_scaleup_history.json
                              /base/                        (not yet populated; populated
                                                            by inference_sapient.py on first
                                                            warmup via symlink trick)

sapient-cache                 HF model caches (LLaMA-3.2-3B, etc.)
sapient-encoder-submodel-cache (was simplr2-tribev2-cache)
sapient-encoder-model-cache   (was tribe-model-cache)
sapient-archive-{inputs,outputs,cache}  (was simplr-*)
```

### Modal secrets

```
hf-token            HUGGINGFACE_TOKEN + HF_TOKEN
wandb               WANDB_API_KEY=disabled (placeholder so deploys don't fail)
freesurfer-license  FS_LICENSE_TEXT (your registered FreeSurfer license body)
```

### Supabase tables on remote

Applied migration `20260524000000_sapienteval`:

- `sapienteval_cache` — `(cache_key, prompt, response, brain_scores, judge_scores, features, divergence, divergence_explanation, judge_model, scorer_version, created_at)`. PK `cache_key`. No TTL.
- `sapienteval_annotations` — `(id, cache_key, dim, human_score, annotator_email, notes, created_at)`. Unique on `(cache_key, dim, annotator_email)`. CHECK constraint on dim ∈ 8 cognitive dims and human_score ∈ [0,1].
- `sapienteval_v1_inferences` — `(id, cache_key, checkpoint_sha, mapping_head_sha, raw_bold_400, modal_call_id, created_at)`. For provenance when v1 endpoint is live.

### Environment variables

```
HUGGINGFACE_TOKEN              already in file.env (user provided)
OPENAI_API_KEY                 already in file.env
BRAINTRUST_API_KEY             already in file.env
SUPABASE_URL                   already in file.env
SUPABASE_SERVICE_ROLE_KEY      already in file.env

SAPIENT_SCORER                 DEFAULT v0 (deterministic). Set to v1 to use trained model.
SAPIENT_V1_MODAL_ENDPOINT      empty by default. Set to Modal endpoint URL after
                               `modal deploy sapienteval/modal/inference_sapient.py` lands.
```

---

## 2. The val_r problem — what it is and what it means

### The numbers

| Run | n_train_pairs | best val_r | vs ship gate (0.20) |
|---|---|---|---|
| Pilot N=4  | 8  | 0.015 | 13× below |
| Scale-up N=19 | 38 | 0.020 | 10× below |

Loss is essentially flat (0.824 → 0.822 over 5 epochs). val_r barely moves. 5× more data + 5× more training pairs got us a 30% improvement in val_r — would need a **10× jump** to reach the gate.

### What this is

NOT a data-quantity issue alone. The base brain-encoding model was pretrained on thousands of hours of fMRI data. Our 19 subjects × ~11 min = ~190 min of brain data isn't enough to meaningfully fine-tune on top of that, even unfreezing 2 transformer layers + readout.

### What this is NOT

A bug in the training pipeline. The 4 sanity checks all pass:
1. Model instantiates with `n_outputs=20484, n_output_timesteps=100`
2. state_dict diff: 108 keys, **0 missing, 0 unexpected**
3. Forward + backward: loss is finite (~0.81)
4. Gradient isolation: 16 unfrozen params get finite grads, 0 frozen params leak grads

The model trains cleanly. It just doesn't learn the signal at this scale.

### Honest interpretation per the pre-registered acceptance criteria

`ACCEPTANCE_CRITERIA.md` §1 says "Ship `SAPIENT_SCORER=v1` to production only if Mean Pearson R across DMN+MFG parcels > 0.20". We are 10× below. **Do not flip the flag.** Document the result honestly.

### What I suspect is going on (untested)

The Sapient model expects per-second fsaverage5 vertex predictions during training, NOT TR-aligned parcels. By computing the loss in parcel space (§6 of SCIENCE.md explains why), we may be optimizing for a target that's a many-to-one projection of the model's natural output. The model's own gradient signal in this space is necessarily weaker.

A diagnostic worth ~30-60 min: log per-batch what `_inject_parcel_targets_into_batch` produces (segment-time alignment to TRs) and verify it matches the model's `n_output_timesteps=100` timeline. The interpolation in `_extract_parcel_targets` may be washing out signal.

---

## 3. What's left to do — the actual remaining tasks

### Pending decisions from you

1. **Pick a path on val_r** — accept/ship v0 only, OR diagnose, OR deploy v1 endpoint as experimental. See §6.
2. **Visually verify Track A in browser** at `localhost:3000/sapienteval`. After the typography restyle the page should match `AnalyzeTab` visual language. Confirm it doesn't still look broken to you.

### If you pick "Accept and ship v0 only"

- Nothing more to do on Modal side. `SAPIENT_SCORER` env var stays at default (v0).
- Update `SCIENCE.md` §6 to add: "v1 scale-up to N=19 failed to meet the 0.20 gate (best val_r = 0.020). v1 deferred until either (a) significantly more subjects added or (b) architectural changes to the fine-tuning approach."
- Mark Task #14 + #21 + #25 done with the negative result documented.
- Track A demo is the deliverable. Done.

### If you pick "Diagnose val_r plateau"

Steps with concrete commands:

1. Add per-batch logging to `_inject_parcel_targets_into_batch` in `sapienteval/modal/finetune_sapient.py`:
   ```python
   # After the chunks loop, before returning:
   if random.random() < 0.01:  # log 1% of batches
       print(f"[parcel-inject] segment={seg.timeline} start={seg.start:.2f}s "
             f"duration={seg.duration:.2f}s TR_range=[{start_tr}:{end_tr}] "
             f"chunk_shape={chunk.shape}")
   ```
2. Run a 1-epoch training to capture logs:
   ```bash
   modal run sapienteval/modal/finetune_sapient.py \
     --subjects sub-01,sub-02,sub-03,sub-05 \
     --epochs 1 --run-name diag_alignment
   ```
3. Inspect the logs. Confirm: segment durations make sense (should be many seconds of brain time, not fractional), TR ranges are valid (within [0, T_full)), chunk shapes are non-degenerate.
4. If misaligned: fix the seg.start interpretation (it may be measured in tokens, not seconds — check `neuralset.segments.Segment` source).
5. Re-run scale-up training:
   ```bash
   modal run sapienteval/modal/finetune_sapient.py \
     --subjects sub-01,sub-02,sub-03,sub-05,sub-06,sub-07,sub-08,sub-11,sub-12,sub-13,sub-14,sub-15,sub-19,sub-20,sub-21,sub-22,sub-23,sub-24,sub-25 \
     --run-name sapient_v1_scaleup_v2 --epochs 5
   ```
6. If val_r ≥ 0.20, proceed to "Deploy v1 endpoint" below.

Estimated cost: ~$2-5 of A100 time.

### If you pick "Deploy v1 endpoint as experimental"

```bash
modal deploy sapienteval/modal/inference_sapient.py
```

Note the URL printed at the end — should look like:
```
https://robert-16572--sapienteval-inference-sapientv1inference-score-endpoint.modal.run
```

Set in `file.env` (and Vercel env for prod):
```
SAPIENT_V1_MODAL_ENDPOINT=https://robert-16572--sapienteval-inference-sapientv1inference-score-endpoint.modal.run
SAPIENT_SCORER=v1
```

Restart the dev server. The /sapienteval Score button will now call the Modal endpoint, which loads `sapient_v1_scaleup.pt` + literature-prior mapping head, returns 8 cognitive scores. If the endpoint fails (timeout / error / malformed response), `api/sapienteval/score.ts` falls back to v0 silently — see lines 273-310.

**Honest framing**: the trained model is below the gate. Deploying the endpoint serves "real model outputs" but those outputs are weak. Useful for end-to-end pipeline testing, NOT for any external claim about model quality.

### Optional follow-ups (lower priority)

- **`06_train_mapping_head.py`** — learned 400→8 cognitive dim mapping head, fit on NeuroEngage engagement labels. Output: `mapping_head.npz` to `sapienteval-checkpoints`. The inference endpoint auto-detects and uses it if present, falls back to literature-prior weights if absent. Not built yet. Required only if you want learned mapping instead of hand-curated literature priors.
- **Hand-annotate 50 (cache_key, dim, human_score) rows** — populates the Correlation Panel on the Validation tab with real numbers. Use `scripts/seed-sapienteval-annotations.ts <tsv-path>` after scoring some examples (so the cache_keys exist). Until then the panel shows "n=0".
- **sub-18 fmriprep** — has been hung/preempted twice. Currently 19/20 subjects done is fine for training. If you want sub-18 specifically: kill any stale local process (`pkill -f "modal run.*sub-18"`), wipe partial state (`modal volume rm -r sapienteval-data /derivatives/sub-18`), relaunch (`modal run --detach sapienteval/modal/fmriprep_runner.py --subjects sub-18`).

---

## 4. Critical context to know before touching anything

### The Sapient package dependency (`/Users/robertgutierrez/Desktop/SIMPLR/sapient-model/`)

`sapienteval/modal/finetune_sapient.py` and `inference_sapient.py` both copy the local sapient repo into their Modal images via `add_local_dir(LOCAL_SAPIENT_REPO, ...)`. If you move or rename that directory, those Modal apps stop deploying. The constant is `LOCAL_SAPIENT_REPO = "/Users/robertgutierrez/Desktop/SIMPLR/sapient-model"` at the top of both files.

### The TRIBE → Sapient config patch

The base checkpoint's `config.yaml` references `TribeSurfaceProjector` (legacy class name from before the rebrand). Current `sapient` package only has `SapientSurfaceProjector`. `build_brain_encoding_model` in `finetune_sapient.py` patches the config in-flight (copies `/base/config.yaml` to `/tmp/sapient_base_patched/config.yaml` with the string replaced, then `SapientModel.from_pretrained` loads from the patched dir). Same patch is duplicated in `inference_sapient.py`. If the upstream sapient package ever registers `TribeSurfaceProjector` as an alias, both patches become unnecessary.

### Why the Fmri extractor is bypassed (the central architectural choice)

Sapient's Fmri data extractor calls `projection.apply()` which needs FreeSurfer's `surf_hybrid_mni_gii/` mesh files — custom MNI-warped meshes that aren't bundled with FreeSurfer and we don't have access to. Rather than fight that dependency, we:

1. Drop all `Fmri` events from the events DataFrame fed to `sapient_xp.data.get_loaders()` — only Word events.
2. Loader produces batches with text features only (no Fmri target).
3. `_inject_parcel_targets_into_batch` looks up the cached (T_full, 400) parcel array per segment timeline and slices to segment.start..segment.start+duration.
4. Model forward produces (B, 20484, T_seg) vertex predictions.
5. Project to (B, 400, T_seg) parcels via the fixed (20484, 400) matrix built in `_build_parcel_projection_matrix`.
6. MSE loss in 400-parcel space.

This is documented in `SCIENCE.md` §6 and `ACCEPTANCE_CRITERIA.md` §"Evaluation space".

### The dataset-quality landmine

ds004996 (NeuroEngage) has BOLD .nii files whose headers lie about TR count — header claims up to 573 volumes when only ~400 actually exist on disk. fmriprep crashes on this with "Expected N bytes, got M bytes" OSError. We fixed it with `sapienteval/modal/repair_nifti.py` which loads each .nii with nibabel, truncates dim[4] to match actual data, writes back in place.

**Run this BEFORE fmriprep on any new subjects**:
```bash
modal run sapienteval/modal/repair_nifti.py --subjects sub-XX,sub-YY,...
```

### Modal `.spawn()` vs `.map()`

For long-running fmriprep runs, `.spawn()` failed silently with empty `RemoteError` ~50% of the time. `.map()` via `--wait` works reliably. So `fmriprep_runner.py` has both: `--wait` for blocking (recommended for ≤10 subjects at a time) and the default `.spawn()` mode for fire-and-forget (use with caution).

Recommended pattern for ≥10 subjects: launch each one as its own `modal run --wait ... &` background process. Yes, ugly. Yes, works.

### Modal volume listing has caching/eventual-consistency

Sometimes `modal volume ls sapienteval-data /derivatives/sub-XX/func` returns stale data — files appear to "disappear" briefly and then reappear. Don't trust a single snapshot. Verify with multiple `ls` calls if something looks wrong. The trained checkpoint .pt file is stable; this affects only the BIDS derivatives.

### Auth model

- Frontend: `useAuth()` from `@clerk/react` → `getToken()` → pass as `Authorization: Bearer <token>` on every fetch
- Backend: `requireClerkUser(req, res)` from `api/_sapient_helpers.ts` verifies the JWT, returns the Clerk user `sub`
- Allowlist: `SAPIENTEVAL_EMAILS` constant in `src/sapienteval/allowlist.ts`. Re-checked server-side in `api/sapienteval/_lib.ts::requireAllowlistedUser`. Email lookup via Clerk users.getUser(sub) with Supabase user_profiles.email fallback.

---

## 5. Files map — where to find what

```
sapient/
├── sapienteval/                          # Track B (Python + Modal)
│   ├── HANDOFF.md                         # this file
│   ├── README.md                          # purpose, scope
│   ├── SCIENCE.md                         # the "paper" — sections 1-10
│   ├── ACCEPTANCE_CRITERIA.md             # pre-registered ship gates
│   ├── splits.json                        # pilot/train/val/test subject lists
│   ├── pyproject.toml, .gitignore
│   ├── 01_download_neuroengage.py         # local (Modal version in modal/)
│   ├── 02_run_fmriprep_modal.py
│   ├── 03_parcellate.py
│   ├── 04_align_stimuli.py
│   ├── 05_finetune_sapient.py
│   └── modal/
│       ├── data_download.py               # sapienteval-data-download
│       ├── fmriprep_runner.py             # sapienteval-fmriprep
│       ├── postprocess.py                 # sapienteval-postprocess
│       ├── finetune_sapient.py            # sapienteval-finetune
│       ├── inference_sapient.py           # sapienteval-inference (not deployed)
│       └── repair_nifti.py                # sapienteval-repair-nifti
│
├── src/sapienteval/                       # Track A (React UI)
│   ├── CognitiveEvalDemo.tsx              # main view at /sapienteval
│   ├── ResponseCard.tsx                   # Response A / Response B panels
│   ├── ScoreOrbs.tsx                      # 8 circular score orbs
│   ├── CompareBar.tsx                     # per-dim A vs B bars
│   ├── WinnerPanel.tsx                    # trophy + verdict
│   ├── ValidationGallery.tsx              # 8 examples + PASS/FAIL
│   ├── CorrelationPanel.tsx               # human-annotation Pearson r
│   ├── RadarChart.tsx                     # mini radar (Validation only)
│   ├── FeatureTable.tsx                   # collapsible feature breakdown
│   ├── comparisonPairs.ts                 # 4 preset A/B pairs
│   └── allowlist.ts                       # SAPIENTEVAL_EMAILS
│
├── src/lib/sapienteval/                   # Track A (shared TS scoring core)
│   ├── scorers.ts                         # 8 deterministic dim scorers
│   ├── features.ts                        # text feature extractor
│   ├── examples.ts                        # 8 showcase examples
│   ├── divergence.ts                      # cosine + feature attribution
│   ├── wordlist-top10k.ts                 # bundled English words
│   └── types.ts                           # CognitiveScores, FeatureBag, etc.
│
├── api/sapienteval/                       # Track A (Express/Vercel handlers)
│   ├── score.ts                           # POST /api/sapienteval/score
│   ├── judge.ts                           # GPT-4o-mini judge baseline
│   ├── annotate.ts                        # POST annotate
│   ├── correlations.ts                    # Pearson r aggregator
│   └── _lib.ts                            # shared auth helpers
│
├── scripts/
│   ├── validate-sapienteval.ts            # CLI 8-example validator
│   └── seed-sapienteval-annotations.ts    # TSV bulk loader
│
├── supabase/migrations/
│   └── 20260524000000_sapienteval.sql     # 3 tables
│
├── src/App.tsx                            # routing + Eval nav tab integration
└── server.ts                              # 3 lazy-import route blocks
```

---

## 6. The three options I asked you about (still open)

### Option 1 — Accept it. Ship Track A. Document Track B honestly.

- Track A demo is shippable as-is. v0 deterministic scorer in production.
- Update `SCIENCE.md` §6 with the val_r=0.020 result.
- Mark Track B v1 as "deferred until significantly more data or architectural changes".
- Most honest ending. ~0 cost to execute.

### Option 2 — Diagnose val_r plateau before giving up

- 30-60 min investigating segment.start → TR alignment in `_inject_parcel_targets_into_batch`.
- Re-run training if a fix is found. ~$2-5 cost.
- Could unblock v1 if alignment is the real issue. Risk: doesn't fix and we still ship v0.

### Option 3 — Deploy v1 endpoint as experimental

- `modal deploy sapienteval/modal/inference_sapient.py`
- Set `SAPIENT_V1_MODAL_ENDPOINT` + `SAPIENT_SCORER=v1` in env
- API now serves real-model predictions (weak, but real)
- Useful for end-to-end pipeline testing; NOT a quality claim
- ~$0.50 cost to deploy + tiny per-call cost

---

## 7. Budget tracking

As of 2026-05-24 ~20:30 PT:

```
sapienteval-fmriprep         $9.14    (20-subject pipeline, biggest item)
sapienteval-postprocess      $0.63
sapienteval-finetune         $0.37    (pilot + scale-up combined)
sapienteval-data-download    $0.07
sapienteval-repair-nifti     $0.007
                            ─────
sapienteval total my work    $10.23
```

Out of the $300 budget you authorized: **$289.77 still available**.

(Note: your wider Modal bill in the same window shows ~$161 total — but $151 of that is your existing `simplr-v2` + `sapient-v2` production pipelines, not my sapienteval work.)

To check current Modal spend yourself:
```bash
modal billing report --for today
modal billing report --start "7 days ago"
```

---

## 8. How to resume — fastest path back to productive work

1. `cd /Users/robertgutierrez/Desktop/sapient && git status` — see what's uncommitted from this session.
2. `npm run dev` if the dev server isn't running.
3. Open `localhost:3000/sapienteval` signed in with `robert@thesapientcompany.com` or `robert@realrobertgutierrez.com`.
4. Read this doc's §1-§4 in order — that's the full state in ~5 min.
5. Pick one of the three options in §6 and tell me (or future-me).

If you want to look at training history:
```bash
modal volume get sapienteval-checkpoints /sapient_v1_scaleup_history.json - 2>/dev/null
```

If you want to inspect the trained checkpoint:
```bash
modal volume ls sapienteval-checkpoints /
```

If something looks broken in the demo, the most likely suspects:
- Clerk JWT expired → re-login
- `SAPIENTEVAL_EMAILS` doesn't include your email → check `src/sapienteval/allowlist.ts`
- Dev server stale on a server.ts change → kill + restart `npm run dev`

---

## 9. Known issues / gotchas (sorted by risk)

| Issue | Risk | Workaround |
|---|---|---|
| Trained v1 doesn't meet 0.20 ship gate | HIGH | Don't flip `SAPIENT_SCORER=v1`. v0 deterministic handles production. |
| sub-18 fmriprep keeps dying | LOW | We have 19/20 subjects; training works. Skip sub-18 unless you specifically need it. |
| Modal volume `ls` has eventual-consistency caching | LOW | Don't trust a single ls. Use `modal volume ls --json` for stable listings. |
| `.spawn()` flaky for fmriprep | LOW | Use `--wait` (blocking) backgrounded with `&`. |
| TRIBE → Sapient config patch is in two files | LOW | If upstream sapient registers TribeSurfaceProjector as an alias, remove the patch from both `finetune_sapient.py` and `inference_sapient.py`. |
| `LOCAL_SAPIENT_REPO` is a hardcoded absolute path | LOW | Pin to `/Users/robertgutierrez/Desktop/SIMPLR/sapient-model`. Don't move that dir without updating both Modal scripts. |
| SCIENCE.md says ds004996 is Swedish (transcripts) but cognitive eval will be invoked on English text | MEDIUM | Documented in §9 of SCIENCE.md as a known limitation. Real ship of v1 should include English transcript validation. |
| Correlation Panel shows n=0 until annotations seeded | LOW | Use `scripts/seed-sapienteval-annotations.ts` with a TSV once you have annotated some cache_keys. |
| FeatureMeans in `divergence.ts` are placeholders | LOW | Tune from data after the validation gallery runs collect enough cache_keys. |

---

## 10. The git status as I close out

```
Branch: rename-simplr-to-sapient
Status: lots uncommitted (most of this session's work)
```

You should `git add -A && git commit -m "..."` when you're ready. I haven't committed anything during the session per the working assumption that you'd review first.

Things to NOT commit:
- `/tmp/*` scripts (I created some helper scripts there for monitors)
- Any local download caches (handled by `sapienteval/.gitignore`)
- Modal secrets (they're in Modal, not in the repo)

Things to definitely commit:
- All `src/sapienteval/*`, `src/lib/sapienteval/*`, `api/sapienteval/*`
- All `sapienteval/*` including this HANDOFF.md
- `scripts/validate-sapienteval.ts`, `scripts/seed-sapienteval-annotations.ts`
- `src/App.tsx` (nav integration), `server.ts` (route blocks)
- `supabase/migrations/20260524000000_sapienteval.sql`

---

End of handoff. Good luck.
