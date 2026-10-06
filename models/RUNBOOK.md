# Sapient Build Runbook

Sequential commands to take `sapient-1` and `sapient-2` from "code lives in the repo" to "weights live on HuggingFace". Copy-paste each block in order. Estimated costs and times are listed.

**Last validated:** Stages 1 (CPU smoke) and 2 (Modal probe) are green. Stages 3+ have not yet been run.

**Architectural rule (2026-05-26):** every dataset download writes directly to the `sapient-data` Modal volume. Nothing lands on your laptop. All `download_*.py` scripts are Modal CPU apps that read from the source (OpenNeuro / CONP) and write into `/data/raw/<dataset>/`. Downstream feature-extraction / training / eval Modal apps read from the same volume. **Your local disk usage stays under 15 MB for the entire build.**

---

## Pre-flight (one-time, ~30 s each)

```bash
# Confirm auth is live (returns user + The-Sapient-Company)
hf auth whoami

# Confirm Modal is live (lists existing apps; no spend)
modal app list | head -5

# Confirm the smoke pipeline still works (~3 min, $0)
cd ~/Desktop/sapient-models/sapient1 && make smoke-test
```

---

## Stage 3a — Download CNeuroMod (~$0.50–$1, ~2–6 h)

```bash
cd ~/Desktop/sapient-models/sapient1
modal run data/download_cneuromod.py
```

What it does: Modal CPU job (4 CPUs, 8 GB RAM, ~$0.04/CPU/hr) that runs `datalad install` + `datalad get` for the 4 CC0 subjects (sub-01, sub-02, sub-03, sub-05), both Friends and Movie10 tasks. **Writes directly to the `sapient-data` Modal volume at `/data/raw/cneuromod/`** — nothing touches your laptop.

When it's done, the Modal volume contains:

```
/data/raw/cneuromod/fmriprep/{friends,movie10}/sub-{01,02,03,05}/...   # BOLD .nii.gz
/data/raw/cneuromod/fmriprep/{friends,movie10}/sourcedata/{friends,movie10}/stimuli/   # videos
```

Total size: ~50–200 GB on the volume (not on your laptop). Idempotent — re-running picks up where it stopped.

**Optional Week-2 augmentation** (run only when needed):
```bash
modal run data/download_bold_moments.py     # OpenNeuro ds005165, ~$0.20
modal run data/download_narratives.py        # OpenNeuro ds002345, ~$0.30 (features-only)
```

---

## Stage 3b — Forced word-level alignment (~$5–15, ~30 min – 2 h)

Required for `extract_text_features.py` because CNeuroMod doesn't ship word-level timing. Uses whisper-timestamped on Modal A100-40GB.

```bash
cd ~/Desktop/sapient-models/sapient1
modal run data/forced_align.py \
    --stim-root /data/raw/cneuromod/fmriprep/friends/sourcedata/friends/stimuli
modal run data/forced_align.py \
    --stim-root /data/raw/cneuromod/fmriprep/movie10/sourcedata/movie10/stimuli
```

Output: writes `<stim>.words.json` next to each video/audio file in the shared `sapient-data` Modal volume. Idempotent — already-aligned files are skipped.

**Cost note:** Friends S1–S6 is ~55 h of audio. At ~3 min of GPU time per hour of audio, that's ~3 h of A100-40GB ≈ $5. Movie10 adds another ~$2.

---

## Stage 3c — fMRI surface projection (~$0.05–$0.20, ~30–60 min)

Runs on Modal CPU (or local CPU if you ever want; the script is environment-agnostic). Projects volumetric BOLD onto fsaverage5 (20,484 vertices) and builds the Schaefer-1000 vertex→parcel matrix once.

Inside a Modal function (reads + writes the `sapient-data` volume):

```python
# Wrap the existing data/prepare_fmri.py main() in a Modal app, or run via
# modal volume put/get if you prefer. The script's CLI args are unchanged:
modal run data/prepare_fmri.py \
    --in /data/raw/cneuromod \
    --out /data/fmri/cneuromod \
    --tr 1.49 \
    --build-parcellation
```

(`prepare_fmri.py` currently has a `__main__` CLI but no Modal wrapper — that's a 20-line addition if you want to run it remotely. Locally on your laptop it would download nilearn's fsaverage5 mesh (~50 MB cache) which is small enough to be fine, but you'd still need the BOLD files mounted somehow — Modal mounting is simpler.)

---

## Stage 3d — Frozen feature extraction (~$3–8, ~2–4 h)

Three Modal apps; each can run in parallel.

```bash
cd ~/Desktop/sapient-models/sapient1
modal run data/extract_video_features.py    # V-JEPA 2 Gigantic, A100-80GB
modal run data/extract_audio_features.py    # W2V-BERT 2.0,     A100-40GB
modal run data/extract_text_features.py     # Llama-3.2-3B,     A100-40GB
```

Output: `/data/features/{vjepa2,w2vbert,llama}/cneuromod/<stim>.npy` per stimulus.

To launch all three in parallel from separate terminals: just open three terminals.

---

## Stage 3e — Build the manifest (~$0.02, ~1 min on Modal)

```bash
cd ~/Desktop/sapient-models/sapient1
modal run data/manifest.py
```

Reads `/data/fmri/cneuromod/*` and `/data/features/*` on the `sapient-data` volume; writes `/data/manifest.json` back to the same volume. This is what `train.py` and `eval.py` consume. Splits per spec §5.1: train = Friends S1–6 (all but last 3 episodes per season) + 3 of 4 Movie10 movies; val = last 3 Friends episodes per season; test = held-out Movie10 movie ("Life" by default).

---

## Stage 4 — Train (~$100–150, 24 h)

**This is the big spend. Make sure you're around to watch it.**

```bash
cd ~/Desktop/sapient-models/sapient1
modal run train.py --config configs/sapient1_llama.yaml
```

What this does: H100-80GB for up to 24 hours, AdamW + OneCycleLR + bf16 mixed, 15 epochs (or fewer if the timeout hits), val Pearson every epoch, best checkpoint saved to `/data/checkpoints/sapient1_llama/`.

W&B logs go to project `sapient` (the Modal secret `wandb` injects your API key).

**Monitor:** the Modal app dashboard for `sapient-1-train-llama`, or the W&B run page. Either kill on the dashboard if metrics go sideways.

---

## Stage 5 — Eval (~$0.50, ~5–15 min)

```bash
cd ~/Desktop/sapient-models/sapient1
modal run eval.py \
    --checkpoint $(modal volume ls sapient-data /checkpoints/sapient1_llama/ | grep best | sort | tail -1)
```

Writes `eval/results.json` + `eval/brain_surface_pearson.png` (a 4-panel fsaverage5 inflated-mesh heatmap; visual cortex should be hottest).

---

## Stage 6 — Release (~$0, ~5 min)

Release runs from your laptop because it needs the HF token from your local config + composes the model card from in-repo `LICENSE` / `NOTICE` files. But the trained checkpoint lives on the Modal volume — fetch it first with `modal volume get`:

```bash
cd ~/Desktop/sapient-models/sapient1

# 1. Pull the best checkpoint + parcellation matrix off the Modal volume.
#    ~150 MB on disk briefly during the upload; you can rm it after step 3.
mkdir -p _release_in
modal volume get sapient-data /checkpoints/sapient1_llama/best.pt _release_in/best.pt
modal volume get sapient-data /parcellate.npz _release_in/parcellate.npz

# 2. Dry-run — assembles release_artifacts/sapient-1-llama/ locally.
python release.py --checkpoint _release_in/best.pt \
                  --parcellation _release_in/parcellate.npz

# 3. Push to HF (private repo).
python release.py --checkpoint _release_in/best.pt \
                  --parcellation _release_in/parcellate.npz \
                  --confirm

# 4. Clean up the local fetch.
rm -rf _release_in release_artifacts

# 5. Later, after legal sign-off: flip to public (interactive confirm).
modal volume get sapient-data /checkpoints/sapient1_llama/best.pt _release_in/best.pt
modal volume get sapient-data /parcellate.npz _release_in/parcellate.npz
python release.py --checkpoint _release_in/best.pt --parcellation _release_in/parcellate.npz \
                  --confirm --public
rm -rf _release_in release_artifacts
```

After step 3 the repo lives at https://huggingface.co/The-Sapient-Company/sapient-1-llama (private).

---

## Sapient-2 — same pattern, different data

After sapient-1 is uploaded, the sapient-2 pipeline is the same with these substitutions:

```bash
cd ~/Desktop/sapient-models/sapient2

# Stage 3a
bash data/download_ds004996.sh
bash data/download_ds001740.sh

# Stage 3b — forced alignment for HRI corpus
modal run data/forced_align.py --stim-root /data/raw/ds004996/stimuli
modal run data/forced_align.py --stim-root /data/raw/ds001740/stimuli

# Stage 3c — fMRI prep
python data/prepare_fmri.py --in data/raw/ds004996 --out /data/fmri/ds004996 \
    --tr <verify from dataset_description.json> --all-subjects
python data/prepare_fmri.py --in data/raw/ds001740 --out /data/fmri/ds001740 \
    --tr <verify from dataset_description.json> --all-subjects

# Stage 3d — features
modal run data/extract_video_features.py
modal run data/extract_audio_features.py
modal run data/extract_text_features.py

# Stage 3e — manifest (pooled across ds004996 + ds001740)
python data/manifest.py \
    --fmri-roots /data/fmri/ds004996 /data/fmri/ds001740 \
    --feat-root /data/features \
    --out /data/manifest.json

# Stage 4a — train from scratch
modal run train.py::main_scratch --config configs/sapient2_scratch_llama.yaml

# Stage 4b — fine-tune from sapient-1 (decisions #11, #12: lr=5e-5, drop subject_embed)
modal run train.py::main_ft --config configs/sapient2_ft_llama.yaml

# Stage 5 + 6 — eval and release, per variant
modal run eval.py --checkpoint ...
python release.py --variant scratch --confirm    # then --variant ft --confirm
```

---

## Total cost projection (both repos, full pipeline)

| Stage | sapient-1 | sapient-2 |
|---|---:|---:|
| 3b forced-align | $5–15 | $1–3 (less audio) |
| 3d features | $3–8 | $1–3 |
| 4 training | $100–150 | $200–300 (two lineages, but smaller data) |
| 5 eval | $0.50 | $1 |
| 6 release | $0 | $0 |
| **Per-repo total** | **~$110–175** | **~$205–310** |

**Combined: ~$315–485 to bring both models to a private HF release.**

The week-2 augmentation runs for sapient-1 (BOLD Moments + Narratives) add another ~$50–100 on top.

---

## When something breaks

1. **Modal download fails silently with truncated error** — the per-function logs in the Modal dashboard usually have stderr. CLI output truncates by default. Open the run URL printed by `modal run` to see the full trace.
2. **datalad-based downloads fail with no useful output** — likely git-annex version skew (apt's is too old) or missing git identity in the container. Both are fixed in `download_cneuromod.py`'s image build (datalad-installer + git config). If you copy that script as a template for a new datalad dataset, keep those `run_commands` lines.
3. **OpenNeuro version pin doesn't sync** — `s3://openneuro.org/<ds>/versions/<X>/` is *not* populated for every dataset. The S3 mirror only guarantees the latest revision at the dataset root. For true version pinning, switch to a datalad-based clone of the OpenNeuroDatasets GitHub mirror + `git checkout <version>` (see `download_cneuromod.py`).
4. **Modal job runs but errors inside** — check the Modal dashboard for the per-function logs. Common: forced-alignment failed for a stimulus → re-run with `--stim-root` narrowed to that file.
5. **Training diverges** — kill it from the dashboard (NOT pkill from CLI — see below). Loss going up usually means lr is too high; lower lr in the YAML and resume from `latest.pt`.
6. **HF upload fails on rate limit** — `release.py` is idempotent; re-run.

## Hard-learned operational rules

These are mistakes I (Claude) made on 2026-05-26 so you don't have to.

1. **NEVER `pkill -f "modal run"`.** `modal run --detach` is leakier than the docs imply. The local CLI process holds a cancellation handle to the remote function; killing it with SIGTERM cancels the in-flight Modal job. I lost 78 GB of BOLD Moments progress and 13 GB of Narratives progress this way. If you need to free up local shells, just close the terminal window (SIGHUP usually doesn't cancel) or wait for `--detach` to actually return after image build.
2. **Modal volume contents persist across cancelled runs.** Re-launching is always free in the sense that aws s3 sync (and datalad get) will resume from what's already on the volume. No bytes wasted on the dataset side — but Modal CPU time during the cancelled run *is* wasted.
3. **`modal volume ls /raw/<ds>` shows top-level entries only.** Active downloads append files *inside* existing subdirectories, so the top-level count can be flat for hours while the dataset still grows in size. Use `modal app logs <app_id>` if you need to see actual progress bytes.
4. **Build images with everything you'll need.** Modal containers default to root with no `sudo`, no git identity, no useful timezone. If a third-party tool (datalad-installer, git annex init, etc.) needs any of those, set them in the image's `apt_install` / `run_commands` — not at function runtime.
5. **`capture_output=True` is non-optional for subprocess inside Modal functions.** Without it, third-party tool stderr is silently dropped; the function fails with a generic CalledProcessError and you spend an hour guessing at root cause. HF accepts repeated uploads to the same repo.

---

## File-by-file reference

If you need to look at any specific piece of the pipeline:

| Concern | File |
|---|---|
| Architecture decisions (the 15 locked rules) | `engineering-outline/00_HANDOFF_README.md` |
| Build spec, section-by-section | `engineering-outline/08_sapient1_sapient2_spec.md` |
| Dataset licensing details | `engineering-outline/09_data_sources_and_licensing.md` |
| File map of the repos | `engineering-outline/CODEBASE_MAP.md` |
| Day-0 readiness checklist | `engineering-outline/PREREQUISITES.md` |
| Model code (the 178M-param transformer) | `sapient1/sapient1/model.py` |
| Training loop | `sapient1/train.py` |
| Eval | `sapient1/eval.py` |
| HF release | `sapient1/release.py` |

All sapient-2 paths mirror sapient-1.
