# sapienteval/modal — Modal apps for the Sapient Cognitive Eval fine-tune pipeline

Modal apps live here. Workspace: `robert-16572`. All apps and volumes in this
directory are isolated from the existing `sapient-v2` / `sapient-scan-trigger`
pipelines.

## Apps

- `sapienteval-fmriprep` — runs fMRIPrep (CPU, `nipreps/fmriprep:24.1.1`) on
  raw BIDS subjects from the `sapienteval-data` volume. Flags:
  `--fs-no-reconall --output-spaces MNI152NLin2009cAsym:res-2` (volumetric
  MNI only — Schaefer-400 parcellation does not need cortical surfaces, so we
  skip the multi-hour surface reconstruction step entirely).
  - 8 vCPU, 16 GB RAM, 6h hard timeout per subject.
  - ~$1/hr × ~4h × 5 pilot subjects (run in parallel) ≈ ~$20 for the pilot.
- `sapienteval-finetune` — fine-tunes the Sapient brain-encoding model on
  aligned `(text, BOLD)` pairs. A100-40GB, ~15 GPU-h for the pilot.
- `sapienteval-inference` — serves the trained checkpoint behind a POST
  `/score` endpoint. Called from `api/sapienteval/score.ts` when
  `SAPIENT_SCORER=v1`.

## Volumes

- `sapienteval-data` — raw BIDS dataset + fMRIPrep derivatives + parcellated
  `.npy` arrays + aligned stimulus pairs. Re-downloaded fresh, not shared with
  the video-scan pipeline.
- `sapienteval-checkpoints` — base brain-encoding checkpoint (re-downloaded
  fresh, not shared with `sapient-v2`), fine-tuned `sapient_v0.5_pilot.pt` and
  `sapient_v1.pt`, and `mapping_head.npz`.

Both volumes are isolated from the existing video-scan pipeline.

## fMRIPrep runner — deploy and run

Deploy (registers the app + image with Modal):

```bash
modal deploy sapienteval/modal/fmriprep_runner.py
```

Run the pilot (default: `sub-01..sub-05` in parallel, one container per subject):

```bash
modal run sapienteval/modal/fmriprep_runner.py
```

Run a specific subset:

```bash
modal run sapienteval/modal/fmriprep_runner.py --subjects sub-01,sub-02
```

Or via the thin Python wrapper (defaults pull from `sapienteval/splits.json`
`pilot.subjects`):

```bash
python sapienteval/02_run_fmriprep_modal.py
python sapienteval/02_run_fmriprep_modal.py --subjects sub-01,sub-02
python sapienteval/02_run_fmriprep_modal.py --dry-run
```

## Inspecting the volume

List everything on the volume:

```bash
modal volume ls sapienteval-data
```

Pull a subject's preprocessed functional outputs locally:

```bash
modal volume get sapienteval-data /derivatives/fmriprep/sub-01/func/ ./local-inspection/
```

The key files for downstream Schaefer-400 parcellation are the
`*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz` volumes (and
their accompanying `*_desc-confounds_timeseries.tsv` for nuisance regression).
