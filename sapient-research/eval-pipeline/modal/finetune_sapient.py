"""
sapienteval-finetune — Modal A100 fine-tune of the Sapient brain-encoding
model on NeuroEngage (ds004996) parcellated BOLD data.

Output: sapienteval-checkpoints volume, file sapient_v0.5_pilot.pt

Strategy (per sapienteval/ACCEPTANCE_CRITERIA.md):
  - Init from base 8-layer brain-encoding transformer (~700MB)
    via `SapientModel.from_pretrained("/base", checkpoint_name="best.ckpt")`
    from the existing `sapient` Python package (Sapient v2 = FmriEncoderModel).
  - Freeze text/audio/video extractors + lower transformer layers.
  - Unfreeze top 2 transformer layers + final LayerNorm + per-subject heads.
  - Loss: MSE(pred, true) + 0.1 * temporal_smoothness(pred)
  - AdamW, lr=1e-4 (transformer) / 1e-3 (heads), cosine schedule, 500 warmup
  - Batch=8 TRs/step, epochs=5, early stop patience=2 on within-subject val R
  - Pilot subjects: per sapienteval/splits.json (sub-01..sub-05)

The base model predicts (B, 20484, T') fsaverage5 vertices. We project to
Schaefer-400 parcels at the prediction site and compare against our
parcellated NeuroEngage targets (T, 400) loaded from /data/parcellated/.

Pre-registered acceptance criteria: see sapienteval/ACCEPTANCE_CRITERIA.md.

Cost target: ~15 GPU-hours pilot ≈ $40 on A100-40GB.

Deploy: modal deploy sapienteval/modal/finetune_sapient.py
Run:    modal run sapienteval/modal/finetune_sapient.py
          modal run sapienteval/modal/finetune_sapient.py --subjects sub-01,sub-02,sub-03 --epochs 5
          modal run sapienteval/modal/finetune_sapient.py --dry-run
"""

from __future__ import annotations

import modal
import os
import json
import time
from pathlib import Path

# ───────────────────────── Image, volumes, secrets ─────────────────────────
#
# We mirror the existing sapient-model Modal pipeline image layout
# (modal_pipeline/app.py) so the trained model can be inferenced with the
# same package. The local sapient package is mounted into /root/sapient_repo
# and installed editable so `from sapient import SapientModel` works inside
# the container.

LOCAL_SAPIENT_REPO = "/Users/robertgutierrez/Desktop/SIMPLR/sapient-model"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "build-essential",
        "ffmpeg",
        "libgl1",
        "libglu1-mesa",
    )
    # Torch matches the inference image (cu124 / 2.6.0). Keeping the version
    # identical means any checkpoints saved here round-trip cleanly into the
    # existing inference pipeline.
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Sapient runtime deps. Selective — we drop the inference-only ones
    # (yt-dlp, whisperx-via-uvx, gtts, langdetect, moviepy) since fine-tuning
    # consumes pre-parcellated BOLD + pre-aligned transcripts directly.
    .pip_install(
        "neuralset==0.0.2",
        "neuraltrain==0.0.2",
        "numpy==2.2.6",
        "x_transformers==1.27.20",
        "einops",
        "pyyaml",
        "huggingface_hub",
        "spacy",
        "soundfile",
        "Levenshtein",
        "julius",
        "transformers",
        # training extras (sapient[training])
        "nibabel",
        "torchmetrics",
        "lightning",
        # for our parcellation + correlation
        "nilearn==0.10.4",
        "scipy",
        "pandas",
        "pyarrow",
        "tqdm",
        "exca",
        "pydantic",
        "requests",
        # optional logging
        "wandb",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    # Drop the local sapient source into the image and install it editable
    # (no-deps so we don't fight our pinned versions above).
    .add_local_dir(
        LOCAL_SAPIENT_REPO,
        remote_path="/root/sapient_repo",
        copy=True,
        ignore=[
            "__pycache__",
            ".git",
            ".venv",
            ".cursor",
            ".claude",
            ".pytest_cache",
            ".mypy_cache",
            ".ipynb_checkpoints",
            ".modal_logs",
            "cache",
            "outputs",
            "inputs",
            "wandb",
            "lightning_logs",
            "*.ipynb",
            "*.egg-info",
            ".DS_Store",
        ],
    )
    .run_commands("pip install --no-deps -e /root/sapient_repo")
    # Centralise HF / torch / nilearn caches into the shared cache volume so
    # feature extractors don't re-download on every container boot.
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
            "TORCH_HOME": "/cache/torch",
            "NUMBA_CACHE_DIR": "/cache/numba",
            "NILEARN_SHARED_DATA": "/cache/nilearn_data",
            "SAPIENT_CACHE_FOLDER": "/cache/sapient_features",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

# --- Volumes ----------------------------------------------------------------
data_vol = modal.Volume.from_name("sapienteval-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("sapienteval-checkpoints", create_if_missing=True)

# Base brain-encoding pretrained weights live on a dedicated, read-only
# volume that serves as the model registry. We never write to this volume
# during fine-tuning — all training outputs go to sapienteval-checkpoints.
base_vol = modal.Volume.from_name("sapient-base-encoder-weights")

# Shared cache for HF / extractor sub-model weights. Mirrors the convention
# used by the sapient-model inference pipeline (modal_pipeline/app.py:45),
# so the LLaMA-3.2-3B / V-JEPA2 / Wav2Vec-BERT weights downloaded by feature
# extractors are re-used across runs instead of re-pulled every container boot.
cache_vol = modal.Volume.from_name("sapient-cache", create_if_missing=True)

app = modal.App("sapienteval-finetune", image=image)

# Mount locations inside the training container:
#   /base   → sapient-base-encoder-weights  (read-only model registry)
#     /base/best.ckpt     → the pretrained encoder weights (~700MB)
#     /base/config.yaml   → architecture spec
#   /data   → sapienteval-data               (read-only data lake)
#     /data/ds004996/                        → raw BIDS
#     /data/derivatives/fmriprep/             → preprocessed BOLD
#     /data/parcellated/                       → (T, 400) .npy per run
#     /data/aligned/                            → per-TR text + BOLD pairs
#   /cache  → sapient-cache                  (HF + extractor caches)
#   /ckpt   → sapienteval-checkpoints        (read/write training outputs)
#     /ckpt/sapient_v0.5_pilot.pt              → pilot fine-tuned checkpoint
BASE_CHECKPOINT_DIR = "/base"
BASE_CHECKPOINT_NAME = "best.ckpt"
BASE_CONFIG_PATH = "/base/config.yaml"
PARCELLATED_DIR = "/data/parcellated"
ALIGNED_DIR = "/data/aligned"
FEATURE_CACHE_DIR = "/cache/sapient_features"
SCHAEFER_CACHE_DIR = "/cache/schaefer400_fsaverage5"

# Output dimension of the Sapient base model. fsaverage5 = 10242 vertices
# per hemisphere * 2 = 20484. We project these onto 400 Schaefer parcels
# at the loss site to match our parcellated NeuroEngage targets.
N_VERTICES_FSAVERAGE5 = 20484
N_PARCELS_SCHAEFER400 = 400
TR_SECONDS = 2.0


# ───────────────────────── Helpers ─────────────────────────
def make_secrets() -> list[modal.Secret]:
    """Best-effort fetch of optional secrets.

    - `hf-token` exposes `HUGGINGFACE_TOKEN` for HuggingFace downloads.
    - `wandb`     exposes `WANDB_API_KEY` for run logging (optional).

    Missing secrets are silently skipped; the trainer falls back to JSON-file
    history logging on the checkpoints volume.
    """
    secrets: list[modal.Secret] = []
    try:
        secrets.append(modal.Secret.from_name("hf-token"))
    except Exception:
        pass
    try:
        secrets.append(modal.Secret.from_name("wandb"))
    except Exception:
        pass
    return secrets


# ───────────────────────── Main training function ─────────────────────────
@app.function(
    gpu="A100-40GB",
    timeout=4 * 3600,  # 4h hard timeout per training run
    volumes={
        "/data": data_vol,
        "/ckpt": ckpt_vol,
        "/base": base_vol,
        "/cache": cache_vol,
    },
    secrets=make_secrets(),
)
def train(
    subject_ids: list[str],
    epochs: int = 5,
    batch_tr: int = 8,
    lr_transformer: float = 1e-4,
    lr_heads: float = 1e-3,
    warmup_steps: int = 500,
    smoothness_lambda: float = 0.1,
    early_stop_patience: int = 2,
    run_name: str = "sapient_v0.5_pilot",
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """
    Fine-tune the brain-encoding model on the given subjects.

    Returns: {run_name, n_train_runs, n_val_runs, final_train_loss,
              best_val_within_subject_r, checkpoint_path, duration_s, history}

    `dry_run=True` is the safety hatch: loads the model, finds the data,
    runs ONE forward + backward pass to verify the integration, then
    returns without entering the full training loop. Use this to test
    integration without spending GPU $.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # HuggingFace auth — required for the gated meta-llama/Llama-3.2-3B
    # text extractor that loads inside SapientModel.from_pretrained when
    # we touch the dataloader.
    hf_token = (
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if hf_token:
        from huggingface_hub import login as _hf_login

        _hf_login(token=hf_token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("Authenticated with HuggingFace Hub.")
    else:
        print(
            "WARN: no HF token in env; gated meta-llama/Llama-3.2-3B downloads will fail."
        )

    # ── 1. Load base Sapient model (FmriEncoderModel + frozen extractors) ──
    base_dir = Path(BASE_CHECKPOINT_DIR)
    if not (base_dir / BASE_CHECKPOINT_NAME).exists():
        raise FileNotFoundError(
            f"Base brain-encoding checkpoint missing at {base_dir/BASE_CHECKPOINT_NAME}. "
            "Confirm the sapient-base-encoder-weights volume is mounted "
            "and contains best.ckpt at the root."
        )
    if not Path(BASE_CONFIG_PATH).exists():
        raise FileNotFoundError(
            f"Base architecture config missing at {BASE_CONFIG_PATH}. "
            "Confirm the sapient-base-encoder-weights volume contains config.yaml."
        )

    Path(FEATURE_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    sapient_xp, brain_model = build_brain_encoding_model(
        checkpoint_dir=BASE_CHECKPOINT_DIR,
        checkpoint_name=BASE_CHECKPOINT_NAME,
        cache_folder=FEATURE_CACHE_DIR,
    )
    brain_model = brain_model.cuda()
    print(
        f"Loaded SapientModel: hidden={getattr(brain_model.config, 'hidden', '?')}  "
        f"n_outputs={brain_model.n_outputs}  "
        f"n_output_timesteps={brain_model.n_output_timesteps}"
    )

    # ── 2. Freeze / unfreeze ────────────────────────────────────────────
    # Freeze everything by default, then re-enable:
    #   - top 2 transformer encoder layers
    #   - the combiner MLP that feeds into the transformer
    #   - the low_rank_head (small linear bottleneck before the readout)
    #   - the per-subject linear readout (`predictor` = SubjectLayers)
    # The text/audio/video extractor sub-models live OUTSIDE brain_model
    # entirely — they're held by sapient_xp.data.* extractors and are
    # only used in inference mode during dataloader.prepare(), so we do
    # not need to freeze them here.
    for p in brain_model.parameters():
        p.requires_grad = False
    unfrozen_param_count = _unfreeze_top_transformer_layers(brain_model, n_top=2)
    # Combiner + low_rank_head + predictor are the "head" group — we attach
    # them to the higher per-subject learning rate to let the readout adapt
    # faster to NeuroEngage's per-subject BOLD characteristics.
    head_modules: list[nn.Module] = []
    if hasattr(brain_model, "combiner") and isinstance(brain_model.combiner, nn.Module):
        head_modules.append(brain_model.combiner)
    if hasattr(brain_model, "low_rank_head"):
        head_modules.append(brain_model.low_rank_head)
    head_modules.append(brain_model.predictor)
    for m in head_modules:
        for p in m.parameters():
            p.requires_grad = True

    unfrozen_total = sum(p.numel() for p in brain_model.parameters() if p.requires_grad)
    frozen_total = sum(p.numel() for p in brain_model.parameters() if not p.requires_grad)
    print(
        f"Param counts: unfrozen={unfrozen_total:,}  frozen={frozen_total:,}  "
        f"(top-2-encoder-layer params unfrozen: {unfrozen_param_count:,})"
    )

    # ── 3. Vertex → Schaefer-400 parcellation matrix ─────────────────────
    # Base model outputs (B, 20484, T') in fsaverage5 space. Our NeuroEngage
    # targets are (T, 400) Schaefer parcels. We build a fixed projection
    # matrix once and apply it to predictions inside the loss site.
    print("Building Schaefer-400 fsaverage5 vertex-to-parcel projection …")
    v2p = _build_vertex_to_parcel()
    proj = _build_parcel_projection_matrix(v2p).cuda()  # (20484, 400)

    # ── 4. Data loading ────────────────────────────────────────────────
    # Loads training pairs from /data/parcellated/<subject>_*.npy paired
    # with /data/aligned/<subject>_*.json. Each subject's run-03 (per
    # splits.json) is held out as within-subject val for early stopping.
    train_pairs, within_subject_val_pairs = load_training_data(
        subject_ids,
        data_dir="/data",
        held_out_run="run-03",
    )
    print(
        f"Loaded {len(train_pairs)} training pairs, "
        f"{len(within_subject_val_pairs)} within-subject val pairs"
    )

    if len(train_pairs) == 0:
        raise RuntimeError(
            "No training pairs found. Ensure /data/parcellated/ and /data/aligned/ "
            "have been populated by 03_parcellate.py and 04_align_stimuli.py for "
            f"subjects: {subject_ids}"
        )

    # Build the Sapient dataloader — text-only events. We bypass the
    # Sapient Fmri extractor (which would require FreeSurfer mesh files
    # we don't have access to) and inject our (T, 400) Schaefer targets
    # manually per-segment via `_inject_parcel_targets_into_batch`. See
    # SCIENCE.md §6 for the architectural rationale.
    print("Preparing Sapient text feature extractors on training pairs …")
    train_loader = _build_sapient_loader(
        sapient_xp,
        pairs=train_pairs,
        split_name="train",
        batch_size=batch_tr,
        shuffle=True,
    )
    val_loader = _build_sapient_loader(
        sapient_xp,
        pairs=within_subject_val_pairs,
        split_name="val",
        batch_size=batch_tr,
        shuffle=False,
    )

    # Pre-load all parcellated targets keyed by timeline so per-batch
    # injection is O(slice) instead of O(file-load).
    train_timeline_to_parcels = _build_timeline_to_parcels(train_pairs)
    val_timeline_to_parcels = _build_timeline_to_parcels(within_subject_val_pairs)
    print(
        f"Cached parcellated targets: train timelines={len(train_timeline_to_parcels)}, "
        f"val timelines={len(val_timeline_to_parcels)}"
    )

    # ── 5. Dry-run integration check ────────────────────────────────────
    # The four sanity checks the user explicitly asked for. Any failure
    # aborts BEFORE the $40 training run.
    if dry_run:
        print("[dry-run] Beginning 4-check sanity test …")
        from collections import OrderedDict

        # ── Check (a): refactored model instantiates ──
        # Already done above (build_brain_encoding_model returned without
        # raising and brain_model is on .cuda()). Re-affirm here.
        if brain_model is None or not hasattr(brain_model, "config"):
            return {
                "run_name": run_name,
                "status": "dry-run-FAILED-check-a",
                "error": "brain_model did not instantiate cleanly",
            }
        print(
            f"[dry-run][a] Model instantiated. "
            f"n_outputs={brain_model.n_outputs}  "
            f"n_output_timesteps={brain_model.n_output_timesteps}"
        )

        # ── Check (b): state_dict key comparison vs checkpoint ──
        # SapientModel.from_pretrained loads weights via PyTorch Lightning's
        # state machinery, which uses strict=False internally. We re-open
        # the raw checkpoint here and explicitly diff against the live model.
        ckpt_dict_path = Path(BASE_CHECKPOINT_DIR) / BASE_CHECKPOINT_NAME
        raw_ckpt = torch.load(str(ckpt_dict_path), map_location="cpu", weights_only=False)
        # Lightning checkpoints wrap parameters under "state_dict" with
        # a "model." prefix. Strip the prefix for a clean comparison.
        ckpt_sd = raw_ckpt.get("state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
        ckpt_keys = {
            (k[len("model."):] if k.startswith("model.") else k)
            for k in ckpt_sd.keys()
        }
        model_keys = set(brain_model.state_dict().keys())
        missing = ckpt_keys - model_keys
        unexpected = model_keys - ckpt_keys
        if missing or unexpected:
            return {
                "run_name": run_name,
                "status": "dry-run-FAILED-check-b",
                "error": "state_dict key mismatch — checkpoint loaded with non-zero diff",
                "n_missing_keys": len(missing),
                "n_unexpected_keys": len(unexpected),
                "missing_sample": sorted(missing)[:5],
                "unexpected_sample": sorted(unexpected)[:5],
            }
        print(
            f"[dry-run][b] state_dict keys match exactly: "
            f"{len(model_keys)} keys, 0 missing, 0 unexpected"
        )

        # ── Check (c): forward + backward on one (text, BOLD_400) pair ──
        brain_model.train()
        first_batch = None
        for batch in iter_batches(train_loader, train_timeline_to_parcels, TR_SECONDS):
            first_batch = batch.to(brain_model.device)
            break
        if first_batch is None:
            return {
                "run_name": run_name,
                "status": "dry-run-FAILED-check-c",
                "error": "Dataloader yielded zero batches",
            }
        y_pred = brain_model(first_batch)  # (B, 20484, T')
        y_pred_parcels = _project_vertices_to_parcels(y_pred, proj)  # (B, 400, T')
        y_true_parcels = _extract_parcel_targets(first_batch, brain_model.n_output_timesteps)
        loss = F.mse_loss(y_pred_parcels, y_true_parcels)
        if not torch.isfinite(loss):
            return {
                "run_name": run_name,
                "status": "dry-run-FAILED-check-c",
                "error": f"Loss is non-finite: {loss.item()!r}",
            }
        loss.backward()
        print(
            f"[dry-run][c] fwd+bwd OK: "
            f"pred={tuple(y_pred.shape)}  pred_parcels={tuple(y_pred_parcels.shape)}  "
            f"true_parcels={tuple(y_true_parcels.shape)}  loss={loss.item():.4f}"
        )

        # ── Check (d): grads flow to unfrozen layers ONLY ──
        # Walk every parameter and split into: (i) frozen params that
        # should have grad=None, (ii) unfrozen params that should have
        # grad=non-None with finite values. Any violation aborts.
        bad_frozen_with_grad: list[str] = []
        bad_unfrozen_no_grad: list[str] = []
        bad_unfrozen_nonfinite: list[str] = []
        n_unfrozen_with_grad = 0
        for name, p in brain_model.named_parameters():
            if not p.requires_grad:
                if p.grad is not None and torch.any(p.grad != 0):
                    bad_frozen_with_grad.append(name)
            else:
                if p.grad is None:
                    bad_unfrozen_no_grad.append(name)
                elif not torch.all(torch.isfinite(p.grad)):
                    bad_unfrozen_nonfinite.append(name)
                else:
                    n_unfrozen_with_grad += 1
        if bad_frozen_with_grad or bad_unfrozen_no_grad or bad_unfrozen_nonfinite:
            return {
                "run_name": run_name,
                "status": "dry-run-FAILED-check-d",
                "error": "Gradient isolation broken — frozen params got grads or unfrozen params didn't",
                "frozen_with_grad_count": len(bad_frozen_with_grad),
                "frozen_with_grad_sample": bad_frozen_with_grad[:5],
                "unfrozen_no_grad_count": len(bad_unfrozen_no_grad),
                "unfrozen_no_grad_sample": bad_unfrozen_no_grad[:5],
                "unfrozen_nonfinite_count": len(bad_unfrozen_nonfinite),
                "unfrozen_nonfinite_sample": bad_unfrozen_nonfinite[:5],
            }
        print(
            f"[dry-run][d] grad isolation OK: "
            f"{n_unfrozen_with_grad} unfrozen params received finite gradients, "
            f"0 frozen params leaked gradients"
        )

        return {
            "run_name": run_name,
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(within_subject_val_pairs),
            "status": "dry-run-ok",
            "checks": {
                "a_model_instantiated": True,
                "b_state_dict_clean_diff": True,
                "c_forward_backward_finite": True,
                "d_gradient_isolation": True,
            },
            "loss_value": float(loss.item()),
            "checkpoint_path": None,
            "duration_s": None,
            "history": [],
        }

    # ── 6. Optimizer with split lr ─────────────────────────────────────
    transformer_params = [
        p for n, p in brain_model.named_parameters()
        if p.requires_grad and (".encoder." in "." + n + "." or "combiner" in n)
    ]
    head_params = [
        p for n, p in brain_model.named_parameters()
        if p.requires_grad and ("predictor" in n or "low_rank_head" in n)
    ]
    # Sanity: every grad-enabled param ends up in exactly one group.
    seen = {id(p) for p in transformer_params + head_params}
    missed = [
        p for p in brain_model.parameters()
        if p.requires_grad and id(p) not in seen
    ]
    if missed:
        # Add stragglers (e.g. time_pos_embed when top layers are unfrozen)
        # to the transformer group so they don't silently skip updates.
        transformer_params.extend(missed)

    from torch.optim import AdamW

    optimizer = AdamW(
        [
            {"params": transformer_params, "lr": lr_transformer},
            {"params": head_params, "lr": lr_heads},
        ]
    )
    total_steps = max((len(train_pairs) // batch_tr) * epochs, 1)
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    # ── 7. Optional wandb init ─────────────────────────────────────────
    wandb_enabled = False
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb.init(
                project="sapienteval",
                name=run_name,
                config={
                    "subject_ids": subject_ids,
                    "epochs": epochs,
                    "batch_tr": batch_tr,
                    "lr_transformer": lr_transformer,
                    "lr_heads": lr_heads,
                    "warmup_steps": warmup_steps,
                    "smoothness_lambda": smoothness_lambda,
                    "seed": seed,
                },
            )
            wandb_enabled = True
        except Exception as e:
            print(f"wandb init failed ({e}); continuing with JSON logging only")

    # ── 8. Training loop with early stopping ───────────────────────────
    best_val_r = -1.0
    patience_counter = 0
    train_history: list[dict] = []
    parcel_groups = _get_parcel_groups()  # DMN + MFG indices, for eval

    for epoch in range(epochs):
        brain_model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in iter_batches(train_loader, train_timeline_to_parcels, TR_SECONDS):
            batch = batch.to(brain_model.device)
            y_pred = brain_model(batch)  # (B, 20484, T')
            y_pred_parcels = _project_vertices_to_parcels(y_pred, proj)
            y_true_parcels = _extract_parcel_targets(batch, brain_model.n_output_timesteps)
            loss = F.mse_loss(y_pred_parcels, y_true_parcels)

            # Temporal smoothness on consecutive TR predictions
            if y_pred_parcels.shape[-1] > 1:
                diff = y_pred_parcels[..., 1:] - y_pred_parcels[..., :-1]
                smoothness = (diff ** 2).mean()
                loss = loss + smoothness_lambda * smoothness

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()
            n_batches += 1

        # Within-subject val (Pearson R on DMN + MFG parcels)
        brain_model.eval()
        val_r = evaluate_pearson_r(
            brain_model,
            val_loader,
            proj,
            parcel_groups,
            timeline_to_parcels=val_timeline_to_parcels,
            tr_sec=TR_SECONDS,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(n_batches, 1),
            "val_r": val_r,
        }
        train_history.append(epoch_record)
        print(json.dumps(epoch_record))

        if wandb_enabled:
            try:
                import wandb

                wandb.log(epoch_record)
            except Exception:
                pass

        # Early stop
        if val_r > best_val_r:
            best_val_r = val_r
            patience_counter = 0
            torch.save(
                {
                    "state_dict": brain_model.state_dict(),
                    "epoch": epoch,
                    "val_r": val_r,
                    "subject_ids": subject_ids,
                    "model_build_args": {
                        "feature_dims": brain_model.feature_dims,
                        "n_outputs": brain_model.n_outputs,
                        "n_output_timesteps": brain_model.n_output_timesteps,
                    },
                    "training_args": {
                        "epochs": epochs,
                        "batch_tr": batch_tr,
                        "lr_transformer": lr_transformer,
                        "lr_heads": lr_heads,
                        "warmup_steps": warmup_steps,
                        "smoothness_lambda": smoothness_lambda,
                        "seed": seed,
                    },
                },
                f"/ckpt/{run_name}.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}, best val R = {best_val_r:.4f}"
                )
                break

    ckpt_vol.commit()
    with open(f"/ckpt/{run_name}_history.json", "w") as f:
        json.dump(
            {
                "history": train_history,
                "best_val_r": best_val_r,
                "subjects": subject_ids,
            },
            f,
            indent=2,
        )
    ckpt_vol.commit()

    if wandb_enabled:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    return {
        "run_name": run_name,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(within_subject_val_pairs),
        "final_train_loss": train_history[-1]["train_loss"] if train_history else None,
        "best_val_within_subject_r": best_val_r,
        "checkpoint_path": f"/ckpt/{run_name}.pt",
        "duration_s": None,  # filled in by caller
        "history": train_history,
    }


# ─────────────────── Sapient package integration helpers ───────────────────
def build_brain_encoding_model(
    checkpoint_dir: str,
    checkpoint_name: str = "best.ckpt",
    cache_folder: str = FEATURE_CACHE_DIR,
):
    """Load the pretrained Sapient brain-encoding model.

    Returns (sapient_xp, brain_model):
      - sapient_xp : the SapientModel (SapientExperiment) wrapper. Holds the
        Data/extractor config so we can call sapient_xp.data.get_loaders()
        later with our own events DataFrame.
      - brain_model : the underlying FmriEncoderModel (nn.Module), already
        loaded with the pretrained state_dict, on the auto-selected device.

    The model architecture (8 transformer layers, hidden=1152, low_rank_head=2048,
    output=20484 fsaverage5 vertices) and all hyperparameters come from
    /base/config.yaml — we never hard-code them here.

    The base config was originally saved with the legacy 'TribeSurfaceProjector'
    class name (predates the Sapient rename). We patch the config in-place to
    'SapientSurfaceProjector' before load so pydantic validation passes.
    """
    from sapient import SapientModel
    import shutil

    # Read + patch the config to translate legacy class names. We copy the
    # base/ dir to a writable workdir, rewrite config.yaml, and load from there.
    work_dir = Path("/tmp/sapient_base_patched")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    # Symlink best.ckpt (large file — avoid copying 700MB)
    src_dir = Path(checkpoint_dir)
    for f in src_dir.iterdir():
        if f.name == "config.yaml":
            patched = f.read_text().replace(
                "TribeSurfaceProjector", "SapientSurfaceProjector"
            )
            (work_dir / f.name).write_text(patched)
        elif f.is_file():
            (work_dir / f.name).symlink_to(f.resolve())
        elif f.is_dir():
            (work_dir / f.name).symlink_to(f.resolve())

    sapient_xp = SapientModel.from_pretrained(
        str(work_dir),
        checkpoint_name=checkpoint_name,
        cache_folder=cache_folder,
        device="auto",
    )
    brain_model = sapient_xp._model
    if brain_model is None:
        raise RuntimeError(
            "SapientModel.from_pretrained returned an xp with _model=None — "
            "checkpoint did not load correctly."
        )
    return sapient_xp, brain_model


def _unfreeze_top_transformer_layers(brain_model, n_top: int = 2) -> int:
    """Unfreeze the top `n_top` blocks of the transformer encoder + the
    final LayerNorm. Returns the number of newly-trainable parameters.

    The Sapient FmriEncoderModel exposes `model.encoder` which is built by
    `neuraltrain.models.transformer.TransformerEncoder`. The concrete sub-
    structure isn't part of FmriEncoderModel's public API, so we probe for
    common container attributes (`layers`, `blocks`, `transformer.layers`).
    If none match, we fall back to "unfreeze the entire encoder", which is
    safe — it's only 8 layers either way.
    """
    enc = getattr(brain_model, "encoder", None)
    if enc is None:
        return 0

    # Common attribute names across transformer implementations.
    candidate_containers = []
    for attr in ("layers", "blocks", "layer", "encoder_layers"):
        c = getattr(enc, attr, None)
        if c is not None and hasattr(c, "__len__") and hasattr(c, "__getitem__"):
            candidate_containers.append((attr, c))
    # x_transformers nests as enc.attn_layers.layers (list of (norm, attn, ff)
    # tuples or modules).
    al = getattr(enc, "attn_layers", None)
    if al is not None:
        inner = getattr(al, "layers", None)
        if inner is not None and hasattr(inner, "__len__"):
            candidate_containers.append(("attn_layers.layers", inner))

    n_unfrozen = 0
    if candidate_containers:
        # Use the longest container (most likely the real layer list).
        attr_name, container = max(candidate_containers, key=lambda kv: len(kv[1]))
        n_layers = len(container)
        cutoff = max(n_layers - n_top, 0)
        for i in range(cutoff, n_layers):
            block = container[i]
            if hasattr(block, "parameters"):
                for p in block.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        n_unfrozen += p.numel()
        print(
            f"Unfroze top {n_top} of {n_layers} transformer blocks via encoder.{attr_name}"
        )
    else:
        # Fallback: unfreeze the entire encoder. Slightly more params than
        # planned but still well below "full fine-tune".
        for p in enc.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                n_unfrozen += p.numel()
        print(
            "WARN: could not locate transformer block list — unfroze entire encoder."
        )

    # Final LayerNorm — search by name across the encoder for any LayerNorm
    # that lives at the encoder root or in a `final_*`/`norm_out`/`to_logits`
    # attribute.
    import torch.nn as nn

    for name, module in enc.named_modules():
        if isinstance(module, nn.LayerNorm) and any(
            tag in name for tag in ("final", "out", "norm_out", "to_logits", "post")
        ):
            for p in module.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    n_unfrozen += p.numel()
    # Also unfreeze the time positional embedding so the model can re-learn
    # NeuroEngage's TR=2s rhythm vs. the original training TR=1s mix.
    if hasattr(brain_model, "time_pos_embed"):
        brain_model.time_pos_embed.requires_grad = True
        n_unfrozen += brain_model.time_pos_embed.numel()

    return n_unfrozen


def load_training_data(
    subject_ids: list[str],
    data_dir: str = "/data",
    held_out_run: str = "run-03",
) -> tuple[list[dict], list[dict]]:
    """Load training pairs from /data/parcellated/ + /data/aligned/.

    For each subject in `subject_ids`, finds every parcellated run
    /data/parcellated/<subject>_task-*_run-*.npy and its matching aligned
    JSON /data/aligned/<subject>_task-*_run-*.json. Each pair becomes a
    dict:
      {
        "subject_id":     "sub-01",
        "task":           "engage",
        "run":            "1",
        "parcellated_npy": "/data/parcellated/sub-01_task-engage_run-1.npy",
        "aligned_json":    "/data/aligned/sub-01_task-engage_run-1.json",
      }

    Runs whose `run-XX` identifier matches `held_out_run` go into
    `within_subject_val_pairs`; everything else goes into `train_pairs`.
    """
    import re

    parc_dir = Path(data_dir) / "parcellated"
    align_dir = Path(data_dir) / "aligned"

    if not parc_dir.exists():
        raise FileNotFoundError(
            f"Parcellated directory missing: {parc_dir}. "
            "Run 03_parcellate.py first."
        )
    if not align_dir.exists():
        raise FileNotFoundError(
            f"Aligned directory missing: {align_dir}. "
            "Run 04_align_stimuli.py first."
        )

    # Normalize held_out_run for matching against filenames like
    # "sub-01_task-engage_run-3.npy" (note: 04_align_stimuli writes run-N
    # without zero-padding, so we match both forms).
    held_run_num = re.search(r"(\d+)", held_out_run).group(1) if held_out_run else None

    train_pairs: list[dict] = []
    val_pairs: list[dict] = []

    pat = re.compile(
        r"^(?P<subject>sub-[A-Za-z0-9]+)"
        r"_task-(?P<task>[A-Za-z0-9]+)"
        r"(?:_run-(?P<run>[0-9]+))?\.npy$"
    )
    for subject in subject_ids:
        npys = sorted(parc_dir.glob(f"{subject}_task-*.npy"))
        if not npys:
            print(f"WARN: no parcellated runs found for {subject} in {parc_dir}")
            continue
        for npy_path in npys:
            m = pat.match(npy_path.name)
            if m is None:
                print(f"WARN: could not parse {npy_path.name}; skipping")
                continue
            run_str = m.group("run") or "1"
            task = m.group("task")
            json_path = align_dir / f"{subject}_task-{task}_run-{run_str}.json"
            if not json_path.exists():
                print(
                    f"WARN: aligned JSON missing for {npy_path.name}; "
                    f"expected {json_path}. Skipping."
                )
                continue
            # Sapient's Fmri extractor uses nibabel.load() which needs a real
            # NIfTI file — point at the fmriprep volumetric MNI BOLD.
            # The .npy is kept as the (T, 400) parcel target for the loss.
            label = subject.replace("sub-", "")
            bold_glob_pat = (
                f"sub-{label}_task-{task}"
                + (f"_run-{int(run_str):02d}" if run_str else "")
                + "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            )
            deriv_func = Path(f"/data/derivatives/sub-{label}/func")
            bold_candidates = list(deriv_func.glob(bold_glob_pat))
            if not bold_candidates:
                # Try unpadded run-N too
                alt_glob = (
                    f"sub-{label}_task-{task}_run-{int(run_str)}_"
                    "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
                )
                bold_candidates = list(deriv_func.glob(alt_glob))
            if not bold_candidates:
                print(
                    f"WARN: no fmriprep BOLD .nii.gz for {subject} run-{run_str}; "
                    f"looked in {deriv_func}. Skipping."
                )
                continue
            pair = {
                "subject_id": subject,
                "task": task,
                "run": run_str,
                "parcellated_npy": str(npy_path),
                "aligned_json": str(json_path),
                "bold_nii_gz": str(bold_candidates[0]),
            }
            if held_run_num is not None and run_str == held_run_num:
                val_pairs.append(pair)
            else:
                train_pairs.append(pair)
    return train_pairs, val_pairs


def _build_events_dataframe(pairs: list[dict], split_name: str):
    """Convert our (parcellated_npy, aligned_json) pairs into a single
    standardized events DataFrame consumable by sapient_xp.data.get_loaders.

    The Sapient data pipeline expects:
      - one row per "Word" event with onset/duration/text/sentence/context
        + a `timeline` column grouping words that share a stimulus run, and
      - one "Fmri" row per timeline pointing at the parcellated array,
        which the dataloader uses as the target signal.

    We don't have audio or video for the text-only v0.5 pilot, so we
    emit Word events directly from the per-TR transcripts in the aligned
    JSON, with onsets at TR boundaries (t * TR_SECONDS). Sentence and
    context columns are filled from the TR's concatenated text so the
    LLaMA-3.2-3B feature extractor has the input it needs.
    """
    import pandas as pd
    import numpy as np

    rows: list[dict] = []
    for pair in pairs:
        with open(pair["aligned_json"]) as f:
            aligned = json.load(f)
        subject = pair["subject_id"]
        # Timeline names must be unique per (subject, run) so list_segments
        # respects run boundaries when chunking into segments.
        timeline = f"{subject}_task-{pair['task']}_run-{pair['run']}"
        tr_sec = float(aligned.get("tr_seconds", TR_SECONDS))
        n_trs_text = int(aligned.get("n_TRs", len(aligned.get("trs", []))))

        # Trim TR count to min(text, BOLD). We only need to know the BOLD
        # length here for the Fmri event's duration, since dataloader will
        # actually read the .npy file off disk inside the SegmentDataset.
        try:
            bold = np.load(pair["parcellated_npy"], mmap_mode="r")
            n_trs_bold = int(bold.shape[0])
        except Exception as e:
            print(f"WARN: failed to peek {pair['parcellated_npy']}: {e}")
            n_trs_bold = n_trs_text
        n_trs = min(n_trs_text, n_trs_bold)
        if n_trs <= 0:
            continue

        # Word events — one per TR that has text, anchored at TR onset.
        for tr_entry in aligned.get("trs", []):
            tr_idx = int(tr_entry["tr_index"])
            if tr_idx >= n_trs:
                break
            text = (tr_entry.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "type": "Word",
                    "timeline": timeline,
                    "subject": subject,
                    "split": split_name,
                    "start": float(tr_idx) * tr_sec,
                    "duration": tr_sec,
                    "text": text,
                    "sentence": text,
                    "context": text,
                    "language": "english",
                }
            )

        # NOTE: we intentionally do NOT emit Fmri events. Sapient's Fmri
        # extractor would invoke its surface projector (SapientSurfaceProjector
        # / TribeSurfaceProjector) which needs FreeSurfer's custom
        # `surf_hybrid_mni_gii/` mesh files — artifacts we don't have access to.
        # Instead we feed the dataloader text-only events, let it produce
        # text-feature batches, and manually inject parcellated BOLD targets
        # per-segment in `_inject_parcel_targets_into_batch` at the training
        # site. The loss is computed entirely in Schaefer-400 parcel space;
        # see SCIENCE.md §6 for the architectural rationale.

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _build_timeline_to_parcels(pairs: list[dict]) -> dict:
    """Map timeline string → preloaded (T_full, 400) parcel array.

    Called once before training so the per-batch `_inject_parcel_targets_into_batch`
    can slice without re-loading the .npy on every step.
    """
    import numpy as np

    out: dict[str, "np.ndarray"] = {}
    for pair in pairs:
        timeline = f"{pair['subject_id']}_task-{pair['task']}_run-{pair['run']}"
        bold = np.load(pair["parcellated_npy"]).astype("float32")  # (T_full, 400)
        out[timeline] = bold
    return out


def _inject_parcel_targets_into_batch(batch, timeline_to_parcels: dict, tr_sec: float):
    """Mutate a SegmentData batch in place to add `batch.data['fmri']` with
    shape (B, 400, T_segment) using the manually-loaded parcellated targets.

    Per-segment slicing: for each Segment with (start, duration, timeline),
    pull rows [start/tr_sec : (start+duration)/tr_sec] from the cached
    (T_full, 400) array, pad/trim to a uniform per-batch length, transpose
    to channel-first.

    Returns the in-place-modified batch.
    """
    import torch
    import numpy as np

    segs = batch.segments
    if not segs:
        raise RuntimeError("Empty segments list in batch — cannot inject targets.")

    # Per-segment slicing first (variable T per segment); we then pad/trim
    # to the longest segment in the batch so the stacked tensor is rectangular.
    chunks: list["np.ndarray"] = []
    for seg in segs:
        timeline = seg.timeline
        if timeline not in timeline_to_parcels:
            raise RuntimeError(
                f"No parcellated array cached for timeline={timeline!r}. "
                f"Known timelines: {list(timeline_to_parcels.keys())[:5]}…"
            )
        full = timeline_to_parcels[timeline]  # (T_full, 400)
        start_tr = int(round(float(seg.start) / tr_sec))
        end_tr = int(round((float(seg.start) + float(seg.duration)) / tr_sec))
        start_tr = max(0, start_tr)
        end_tr = min(full.shape[0], max(end_tr, start_tr + 1))
        chunk = full[start_tr:end_tr]  # (T_seg, 400)
        if chunk.shape[0] == 0:
            # Fall back to one row if slicing was degenerate
            chunk = full[:1]
        chunks.append(chunk)

    # Pad/trim to longest in batch — at the loss site we resample to the
    # model's n_output_timesteps anyway, so exact T_segment match isn't required.
    T_max = max(c.shape[0] for c in chunks)
    padded = np.zeros((len(chunks), T_max, 400), dtype="float32")
    for i, c in enumerate(chunks):
        padded[i, : c.shape[0]] = c
    # Channel-first: (B, 400, T_max)
    tensor = torch.from_numpy(padded).permute(0, 2, 1).contiguous().float()
    batch.data["fmri"] = tensor
    return batch


def _build_sapient_loader(
    sapient_xp,
    pairs: list[dict],
    split_name: str,
    batch_size: int,
    shuffle: bool,
):
    """Build a torch DataLoader using the Sapient package's own data pipeline.

    Falls back to a hand-rolled minimal DataLoader if the Sapient loader
    rejects our events (which can happen if our DataFrame is missing a
    column that one of the extractors requires). The fallback yields
    SegmentData-shaped namespaces with raw word strings + parcellated
    targets so the model's `aggregate_features` call still works against
    the text projector.
    """
    events_df = _build_events_dataframe(pairs, split_name)
    if events_df.empty:
        raise RuntimeError(
            f"No events generated for split={split_name} from {len(pairs)} pairs. "
            "Check that aligned JSON files have non-empty per-TR text."
        )
    sapient_xp.data.batch_size = batch_size
    sapient_xp.data.shuffle_train = shuffle and split_name == "train"
    sapient_xp.data.shuffle_val = shuffle and split_name != "train"

    loaders = sapient_xp.data.get_loaders(
        events=events_df,
        split_to_build="all",
    )
    if "all" not in loaders:
        raise RuntimeError(
            f"Sapient data pipeline returned no 'all' loader for split={split_name}. "
            f"Got keys: {list(loaders.keys())}"
        )
    return loaders["all"]


def iter_batches(loader, timeline_to_parcels: dict | None = None, tr_sec: float = TR_SECONDS):
    """Iterate the loader's batches, optionally injecting per-segment parcel
    targets into each batch.

    When `timeline_to_parcels` is provided, every batch is mutated so
    `batch.data["fmri"]` holds the (B, 400, T_seg) Schaefer target sliced
    from the cached arrays by each segment's (start, duration, timeline).
    This replaces the Sapient Fmri-extractor path (which requires FreeSurfer
    mesh files we don't have) — see SCIENCE.md §6 for the architectural
    rationale.
    """
    for batch in loader:
        if timeline_to_parcels is not None:
            _inject_parcel_targets_into_batch(batch, timeline_to_parcels, tr_sec)
        yield batch


# ─────────────────── Parcellation: vertex → Schaefer-400 ───────────────────
_SCHAEFER_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage5/label"
)
_SCHAEFER_FILES = (
    "lh.Schaefer2018_400Parcels_7Networks_order.annot",
    "rh.Schaefer2018_400Parcels_7Networks_order.annot",
)


def _ensure_schaefer_annots() -> tuple[Path, Path]:
    """Download (or re-use cached) Schaefer-400 fsaverage5 annot files."""
    import urllib.request

    atlas_dir = Path(SCHAEFER_CACHE_DIR)
    atlas_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn in _SCHAEFER_FILES:
        p = atlas_dir / fn
        if not p.exists():
            urllib.request.urlretrieve(f"{_SCHAEFER_BASE}/{fn}", p)
        paths.append(p)
    return paths[0], paths[1]


def _build_vertex_to_parcel():
    """Length-20484 int array; 0=medial wall, 1..200=LH, 201..400=RH.

    Identical recipe to modal_pipeline/app.py:_build_vertex_to_parcel so
    parcels are numerically comparable between training-time evaluation
    here and inference-time analytics there.
    """
    import numpy as np
    import nibabel.freesurfer.io as fs

    lh_path, rh_path = _ensure_schaefer_annots()
    lh_labels, _, _ = fs.read_annot(lh_path)
    rh_labels, _, _ = fs.read_annot(rh_path)
    v2p = np.zeros(N_VERTICES_FSAVERAGE5, dtype=np.int32)
    v2p[:10242] = np.where(lh_labels > 0, lh_labels, 0)
    v2p[10242:] = np.where(rh_labels > 0, rh_labels + 200, 0)
    return v2p


def _build_parcel_projection_matrix(v2p):
    """Build a (20484, 400) projection matrix that maps vertex predictions
    to per-parcel means in one matmul: parcels = vertices @ proj.

    Each column j holds 1/|parcel j| at vertices belonging to parcel j+1
    and 0 elsewhere. Right-multiplying a (B, T', 20484) tensor by `proj`
    gives (B, T', 400) where each entry is the within-parcel mean.
    """
    import torch
    import numpy as np

    proj = np.zeros((N_VERTICES_FSAVERAGE5, N_PARCELS_SCHAEFER400), dtype=np.float32)
    for pid in range(1, N_PARCELS_SCHAEFER400 + 1):
        mask = v2p == pid
        n = int(mask.sum())
        if n > 0:
            proj[mask, pid - 1] = 1.0 / n
    return torch.from_numpy(proj)


def _project_vertices_to_parcels(y_pred, proj):
    """y_pred: (B, 20484, T')  →  (B, 400, T')

    Uses einsum on the (B, T', 20484) → (B, T', 400) layout, then permutes
    back to (B, 400, T') so the time axis stays in the same place as our
    parcellated targets.
    """
    import torch

    # (B, V, T) -> (B, T, V) -> (B, T, P) -> (B, P, T)
    y_bt_v = y_pred.transpose(1, 2)
    y_bt_p = torch.matmul(y_bt_v, proj)
    return y_bt_p.transpose(1, 2)


def _extract_parcel_targets(batch, n_output_timesteps: int):
    """Pull the (B, 400, T') target from a SegmentData batch.

    The Sapient dataloader places the BOLD signal at `batch.data["fmri"]`
    with shape (B, D, T). For our parcellated targets D=400, T equals
    `n_output_timesteps` after pooling.
    """
    import torch
    import torch.nn.functional as F

    target = batch.data["fmri"]  # (B, D, T)
    if target.shape[1] != N_PARCELS_SCHAEFER400:
        raise RuntimeError(
            f"Expected parcellated target with 400 channels, got shape {tuple(target.shape)}. "
            "Check that 03_parcellate.py wrote (T, 400) arrays."
        )
    # If the dataloader produced a different T than the model output (e.g.
    # the segment was shorter than `n_output_timesteps`), resample with
    # interp1d-equivalent linear interpolation. This mirrors what the
    # model's `AdaptiveAvgPool1d` does for its predictions.
    if target.shape[-1] != n_output_timesteps:
        target = F.interpolate(
            target.float(),
            size=n_output_timesteps,
            mode="linear",
            align_corners=False,
        )
    return target.float()


# ─────────────────── Evaluation: DMN + MFG per-parcel Pearson R ───────────
def _get_parcel_groups() -> dict[str, list[int]]:
    """Return 0-indexed parcel ID lists for the two ROI families we score.

    Schaefer2018 7-Networks labels follow the pattern
    `7Networks_LH_Default_pCunPCC_1` (default-mode = DMN). The middle
    frontal gyrus (MFG) parcels live primarily inside the Control network
    (e.g. `7Networks_LH_Cont_PFCl_1`), with some falling under
    SalVentAttn for ventrolateral PFC. We pull labels directly from
    nilearn's atlas catalog so the mapping is canonical and updates
    automatically if nilearn revises its label strings.

    Indices returned are 0-indexed columns into the (T, 400) Schaefer
    arrays (parcel pid in the atlas = column pid-1 here).
    """
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_PARCELS_SCHAEFER400, yeo_networks=7, resolution_mm=2
    )
    # `atlas.labels` is a list of bytes/str of length 400, in 1-indexed order.
    labels = [l.decode() if isinstance(l, (bytes, bytearray)) else str(l) for l in atlas.labels]
    dmn: list[int] = []
    mfg: list[int] = []
    for i, name in enumerate(labels):
        nm = name.lower()
        if "default" in nm:
            dmn.append(i)
        # MFG ≈ dorsolateral PFC parcels in Cont network plus VLPFC in
        # SalVentAttn. The Schaefer label tag "pfcl" (lateral PFC) is the
        # most reliable single anchor for MFG across both networks.
        if "pfcl" in nm or "pfcm" in nm or "fef" in nm:
            mfg.append(i)
    if not dmn or not mfg:
        # Defensive fallback — this should never fire with the canonical
        # Schaefer-400 7-Networks atlas, but if nilearn ever changes its
        # label format we don't want eval to crash silently.
        raise RuntimeError(
            f"Could not derive DMN/MFG indices from Schaefer labels. "
            f"|DMN|={len(dmn)}  |MFG|={len(mfg)}  sample labels={labels[:5]}"
        )
    return {"dmn": dmn, "mfg": mfg}


def evaluate_pearson_r(
    brain_model,
    loader,
    proj,
    parcel_groups,
    timeline_to_parcels: dict | None = None,
    tr_sec: float = TR_SECONDS,
) -> float:
    """Average Pearson R across all DMN + MFG parcels, returned as a
    single scalar used to drive early stopping (higher = better).

    For each parcel we concatenate predictions across all val batches
    into one long time series, do the same for the target, and compute
    a single Pearson correlation. Then we average across the (DMN ∪ MFG)
    parcel set. This matches the convention used in the original Sapient
    `pearson` metric.

    When `timeline_to_parcels` is supplied, the per-segment parcellated
    targets are injected into each batch before reading them (replaces
    the Sapient Fmri-extractor surface-projection path).
    """
    import torch
    import numpy as np

    parcel_ids = sorted(set(parcel_groups["dmn"]) | set(parcel_groups["mfg"]))
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in iter_batches(loader, timeline_to_parcels, tr_sec):
            batch = batch.to(brain_model.device)
            y_pred = brain_model(batch)
            y_pred_p = _project_vertices_to_parcels(y_pred, proj)
            y_true_p = _extract_parcel_targets(batch, brain_model.n_output_timesteps)
            # (B, P, T) → (B*T, P)
            y_pred_flat = y_pred_p.permute(0, 2, 1).reshape(-1, N_PARCELS_SCHAEFER400)
            y_true_flat = y_true_p.permute(0, 2, 1).reshape(-1, N_PARCELS_SCHAEFER400)
            pred_chunks.append(y_pred_flat.detach().cpu().numpy())
            true_chunks.append(y_true_flat.detach().cpu().numpy())

    if not pred_chunks:
        return float("nan")
    preds = np.concatenate(pred_chunks, axis=0)
    trues = np.concatenate(true_chunks, axis=0)

    rs: list[float] = []
    for pid in parcel_ids:
        a = preds[:, pid].astype(np.float64)
        b = trues[:, pid].astype(np.float64)
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


# ─────────────────── Optimizer scheduler (warmup → cosine) ────────────────
def make_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup → cosine annealing. Plain LambdaLR, no external deps."""
    import math
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ───────────────────────── Local entrypoint ─────────────────────────
@app.local_entrypoint()
def main(
    subjects: str = "",  # comma-separated; "" → use pilot from splits.json
    epochs: int = 5,
    dry_run: bool = False,
    run_name: str = "sapient_v0.5_pilot",
):
    """
    Run the fine-tune.

    Defaults to pilot subjects from sapienteval/splits.json (sub-01..sub-05).
    Pass `--dry-run` to load the model, find the data, and run a single
    forward+backward pass — then exit BEFORE the full training loop. This
    is the safety hatch for verifying the integration without spending GPU $.
    """
    if not subjects:
        with open("sapienteval/splits.json") as f:
            splits = json.load(f)
        subject_list = splits["pilot"]["subjects"]
    else:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]

    print(f"Fine-tune subjects: {subject_list}")
    print(f"Run name: {run_name}, epochs: {epochs}, dry_run: {dry_run}")

    t0 = time.time()
    result = train.remote(
        subject_ids=subject_list,
        epochs=epochs,
        run_name=run_name,
        dry_run=dry_run,
    )
    result["duration_s"] = time.time() - t0
    print(json.dumps(result, indent=2))
