"""HuggingFace release assembly for sapient-1. Per spec §6.

Build a release_artifacts/sapient-1-llama/ folder containing everything HF
needs, then create the repo (private=True) and upload. The script DOES NOT
push unless `--confirm` is passed — defensive gate against accidental
publication.

  HF_ORG = "The-Sapient-Company"     # canonical slug (NOT "sapient")
  REPO_ID = "The-Sapient-Company/sapient-1-llama"

Usage:
  python release.py                              # assemble + dry-run summary
  python release.py --confirm                    # assemble + actually push
  python release.py --checkpoint <path.pt>       # explicit checkpoint
  python release.py --public                     # ONLY after legal sign-off
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HF_ORG = "The-Sapient-Company"
VARIANT = "sapient-1-llama"
REPO_ID = f"{HF_ORG}/{VARIANT}"
DEFAULT_ARTIFACT_DIR = Path("release_artifacts") / VARIANT


def _model_card() -> str:
    return f"""---
license: apache-2.0
library_name: pytorch
tags:
- brain-encoding
- fmri
- multimodal
- neuroscience
- foundation-model
datasets:
- courtois-neuromod/cneuromod
---

# Sapient-1 (Llama-3.2 variant)

A trimodal brain encoder predicting whole-cortex fMRI response to video,
audio, and text stimuli. Trained from scratch by The Sapient Company on
publicly-released, CC0-licensed neuroimaging data.

**Built with Llama.**

## Architecture

- 8-layer transformer, hidden=1152, 8 attention heads, dropout 0.1
- Frozen feature encoders (loaded separately, never retrained):
  - **Video:** V-JEPA 2 Gigantic (`facebook/vjepa2-vitg-fpc64-256`, MIT) — 1280-dim @ 2 Hz
  - **Audio:** Wav2Vec-BERT 2.0 (`facebook/w2v-bert-2.0`, MIT) — 1024-dim @ 2 Hz
  - **Text:**  Llama-3.2-3B (`meta-llama/Llama-3.2-3B`, Llama 3.2 Community) — 3072-dim @ 2 Hz
- Output: 20,484 fsaverage5 cortical-surface vertices @ 1 Hz
- ~178M trainable parameters

## Training Data

- **Courtois NeuroMod** open subset (sub-01, sub-02, sub-03, sub-05) — CC0
  via the Canadian Open Neuroscience Platform (CONP). 80 h × 4 subjects =
  320 h of paired stimulus-fMRI.
- *Optional week-2 augmentation*: BOLD Moments Dataset (Lahner et al. 2024,
  OpenNeuro ds005165) and Narratives (Nastase et al. 2021, OpenNeuro
  ds002345). Stimulus audio from Narratives is used through frozen encoders
  only and is **not redistributed** with this model.

## License

This model: **Apache 2.0** (see `LICENSE`).

The frozen Llama-3.2-3B text encoder is subject to the
[Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).
V-JEPA 2 and Wav2Vec-BERT 2.0 are MIT. See `NOTICE` for the full third-party
attribution.

## Built On

- TRIBE v1 (D'Ascoli et al. 2025, arXiv:2507.22229) — architectural
  inspiration only; this model contains **no** TRIBE-derived weights.
- V-JEPA 2 (Meta, MIT)
- Wav2Vec-BERT 2.0 (Meta, MIT)
- Llama-3.2-3B (Meta, Llama 3.2 Community License)

## Inference

```python
import torch
from huggingface_hub import hf_hub_download
import safetensors.torch as st
from sapient1 import SapientConfig, SapientModel, vertices_to_parcels, load_parcellation

# Load config + weights
cfg_path = hf_hub_download("{REPO_ID}", "config.json")
sd_path  = hf_hub_download("{REPO_ID}", "model.safetensors")
cfg = SapientConfig(**json.load(open(cfg_path)))
model = SapientModel(cfg).eval()
model.load_state_dict(st.load_file(sd_path))

# Forward pass on cached encoder features
# video (B, 200, 1280), audio (B, 200, 1024), text (B, 200, 3072), subject (B,)
with torch.no_grad():
    pred = model(video, audio, text, subject)   # (B, 100, 20484)

# Project to Schaefer-1000 parcels (for downstream sapienteval/ consumption)
parcellate = load_parcellation(hf_hub_download("{REPO_ID}", "parcellate.npz"))
parcels = vertices_to_parcels(pred, parcellate)  # (B, 100, 1000)
```

## Citation

See `CITATION.cff`.
"""


def _inference_example() -> str:
    return '''"""Minimal inference example for sapient-1-llama.

Loads the released checkpoint and runs a forward pass against pre-cached
encoder features (V-JEPA 2 video, W2V-BERT audio, Llama-3.2-3B text). The
model never re-runs the encoders — features are computed once upstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import safetensors.torch as st
import torch
from huggingface_hub import hf_hub_download

from sapient1 import (
    SapientConfig, SapientModel,
    vertices_to_parcels, load_parcellation,
)

REPO_ID = "The-Sapient-Company/sapient-1-llama"


def load_model() -> SapientModel:
    cfg_path = hf_hub_download(REPO_ID, "config.json")
    sd_path  = hf_hub_download(REPO_ID, "model.safetensors")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = SapientConfig(**cfg_dict)
    model = SapientModel(cfg).eval()
    model.load_state_dict(st.load_file(sd_path))
    return model


if __name__ == "__main__":
    model = load_model()
    # Dummy features so the example is runnable end-to-end.
    B = 1
    video = torch.zeros(B, 200, 1280)
    audio = torch.zeros(B, 200, 1024)
    text  = torch.zeros(B, 200, 3072)
    subj  = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        pred = model(video, audio, text, subj)
    print(f"vertex prediction: {tuple(pred.shape)}")  # (B, 100, 20484)

    parcellate = load_parcellation(hf_hub_download(REPO_ID, "parcellate.npz"))
    parcels = vertices_to_parcels(pred, parcellate)
    print(f"parcel prediction: {parcels.shape}")  # (B, 100, 1000)
'''


def _config_json_from_checkpoint(ckpt: dict) -> dict:
    """Extract just the model dimensions to ship as config.json (HF convention).

    We dump the SapientConfig fields, not the entire training YAML.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from sapient1 import SapientConfig
    cfg_yaml = ckpt.get("config", {})
    mcfg = SapientConfig.from_yaml(cfg_yaml)
    return {
        "model_type": "sapient",
        "variant": VARIANT,
        "architectures": ["SapientModel"],
        **{k: getattr(mcfg, k) for k in (
            "d_video", "d_audio", "d_text",
            "hidden", "n_layers", "n_heads", "ff_mult", "dropout",
            "low_rank_dim", "n_vertices",
            "modality_dropout", "subject_dropout",
            "max_stim_len", "max_fmri_len", "n_subjects",
        )},
    }


def assemble(checkpoint_path: Path, artifact_dir: Path,
             parcellation_path: Path) -> dict:
    """Build the artifact directory. Returns the manifest of files included."""
    import torch
    import safetensors.torch as st

    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Assembling → {artifact_dir}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # 1. config.json
    cfg = _config_json_from_checkpoint(ckpt)
    (artifact_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # 2. model.safetensors
    st.save_file(state_dict, str(artifact_dir / "model.safetensors"))

    # 3. README (model card)
    (artifact_dir / "README.md").write_text(_model_card())

    # 4. LICENSE + NOTICE (copy from repo root)
    for fname in ("LICENSE", "NOTICE"):
        src = Path(__file__).parent / fname
        if src.exists():
            shutil.copy(src, artifact_dir / fname)

    # 5. parcellate.npz (vertex → Schaefer-1000 averaging matrix)
    if parcellation_path.exists():
        shutil.copy(parcellation_path, artifact_dir / "parcellate.npz")
    else:
        print(f"  ⚠️ {parcellation_path} missing — release without parcellation matrix")

    # 6. inference.py (minimal example)
    (artifact_dir / "inference.py").write_text(_inference_example())

    files = sorted(p.name for p in artifact_dir.iterdir())
    print("  files:")
    for f in files:
        size = (artifact_dir / f).stat().st_size
        print(f"    {f:<28} {size:>14,} bytes")
    return {"artifact_dir": str(artifact_dir), "files": files}


def push(artifact_dir: Path, repo_id: str, public: bool = False) -> None:
    """Create the HF repo (idempotent) and upload everything in artifact_dir."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type="model", private=not public, exist_ok=True)
    print(f"Uploading {artifact_dir} → {repo_id}  (private={not public})")
    api.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial release: sapient-1-llama v0.1",
    )
    print(f"✅ Uploaded.  https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/sapient1_llama/latest.pt"))
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument("--parcellation", type=Path, default=Path("data/parcellate.npz"))
    ap.add_argument("--confirm", action="store_true",
                    help="Actually push to HF (default: dry-run / local assembly only)")
    ap.add_argument("--public", action="store_true",
                    help="Upload as PUBLIC repo. Default is private. Only after legal sign-off.")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ABORT: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(2)

    manifest = assemble(args.checkpoint, args.artifact_dir, args.parcellation)
    print(f"\nDry run complete. Artifact dir: {manifest['artifact_dir']}")
    if not args.confirm:
        print("\nNot pushing. Re-run with --confirm to upload to "
              f"{REPO_ID} (private={not args.public}).")
        return

    if args.public:
        answer = input(f"\nAbout to push PUBLIC to {REPO_ID}. Type the repo "
                       f"name to confirm: ").strip()
        if answer != REPO_ID:
            print("ABORT: confirmation mismatch.", file=sys.stderr)
            sys.exit(2)

    push(args.artifact_dir, REPO_ID, public=args.public)


if __name__ == "__main__":
    main()
