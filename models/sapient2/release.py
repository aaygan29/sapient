"""HuggingFace release assembly for sapient-2. Per spec §6.

Two release variants share this file. Pass --variant to pick which to assemble:
  • scratch → release_artifacts/sapient-2-scratch-llama/ → The-Sapient-Company/sapient-2-scratch-llama
  • ft      → release_artifacts/sapient-2-ft-llama/      → The-Sapient-Company/sapient-2-ft-llama

DOES NOT push unless `--confirm` is passed. Defensive gate against accidental
publication.

  HF_ORG = "The-Sapient-Company"     # canonical slug (NOT "sapient")

Usage:
  python release.py --variant scratch                 # assemble + dry-run
  python release.py --variant ft --confirm            # assemble + push (private)
  python release.py --variant ft --confirm --public   # ONLY after legal sign-off
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HF_ORG = "The-Sapient-Company"

VARIANT_REPO = {
    "scratch": "sapient-2-scratch-llama",
    "ft":      "sapient-2-ft-llama",
}


def _model_card(variant: str) -> str:
    repo = f"{HF_ORG}/{VARIANT_REPO[variant]}"
    is_ft = (variant == "ft")
    lineage = (
        "\n## Lineage\n\nThis variant is **fine-tuned from "
        f"[{HF_ORG}/sapient-1-llama](https://huggingface.co/{HF_ORG}/sapient-1-llama)** "
        "at `lr=5e-5`. The `subject_embed` table is dropped at load because the "
        "subject set differs between sapient-1 (4 CNeuroMod subjects) and "
        "sapient-2 (~75 pooled ds004996 + ds001740 subjects). Decisions #11 and "
        "#12 in the architecture lock.\n"
    ) if is_ft else (
        "\n## Lineage\n\nThis variant is trained **from scratch** (random init) "
        "on ds004996 + ds001740. No sapient-1 weight lineage. The sibling "
        f"[{HF_ORG}/sapient-2-ft-llama](https://huggingface.co/{HF_ORG}/sapient-2-ft-llama) "
        "variant warm-starts from sapient-1; train both, ship the winner.\n"
    )
    return f"""---
license: apache-2.0
library_name: pytorch
tags:
- brain-encoding
- fmri
- multimodal
- human-robot-interaction
- conversation
- neuroscience
- foundation-model
datasets:
- openneuro/ds004996
- openneuro/ds001740
---

# Sapient-2 ({"fine-tuned, Llama-3.2 variant" if is_ft else "from-scratch, Llama-3.2 variant"})

A conversation-specialized brain encoder. Same architecture as
[`{HF_ORG}/sapient-1-llama`](https://huggingface.co/{HF_ORG}/sapient-1-llama),
retrained on the only two public fMRI corpora of human-robot interaction.

**Built with Llama.**
{lineage}
## Architecture

- 8-layer transformer, hidden=1152, 8 attention heads, dropout 0.1
- Frozen feature encoders (loaded separately, never retrained):
  - **Video:** V-JEPA 2 Gigantic (`facebook/vjepa2-vitg-fpc64-256`, MIT)
  - **Audio:** Wav2Vec-BERT 2.0 (`facebook/w2v-bert-2.0`, MIT)
  - **Text:**  Llama-3.2-3B (`meta-llama/Llama-3.2-3B`, Llama 3.2 Community)
- Output: 20,484 fsaverage5 cortical-surface vertices @ 1 Hz
- ~178M trainable parameters

## Training Data

- **ds004996 NeuroEngage** (Toruparova et al. 2025; ~50 subjects of
  human-human and human-robot dialogue fMRI). CC0 via OpenNeuro.
- **ds001740 Rauchbauer 2019/2020** (25 French speakers; human-human and
  human-robot conversation with the Furhat robot, Wizard-of-Oz). Pinned to
  v2.1.0. CC0 via OpenNeuro.

Pooled training corpus: ~26.5 hours, ~75 subjects. Subject embeddings
separate the populations; this forces sapient-2 to learn robot-agnostic
conversation representations.

## License

This model: **Apache 2.0** (see `LICENSE`).

The frozen Llama-3.2-3B text encoder is subject to the
[Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). V-JEPA 2
and Wav2Vec-BERT 2.0 are MIT. See `NOTICE` for the full third-party attribution.

## Inference

```python
import json, torch
import safetensors.torch as st
from huggingface_hub import hf_hub_download
from sapient2 import SapientConfig, SapientModel, vertices_to_parcels, load_parcellation

cfg_path = hf_hub_download("{repo}", "config.json")
sd_path  = hf_hub_download("{repo}", "model.safetensors")
cfg = SapientConfig(**json.load(open(cfg_path)))
model = SapientModel(cfg).eval()
model.load_state_dict(st.load_file(sd_path))

with torch.no_grad():
    pred = model(video, audio, text, subject)   # (B, 100, 20484)

parcellate = load_parcellation(hf_hub_download("{repo}", "parcellate.npz"))
parcels = vertices_to_parcels(pred, parcellate) # (B, 100, 1000)
```

## Citation

See `CITATION.cff`.
"""


def _inference_example(variant: str) -> str:
    repo = f"{HF_ORG}/{VARIANT_REPO[variant]}"
    return f'''"""Minimal inference example for {VARIANT_REPO[variant]}."""

from __future__ import annotations

import json
import safetensors.torch as st
import torch
from huggingface_hub import hf_hub_download

from sapient2 import (
    SapientConfig, SapientModel,
    vertices_to_parcels, load_parcellation,
)

REPO_ID = "{repo}"


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
    B = 1
    video = torch.zeros(B, 200, 1280)
    audio = torch.zeros(B, 200, 1024)
    text  = torch.zeros(B, 200, 3072)
    subj  = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        pred = model(video, audio, text, subj)
    print(f"vertex prediction: {{tuple(pred.shape)}}")

    parcellate = load_parcellation(hf_hub_download(REPO_ID, "parcellate.npz"))
    parcels = vertices_to_parcels(pred, parcellate)
    print(f"parcel prediction: {{parcels.shape}}")
'''


def _config_json_from_checkpoint(ckpt: dict, variant: str) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from sapient2 import SapientConfig
    cfg_yaml = ckpt.get("config", {})
    mcfg = SapientConfig.from_yaml(cfg_yaml)
    return {
        "model_type": "sapient",
        "variant": VARIANT_REPO[variant],
        "architectures": ["SapientModel"],
        **{k: getattr(mcfg, k) for k in (
            "d_video", "d_audio", "d_text",
            "hidden", "n_layers", "n_heads", "ff_mult", "dropout",
            "low_rank_dim", "n_vertices",
            "modality_dropout", "subject_dropout",
            "max_stim_len", "max_fmri_len", "n_subjects",
        )},
    }


def assemble(variant: str, checkpoint_path: Path, artifact_dir: Path,
             parcellation_path: Path) -> dict:
    import torch
    import safetensors.torch as st

    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Assembling {VARIANT_REPO[variant]} → {artifact_dir}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    cfg = _config_json_from_checkpoint(ckpt, variant)
    (artifact_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    st.save_file(state_dict, str(artifact_dir / "model.safetensors"))

    (artifact_dir / "README.md").write_text(_model_card(variant))

    for fname in ("LICENSE", "NOTICE"):
        src = Path(__file__).parent / fname
        if src.exists():
            shutil.copy(src, artifact_dir / fname)

    if parcellation_path.exists():
        shutil.copy(parcellation_path, artifact_dir / "parcellate.npz")
    else:
        print(f"  ⚠️ {parcellation_path} missing")

    (artifact_dir / "inference.py").write_text(_inference_example(variant))

    files = sorted(p.name for p in artifact_dir.iterdir())
    print("  files:")
    for f in files:
        size = (artifact_dir / f).stat().st_size
        print(f"    {f:<28} {size:>14,} bytes")
    return {"artifact_dir": str(artifact_dir), "files": files}


def push(artifact_dir: Path, repo_id: str, public: bool = False) -> None:
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(repo_id, repo_type="model", private=not public, exist_ok=True)
    print(f"Uploading {artifact_dir} → {repo_id}  (private={not public})")
    api.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Initial release: {repo_id.split('/')[-1]} v0.1",
    )
    print(f"✅ Uploaded.  https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("scratch", "ft"), required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--artifact-dir", type=Path, default=None)
    ap.add_argument("--parcellation", type=Path, default=Path("data/parcellate.npz"))
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    repo_name = VARIANT_REPO[args.variant]
    repo_id = f"{HF_ORG}/{repo_name}"
    artifact_dir = args.artifact_dir or (Path("release_artifacts") / repo_name)
    checkpoint = args.checkpoint or Path(
        f"checkpoints/sapient2_{args.variant}_llama/latest.pt"
    )

    if not checkpoint.exists():
        print(f"ABORT: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(2)

    manifest = assemble(args.variant, checkpoint, artifact_dir, args.parcellation)
    print(f"\nDry run complete. Artifact dir: {manifest['artifact_dir']}")
    if not args.confirm:
        print(f"\nNot pushing. Re-run with --confirm to upload to "
              f"{repo_id} (private={not args.public}).")
        return

    if args.public:
        answer = input(f"\nAbout to push PUBLIC to {repo_id}. Type the repo "
                       f"name to confirm: ").strip()
        if answer != repo_id:
            print("ABORT: confirmation mismatch.", file=sys.stderr)
            sys.exit(2)

    push(artifact_dir, repo_id, public=args.public)


if __name__ == "__main__":
    main()
