"""
sapienteval-inference — Modal endpoint that serves Sapient-v1 cognitive scores.

Loads the fine-tuned brain-encoding checkpoint + the (400, 8) mapping head
once at container warmup, then exposes a single HTTP POST endpoint:

    POST /score
    body: { "prompt": str, "response": str }
    returns: {
      "scores": {
        "attention": float, "comprehension": float, "engagement": float,
        "frustration": float, "surprise": float, "theory_of_mind": float,
        "cognitive_load": float, "predicted_dropout": float
      },
      "model_version": "sapient_v0.5_pilot" | "sapient_v1" | "literature_prior",
      "checkpoint_sha": str,
      "mapping_source": "learned" | "literature_prior",
      "raw_bold_400": list[float] | None,
      "duration_ms": int
    }

Called from the TS side at `api/sapienteval/score.ts` when SAPIENT_SCORER=v1.
If the .pt or mapping head is missing, falls back to the literature-prior
mapping so the endpoint always returns something callable.

Deploy:    modal deploy sapienteval/modal/inference_sapient.py
Test:      modal run sapienteval/modal/inference_sapient.py::smoke_test
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

# ───────────────────────── Image, volumes, secrets ─────────────────────────

LOCAL_SAPIENT_REPO = "/Users/robertgutierrez/Desktop/SIMPLR/sapient-model"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "ca-certificates", "build-essential", "ffmpeg", "libgl1")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
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
        "nibabel",
        "nilearn==0.10.4",
        "scipy",
        "pandas",
        "pyarrow",
        "tqdm",
        "exca",
        "pydantic",
        "requests",
        "fastapi[standard]",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_dir(
        LOCAL_SAPIENT_REPO,
        remote_path="/root/sapient_repo",
        copy=True,
        ignore=[
            "__pycache__", ".git", ".venv", ".cursor", ".claude",
            ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
            ".modal_logs", "cache", "outputs", "inputs", "wandb",
            "lightning_logs", "*.ipynb", "*.egg-info", ".DS_Store",
        ],
    )
    .run_commands("pip install --no-deps -e /root/sapient_repo")
    .env({
        "HF_HOME": "/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
        "TORCH_HOME": "/cache/torch",
        "NILEARN_SHARED_DATA": "/cache/nilearn_data",
        "SAPIENT_CACHE_FOLDER": "/cache/sapient_features",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

base_vol = modal.Volume.from_name("sapient-base-encoder-weights")
ckpt_vol = modal.Volume.from_name("sapienteval-checkpoints", create_if_missing=True)
cache_vol = modal.Volume.from_name("sapient-cache", create_if_missing=True)


def make_secrets() -> list[modal.Secret]:
    secrets: list[modal.Secret] = []
    try:
        secrets.append(modal.Secret.from_name("hf-token"))
    except Exception:
        pass
    return secrets


app = modal.App("sapienteval-inference", image=image)

# Same canonical constants as the training script
COGNITIVE_DIMS = (
    "attention", "comprehension", "engagement", "frustration",
    "surprise", "theory_of_mind", "cognitive_load", "predicted_dropout",
)
N_PARCELS = 400
N_VERTICES_FSAVERAGE5 = 20484
TR_SECONDS = 2.0

CHECKPOINT_CANDIDATES = (
    "/ckpt/sapient_v1.pt",            # scale-up checkpoint (preferred)
    "/ckpt/sapient_v0.5_pilot.pt",    # pilot fallback
)
MAPPING_HEAD_PATH = "/ckpt/mapping_head.npz"
BASE_CHECKPOINT_DIR = "/base"
BASE_CHECKPOINT_NAME = "best.ckpt"
FEATURE_CACHE_DIR = "/cache/sapient_features"


# ───────────────────────── Literature-prior mapping head ──────────────────
# Hand-curated (400, 8) weights so the endpoint can return scores even
# before the learned mapping head exists. Each entry maps a parcel-naming
# substring -> {dim: weight} contribution. Applied at warmup.
#
# Citations live in SCIENCE.md §7. Mostly from the meta-analyses we
# already cite in scorers.ts (Saxe & Kanwisher 2003 TPJ for ToM, Owen
# 2005 / Niendam 2012 DLPFC for cognitive load, Botvinick 2004 ACC
# for frustration/surprise, Hasson 2008 / Simony 2016 DMN posterior
# for engagement, etc.).
LITERATURE_PRIOR_RULES = [
    # (substring_to_match_in_parcel_label, {dim: weight})
    ("default",      {"engagement": +0.6, "theory_of_mind": +0.3}),         # DMN
    ("pcun",         {"engagement": +0.4, "theory_of_mind": +0.3}),         # precuneus
    ("pfcl",         {"cognitive_load": +0.6, "attention": +0.3}),           # DLPFC
    ("pfcm",         {"cognitive_load": +0.5}),                              # medial PFC
    ("fef",          {"attention": +0.5}),                                   # frontal eye fields
    ("ips",          {"attention": +0.5}),                                   # intraparietal sulcus
    ("temppar",      {"theory_of_mind": +0.6}),                              # TPJ
    ("tempp",        {"theory_of_mind": +0.4, "comprehension": +0.3}),       # temporal pole / parietal
    ("salventattn",  {"frustration": +0.3, "surprise": +0.4}),               # salience
    ("ains",         {"frustration": +0.4}),                                 # anterior insula
    ("acc",          {"frustration": +0.4, "surprise": +0.3}),               # ACC
    ("post",         {"engagement": +0.2}),                                  # posterior catch-all
    ("ifg",          {"comprehension": +0.5}),                               # IFG (Broca)
    ("stg",          {"comprehension": +0.5}),                               # STG (Wernicke)
    ("v1",           {"surprise": +0.2}),                                    # primary visual (low-level surprise)
    ("somatomotor",  {"attention": +0.1}),                                   # SM cortex (low weight)
]


def _build_literature_prior_mapping():
    """Build a (400, 8) numpy matrix from LITERATURE_PRIOR_RULES."""
    import numpy as np
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=N_PARCELS, yeo_networks=7, resolution_mm=2,
    )
    labels = [
        (l.decode() if isinstance(l, (bytes, bytearray)) else str(l)).lower()
        for l in atlas.labels
    ]
    n_dims = len(COGNITIVE_DIMS)
    W = np.zeros((N_PARCELS, n_dims), dtype="float32")
    for i, label in enumerate(labels):
        for substr, dim_weights in LITERATURE_PRIOR_RULES:
            if substr in label:
                for dim, w in dim_weights.items():
                    j = COGNITIVE_DIMS.index(dim)
                    W[i, j] += w
    # predicted_dropout is a derived dimension: high when load+frustration high,
    # low when engagement high. We set its weights as a linear combo on the
    # parcel side (rather than at inference time) so the API contract stays
    # consistent regardless of which mapping head ships.
    pd_idx = COGNITIVE_DIMS.index("predicted_dropout")
    cl_idx = COGNITIVE_DIMS.index("cognitive_load")
    fr_idx = COGNITIVE_DIMS.index("frustration")
    en_idx = COGNITIVE_DIMS.index("engagement")
    W[:, pd_idx] = 0.4 * W[:, cl_idx] + 0.4 * W[:, fr_idx] - 0.5 * W[:, en_idx]
    # Bias so each dim averages to ~0.5 on random input (sigmoid baseline)
    b = np.zeros(n_dims, dtype="float32")
    return W, b


# ───────────────────────── Inference class ────────────────────────────────

@app.cls(
    gpu="A100-40GB",
    volumes={"/base": base_vol, "/ckpt": ckpt_vol, "/cache": cache_vol},
    secrets=make_secrets(),
    timeout=10 * 60,
    min_containers=0,                # warm-up to 1 when SAPIENT_SCORER=v1 traffic ramps
    max_containers=2,
    scaledown_window=300,
)
class SapientV1Inference:
    """Holds the loaded model + mapping head in a warm container."""

    @modal.enter()
    def setup(self):
        import os, sys, shutil
        import torch
        import numpy as np

        sys.path.insert(0, "/root/sapient_repo")

        hf_token = (
            os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

        # --- 1. Build SapientModel from the base checkpoint ---
        # Apply the Tribe→Sapient config patch (same as in finetune script).
        work_dir = Path("/tmp/sapient_base_patched")
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        src_dir = Path(BASE_CHECKPOINT_DIR)
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

        Path(FEATURE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

        from sapient import SapientModel
        self.sapient_xp = SapientModel.from_pretrained(
            str(work_dir),
            checkpoint_name=BASE_CHECKPOINT_NAME,
            cache_folder=FEATURE_CACHE_DIR,
            device="auto",
        )
        self.brain_model = self.sapient_xp._model

        # --- 2. Overlay the fine-tuned checkpoint if available ---
        self.model_version = "literature_prior"
        self.checkpoint_sha = "base-only"
        for ck in CHECKPOINT_CANDIDATES:
            if Path(ck).exists():
                state = torch.load(ck, map_location="cuda", weights_only=False)
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                # Strip Lightning prefix if present
                clean_sd = {
                    (k[len("model."):] if k.startswith("model.") else k): v
                    for k, v in sd.items()
                }
                missing, unexpected = self.brain_model.load_state_dict(
                    clean_sd, strict=False
                )
                print(f"Loaded fine-tuned ckpt {ck}: "
                      f"missing={len(missing)} unexpected={len(unexpected)}")
                self.model_version = (
                    "sapient_v1"
                    if "v1" in Path(ck).name
                    else "sapient_v0.5_pilot"
                )
                import hashlib
                self.checkpoint_sha = hashlib.sha256(
                    Path(ck).read_bytes()
                ).hexdigest()[:16]
                break
        self.brain_model = self.brain_model.cuda().eval()

        # --- 3. Build vertex→parcel projection ---
        from urllib.request import urlretrieve
        atlas_dir = Path("/cache/schaefer400_fsaverage5")
        atlas_dir.mkdir(parents=True, exist_ok=True)
        base_url = (
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
            "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
            "Parcellations/FreeSurfer5.3/fsaverage5/label"
        )
        for fn in ("lh.Schaefer2018_400Parcels_7Networks_order.annot",
                   "rh.Schaefer2018_400Parcels_7Networks_order.annot"):
            p = atlas_dir / fn
            if not p.exists():
                urlretrieve(f"{base_url}/{fn}", p)

        import nibabel.freesurfer.io as fs
        lh_labels, _, _ = fs.read_annot(atlas_dir / "lh.Schaefer2018_400Parcels_7Networks_order.annot")
        rh_labels, _, _ = fs.read_annot(atlas_dir / "rh.Schaefer2018_400Parcels_7Networks_order.annot")
        v2p = np.zeros(N_VERTICES_FSAVERAGE5, dtype=np.int32)
        v2p[:10242] = np.where(lh_labels > 0, lh_labels, 0)
        v2p[10242:] = np.where(rh_labels > 0, rh_labels + 200, 0)
        proj = np.zeros((N_VERTICES_FSAVERAGE5, N_PARCELS), dtype="float32")
        for p_id in range(1, N_PARCELS + 1):
            members = v2p == p_id
            if members.any():
                proj[members, p_id - 1] = 1.0 / members.sum()
        self.proj = torch.from_numpy(proj).cuda()  # (20484, 400)

        # --- 4. Load mapping head (learned if present, else literature prior) ---
        if Path(MAPPING_HEAD_PATH).exists():
            head_data = np.load(MAPPING_HEAD_PATH)
            self.W = torch.from_numpy(head_data["W"]).cuda().float()  # (400, 8)
            self.b = torch.from_numpy(head_data["b"]).cuda().float()  # (8,)
            self.mapping_source = "learned"
            print(f"Loaded learned mapping head from {MAPPING_HEAD_PATH}")
        else:
            W_np, b_np = _build_literature_prior_mapping()
            self.W = torch.from_numpy(W_np).cuda().float()
            self.b = torch.from_numpy(b_np).cuda().float()
            self.mapping_source = "literature_prior"
            print("Using literature-prior mapping head (no learned head found)")

        print(f"Inference container ready. model_version={self.model_version} "
              f"mapping={self.mapping_source}")

    @modal.method()
    def score(self, prompt: str, response: str, return_raw_bold: bool = False) -> dict:
        """Score a (prompt, response) pair on the 8 cognitive dimensions."""
        import torch
        import numpy as np
        import pandas as pd
        from neuralset.dataloader import SegmentData

        t0 = time.time()

        # Build a minimal Word-events DataFrame for this single (prompt, response)
        # pair. We concatenate prompt + response with a separator, assign fake
        # sequential timestamps at 0.5s/word so the segment is long enough.
        text_full = f"{prompt.strip()} [SEP] {response.strip()}"
        words = text_full.split()
        if not words:
            words = ["[EMPTY]"]
        words_per_second = 2.0
        rows = []
        for i, w in enumerate(words):
            t_start = i / words_per_second
            rows.append({
                "type": "Word",
                "timeline": "inference_segment",
                "subject": "inference",
                "split": "inference",
                "start": t_start,
                "duration": 1.0 / words_per_second,
                "text": w,
                "sentence": text_full,
                "context": text_full,
                "language": "english",
            })
        events_df = pd.DataFrame(rows)

        self.sapient_xp.data.batch_size = 1
        loaders = self.sapient_xp.data.get_loaders(
            events=events_df, split_to_build="all"
        )
        loader = loaders["all"]

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.brain_model.device)
                y_pred = self.brain_model(batch)  # (1, 20484, T')
                y_pred_parcels = (
                    y_pred.permute(0, 2, 1).contiguous() @ self.proj
                ).permute(0, 2, 1)  # (1, 400, T')

                # Time-pool predictions to one (400,) vector per request.
                # Average across time is conservative; for inference we don't
                # have a fine-grained TR target so a single vector is fine.
                bold_400 = y_pred_parcels.mean(dim=-1).squeeze(0)  # (400,)

                # Apply mapping head: (400,) @ (400, 8) + (8,) -> sigmoid
                raw = bold_400 @ self.W + self.b
                scores = torch.sigmoid(raw).cpu().numpy()
                break

        out_scores = {dim: float(scores[i]) for i, dim in enumerate(COGNITIVE_DIMS)}
        result = {
            "scores": out_scores,
            "model_version": self.model_version,
            "checkpoint_sha": self.checkpoint_sha,
            "mapping_source": self.mapping_source,
            "duration_ms": int((time.time() - t0) * 1000),
            "raw_bold_400": (
                bold_400.cpu().numpy().tolist()
                if return_raw_bold else None
            ),
        }
        return result

    @modal.fastapi_endpoint(method="POST")
    def score_endpoint(self, body: dict) -> dict:
        """HTTP endpoint mirroring the .score() method."""
        prompt = body.get("prompt", "")
        response = body.get("response", "")
        return_raw = bool(body.get("return_raw_bold", False))
        if not isinstance(prompt, str) or not isinstance(response, str):
            return {"error": "prompt and response must be strings"}
        return self.score.local(prompt, response, return_raw_bold=return_raw)


@app.local_entrypoint()
def smoke_test(prompt: str = "Is my plan to quit my job and day-trade crypto a good idea?",
               response: str = "Great question! That sounds exciting — here are some general considerations…"):
    """Local smoke test: instantiates the inference class, scores one pair."""
    runner = SapientV1Inference()
    result = runner.score.remote(prompt=prompt, response=response, return_raw_bold=False)
    print(json.dumps(result, indent=2))
