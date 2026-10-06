"""mary-serve — live Mary brain-encoder serving for the Mary chat product.

A warm GPU container loads Mary (the `improving` checkpoint) + the active feature
extractors ONCE, exposes a `submit` web endpoint that spawns a GPU job, and the
job writes status/result back into the `mary_runs` Supabase table as it runs.

Phase-1 slice: capability=brain_map, modality=text (qwen_ctx only). Audio
(whisper+beats+forced-align) is structured to bolt onto the same class.

Run/verify:
  modal run mary/modal/serve.py::prepare_yeo7         # one-time: cache the Yeo-7 map
  modal run mary/modal/serve.py::smoke                # end-to-end text smoke test
  modal deploy mary/modal/serve.py                    # publish the submit endpoint
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "mary-serve"

# serve.py lives at sapient-models/mary/modal/serve.py LOCALLY (used only for
# add_local_dir at image-build time). Inside the Modal container this module is
# mounted as /root/serve.py, where .parents[2] would IndexError — so guard it;
# the in-container values are unused (the dirs are already at /root/maryrepo,
# /root/qualia).
_HERE = Path(__file__).resolve()
_MARY_REPO = _HERE.parents[1] if len(_HERE.parents) >= 2 else Path("/root/maryrepo")
_QUALIA = (_HERE.parents[2] / "qualia") if len(_HERE.parents) >= 3 else Path("/root/qualia")

# Canonical Yeo-7 names — MUST match the TS Yeo7 union in src/lib/maryBrain.ts.
SCHAEFER_TOKEN_TO_YEO7 = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal",
    "Default": "Default Mode",
}
YEO7_ORDER = list(SCHAEFER_TOKEN_TO_YEO7.values())
YEO7_PLAIN = {
    "Visual": "Seeing & imagery",
    "Somatomotor": "Body & movement",
    "Dorsal Attention": "Focused attention",
    "Ventral Attention": "Salience & surprise",
    "Limbic": "Emotion & reward",
    "Frontoparietal": "Reasoning & effort",
    "Default Mode": "Reflection & self",
}

# ── 7 Yeo networks → the live 10 KPIs the verdict-v3 UI consumes ──────────────
# This is a TRANSLATION of Mary's 7-network prediction into the product's 10-KPI
# vocabulary — NOT 10 independent measurements. Each KPI is a fixed, documented
# weighted blend of the per-second network timecourse. Tuned so each KPI is led
# by the network(s) most plausibly behind that marketing signal. The KPI NAMES
# MUST match `KpiName` in src/lib/kpiExplainers.ts (canonical long forms).
KPI_NAMES = [
    "Visual Attention",
    "Auditory Engagement",
    "Face Processing",
    "Reading Engagement",
    "Language Comprehension",
    "Cognitive Effort",
    "Reward Valuation (cortical proxy)",
    "Emotional Salience",
    "Narrative Absorption",
    "Surprise / Novelty",
]
# weights keyed by Yeo-7 network name; missing networks weight 0. Each KPI is a
# weighted average (weights sum-normalized) of the per-second network values.
KPI_FROM_NETWORKS = {
    "Visual Attention":                   {"Visual": 0.65, "Dorsal Attention": 0.35},
    "Auditory Engagement":                {"Somatomotor": 1.0},
    "Face Processing":                    {"Visual": 1.0},
    "Reading Engagement":                 {"Visual": 0.6, "Default Mode": 0.4},
    "Language Comprehension":             {"Default Mode": 0.5, "Frontoparietal": 0.5},
    "Cognitive Effort":                   {"Frontoparietal": 1.0},
    "Reward Valuation (cortical proxy)":  {"Limbic": 1.0},
    "Emotional Salience":                 {"Limbic": 0.6, "Ventral Attention": 0.4},
    "Narrative Absorption":               {"Default Mode": 1.0},
    "Surprise / Novelty":                 {"Ventral Attention": 1.0},
}

# 5 product composites, each a blend of the 10 KPIs above. tier banding mirrors
# the verdict-v3 convention: >=70 strength, 40..69 okay, <40 weakness.
COMPOSITE_FROM_KPIS = {
    "Voice Impact":   {"Auditory Engagement": 0.6, "Language Comprehension": 0.4},
    "Visual Pull":    {"Visual Attention": 0.6, "Face Processing": 0.25, "Reading Engagement": 0.15},
    "Cognitive Grip": {"Cognitive Effort": 0.5, "Language Comprehension": 0.3, "Surprise / Novelty": 0.2},
    "Emotional Hit":  {"Emotional Salience": 0.6, "Reward Valuation (cortical proxy)": 0.4},
    "Memorability":   {"Narrative Absorption": 0.6, "Reward Valuation (cortical proxy)": 0.2, "Surprise / Novelty": 0.2},
}

# ── Corpus baseline for RELATIVE scoring ──────────────────────────────────────
# The raw per-KPI mean activation is nearly constant across ALL content, so an
# absolute mean→score map collapses every clip to ~51 and an A/B never shows a
# winner. Instead we score each KPI as a z-score against the corpus baseline
# (mean/std): a clip that activates a network MORE than the typical clip scores
# >50, less scores <50.
# RECALIBRATED 2026-06-07 for the mary_multi_stable_resp_s13 checkpoint (the new
# served model) THROUGH the responsive un-flatten. The un-flatten (each Yeo-7 reduction
# restricted to ISC>0.05 vertices, ISC-weighted) both raises magnitudes (old ~0.075 →
# ~0.145 here) and — critically — restores real cross-content spread (std ~0.0002, which
# saturated every z, → ~0.012), so scores differentiate instead of slamming the clamp.
# REEL-CENTERED: computed from 11 REAL short-form reels (the actual content scored) run
# through the deployed video path (build_reel_baseline entrypoint, probe_video → the same
# audio+slowfast+un-flatten compute production uses), NOT the movie/audiobook val corpus
# (which lands ~0.097 and pushed every reel to +3σ ≈ 85). With this reel baseline the 11
# reels recenter on ~50 with healthy spread (raw 0.142–0.155). Raw KPI means are
# baseline-independent given the fixed reduction; expand the reel corpus to re-tighten as
# more live runs accrue. (mean, std) per KPI — keys MUST match KPI_NAMES exactly. MUST
# stay in sync with src/mary/MaryPillars.tsx.
KPI_BASELINE = {
    "Visual Attention":                   (0.14189, 0.01159),
    "Auditory Engagement":                (0.14424, 0.01167),
    "Face Processing":                    (0.14147, 0.01159),
    "Reading Engagement":                 (0.14436, 0.01180),
    "Language Comprehension":             (0.15075, 0.01241),
    "Cognitive Effort":                   (0.15280, 0.01272),
    "Reward Valuation (cortical proxy)":  (0.15463, 0.01283),
    "Emotional Salience":                 (0.15181, 0.01252),
    "Narrative Absorption":               (0.14869, 0.01210),
    "Surprise / Novelty":                 (0.14757, 0.01206),
}
# z=+1σ → +11 pts (a clearly-above-typical clip lands ~65, the strongest real
# clips ~76, weakest ~22); clamp keeps scores in a believable, non-saturating
# band so nothing reads as a perfect/zero. Tuned 2026-06-03 vs the 28-run corpus.
KPI_Z_SPREAD = 11.0
KPI_SCORE_LO, KPI_SCORE_HI = 12, 94

YEO7_MAP_PATH = "/data/cache/mary_yeo7_vertex_map.npz"  # cached on sapient-data

# Per-vertex ISC noise ceiling (20484,) on sapient-data. ~90% of fsaverage5
# vertices are noise (ISC≈0); restricting each Yeo-network reduction to RESPONSIVE
# vertices (ISC > NOISE_CEILING_THRESHOLD), ISC-weighted, stops the noise from
# diluting the live engagement magnitudes ~10:1. Guarded load: if absent the serve
# falls back to the original all-vertex mean.
NOISE_CEILING_PATH = "/data/noise_ceiling_isc.npy"
NOISE_CEILING_THRESHOLD = 0.05

# Paper D retinotopic atlas (Benson-2014 resampled to fsaverage5). Holds the
# per-vertex (20484,) arrays varea/eccen/angle/sigma/hemi_id + the V1/V2/V3
# selection mask. Must be staged on the sapient-data volume before a
# `visual_field` read-out can run (see report — produced by
# Mary-Papers/paper-D-retinotopic/code/benson_atlas.py). Falls back gracefully
# (clear error) if absent.
BENSON_ATLAS_PATH = "/data/crossmodal/benson_fsaverage5.npz"

# Visual-field render constants — mirror Paper D render_analyze.py EXACTLY so the
# live field image matches the paper's method.
VF_CANVAS = 512
VF_FIELD_DEG = 20.0
VF_SIGMA_CLAMP = (0.5, 5.0)
VF_SMOOTH_PX = 2.0
VF_DEG_PER_PX = (2 * VF_FIELD_DEG) / VF_CANVAS

# Official microsoft/unilm BEATs inference source — vendored at build time into
# /opt/beats (same as data/extract_beats.py's batch image).
_UNILM_RAW = "https://raw.githubusercontent.com/microsoft/unilm/master/beats"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "curl")
    # Layer 1: backbone first — torch + setuptools<81 (provides pkg_resources for
    # the legacy openai-whisper sdist build, which imports pkg_resources).
    .pip_install(
        "setuptools<81",
        "wheel",
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "numpy>=1.26,<3",
    )
    # Layer 2: whisper aligner deps WITHOUT build isolation so their setup.py sees
    # the setuptools/torch installed above (mirror extract_qwen_ctx.py).
    .pip_install(
        "openai-whisper==20240930",
        "whisper-timestamped==1.15.4",
        extra_options="--no-build-isolation",
    )
    # Layer 3: everything else — original text deps + audio extractor deps.
    # Re-pin torch/torchaudio here: a modern timm + torchvision would otherwise
    # drag torch up to 2.12 and break the torchaudio==2.4.1 C-extension
    # (libtorchaudio.so undefined symbol). Holding torch==2.4.1 (+ the matching
    # torchvision 0.19.1 ABI) forces pip to pick a timm compatible with it.
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.51.3",     # >=4.51 for the qwen3 architecture
        "accelerate>=0.34",
        "sentencepiece==0.2.0",
        "tokenizers>=0.21",
        "scipy>=1.13",
        "pyyaml>=6",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
        "nilearn>=0.10,<0.12",
        "nibabel>=5",
        "supabase>=2",
        "fastapi[standard]",
        "requests",
        "yt-dlp",                   # fetch media from IG/TikTok/YouTube links
        # audio extractor deps:
        # NOTE: the batch BEATs image pins timm==0.4.5, but the official
        # microsoft/unilm BEATs inference source (BEATs.py/backbone.py/modules.py)
        # imports NO timm — and transformers==4.51.3's timm_wrapper needs a modern
        # timm (ImageNetInfo, added in 0.9.x). Pin a timm contemporaneous with
        # torch 2.4 (has ImageNetInfo, doesn't force a torch upgrade); BEATs
        # itself imports no timm so it's unaffected.
        "timm==1.0.11",
        "einops",
        "soundfile==0.12.1",
        "librosa==0.10.2",
        # visual_field read-out (Paper D) — pure-CPU Gaussian-splat renderer.
        "pillow>=10",
        "matplotlib>=3.7",
    )
    # Vendor the official microsoft/unilm BEATs source (inference subset) → /opt/beats.
    .run_commands(
        "mkdir -p /opt/beats",
        f"curl -fsSL {_UNILM_RAW}/BEATs.py    -o /opt/beats/BEATs.py",
        f"curl -fsSL {_UNILM_RAW}/backbone.py -o /opt/beats/backbone.py",
        f"curl -fsSL {_UNILM_RAW}/modules.py  -o /opt/beats/modules.py",
        "touch /opt/beats/__init__.py",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch",
          "NILEARN_DATA": "/cache/nilearn", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    .add_local_dir(
        str(_MARY_REPO), remote_path="/root/maryrepo",
        ignore=["**/__pycache__", "**/.venv", "**/*.pt", "**/*.npy",
                "**/.git", "**/logs/**"],
    )
    .add_local_dir(
        str(_QUALIA), remote_path="/root/qualia",
        ignore=["**/__pycache__", "**/.venv", "**/*.pt", "**/*.npy", "**/.git"],
    )
)

data_volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("mary-hf-cache", create_if_missing=True)
VOLUMES = {"/data": data_volume, "/cache": hf_cache_volume}
SECRETS = [
    modal.Secret.from_name("hf-token"),
    modal.Secret.from_name("mary-supabase"),
    modal.Secret.from_name("mary-pipeline-secret"),
    modal.Secret.from_name("sapient-rapidapi"),  # RAPIDAPI_KEY for the link-download ladder
]

app = modal.App(APP_NAME)


def _setup_path() -> None:
    """Make `qualia` (at /root) and `mary`/`extractors` (at /root/maryrepo) importable."""
    for p in ("/root", "/root/maryrepo"):
        if p not in sys.path:
            sys.path.insert(0, p)


# ── One-time: build & cache the fsaverage5 vertex → Yeo-7 network map ─────────
@app.function(image=image, volumes=VOLUMES, timeout=30 * 60)
def prepare_yeo7() -> dict:
    """Build a (20484,) int8 array of Yeo-7 network index per vertex (−1 = none),
    cached to the sapient-data volume so the warm server reduces in pure numpy.
    Self-contained (nilearn) — no dependency on the Mary-Papers shared module."""
    import re
    import numpy as np
    import nibabel as nib
    from nilearn import datasets, surface

    N_VERTICES, N_PARCELS = 20484, 1000
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=N_PARCELS, yeo_networks=7, resolution_mm=1)
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    atlas_img = nib.load(atlas.maps)

    def _proj(mesh):
        for interp in ("nearest_most_frequent", "nearest"):
            try:
                return surface.vol_to_surf(atlas_img, mesh, radius=3.0, interpolation=interp)
            except ValueError:
                continue
        raise RuntimeError("vol_to_surf failed for both interpolation names")

    lh, rh = _proj(fsavg.pial_left), _proj(fsavg.pial_right)
    labels = np.nan_to_num(np.concatenate([lh, rh]), nan=0.0).astype(np.int32)  # (20484,) 0..1000

    def _decode(raw):
        out = [(l.decode() if isinstance(l, bytes) else str(l)) for l in raw]
        if len(out) == N_PARCELS + 1 and "background" in out[0].lower():
            out = out[1:]
        return out

    parcel_labels = _decode(atlas.labels)  # parcel id 1..1000

    def _token(label: str) -> str | None:
        m = re.search(r"7Networks_(?:LH|RH)_([A-Za-z]+)_", label)
        return m.group(1) if m else None

    parcel_yeo_idx = np.full(N_PARCELS + 1, -1, dtype=np.int8)  # index 0 unused
    for pid, lab in enumerate(parcel_labels, start=1):
        tok = _token(lab)
        name = SCHAEFER_TOKEN_TO_YEO7.get(tok) if tok else None
        parcel_yeo_idx[pid] = YEO7_ORDER.index(name) if name in YEO7_ORDER else -1

    vertex_yeo = parcel_yeo_idx[labels]  # (20484,) int8 network index, −1 = unlabeled
    Path(YEO7_MAP_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(YEO7_MAP_PATH, vertex_yeo=vertex_yeo, yeo7_order=np.array(YEO7_ORDER))
    data_volume.commit()
    counts = {YEO7_ORDER[i]: int((vertex_yeo == i).sum()) for i in range(len(YEO7_ORDER))}
    return {"path": YEO7_MAP_PATH, "n_labeled": int((vertex_yeo >= 0).sum()), "per_network": counts}


# ── SlowFast R101 visual stream — its own container ──────────────────────────
# pytorchvideo 0.1.5 needs torch==2.0.1 (functorch import breaks on torch>=2.1),
# which is incompatible with the serve image. So the visual stream runs in a
# SEPARATE Modal container; the warm MaryServer calls
# `SlowFastExtractor().featurize.remote(video_bytes)` to get (T_2Hz, 2304) and
# merges it into the features dict. The featurize logic mirrors
# data/extract_slowfast.py EXACTLY so live visual features match the training
# contract the slowfast stream-encoder weights were trained against.
slowfast_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.0.1", "torchvision==0.15.2", "pytorchvideo==0.1.5",
        "fvcore", "decord==0.6.0", "numpy>=1.23,<2", "huggingface_hub>=0.25,<1",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"})
)

_SF = dict(HIDDEN_DIM=2304, RATE_HZ=2.0, CLIP_SECONDS=2.0,
           SLOW_FRAMES=8, FAST_FRAMES=32, SIDE=256)


# Undecorated mixin (mirrors `_MaryServerBase`) so a SEPARATE app — e.g.
# `mary-serve-api` — can register its OWN SlowFast @app.cls by subclassing this
# without re-decorating an already-bound class. The featurize math lives here and
# is shared byte-for-byte across apps.
class _SlowFastExtractorBase:
    @modal.enter()
    def load(self):
        import torch
        m = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r101", pretrained=True)
        self.model = m.eval().cuda()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._cap: dict = {}
        head = self.model.blocks[-1]
        target = getattr(head, "dropout", None) or getattr(head, "pool", None)
        assert target is not None, "could not find SlowFast head pool/dropout"
        target.register_forward_hook(lambda _m, _i, o: self._cap.__setitem__("feat", o))

    @modal.method()
    def featurize(self, video_bytes: bytes, max_sec: int = 60) -> bytes:
        """Raw video bytes → npy bytes of (T_2Hz, 2304) float16. b"" if too short.
        Mirrors data/extract_slowfast.py: 32-frame fast / 8-frame slow clips ending
        at each 2 Hz step, head pooled features (slow 2048 + fast 256 = 2304)."""
        import io
        import tempfile
        import numpy as np
        import torch
        from decord import VideoReader, cpu

        HID, RATE, CLIP = _SF["HIDDEN_DIM"], _SF["RATE_HZ"], _SF["CLIP_SECONDS"]
        SLOW, FAST, SIDE = _SF["SLOW_FRAMES"], _SF["FAST_FRAMES"], _SF["SIDE"]
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1).cuda()
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1).cuda()

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        with open(tmp, "wb") as f:
            f.write(video_bytes)
        try:
            vr = VideoReader(tmp, ctx=cpu(0))
        except Exception as e:
            print(f"[slowfast] decode fail: {e!r}")
            return b""
        fps = float(vr.get_avg_fps()) or 30.0
        n_frames = len(vr)
        duration_s = min(n_frames / fps, float(max_sec))
        n_steps = int(duration_s * RATE)
        if n_steps < 2:
            return b""

        def pack(frames):  # (C,T_fast,H,W) → [slow, fast]
            idx = torch.linspace(0, frames.shape[1] - 1, SLOW, device=frames.device).long()
            return [frames.index_select(1, idx).unsqueeze(0), frames.unsqueeze(0)]

        feats = np.empty((n_steps, HID), dtype=np.float16)
        for step in range(n_steps):
            t_end = (step + 1) / RATE
            t_start = max(0.0, t_end - CLIP)
            fidx = np.linspace(t_start * fps, min(n_frames - 1, t_end * fps - 1e-3), FAST)
            fidx = np.clip(np.round(fidx), 0, n_frames - 1).astype(np.int64)
            clip = vr.get_batch(list(fidx)).asnumpy()  # (T,H,W,3) uint8
            c = torch.from_numpy(clip).to("cuda").permute(3, 0, 1, 2).float() / 255.0
            c = torch.nn.functional.interpolate(c, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
            c = (c - mean) / std
            self._cap.clear()
            with torch.no_grad():
                _ = self.model(pack(c))
            t = self._cap["feat"].float()
            ax = next((i for i, s in enumerate(t.shape) if s == HID), None)
            assert ax is not None, f"no axis == {HID} in slowfast feat {tuple(t.shape)}"
            red = [i for i in range(t.dim()) if i != ax]
            if red:
                t = t.mean(dim=red)
            feats[step] = t.reshape(-1).cpu().numpy().astype(np.float16)

        buf = io.BytesIO()
        np.save(buf, feats)
        return buf.getvalue()


# The live-app binding of the SlowFast extractor. A SEPARATE deployed app
# (api_serve.py) registers its OWN subclass of `_SlowFastExtractorBase` on its app
# so its serving class can `.remote()`-call a SlowFast extractor it actually owns
# (calling THIS one from another app raises the "App ... is not running" hydration
# error and silently degrades video → audio-only).
@app.cls(
    image=slowfast_image, gpu="A10G", volumes=VOLUMES,
    secrets=[modal.Secret.from_name("hf-token")],
    min_containers=0, scaledown_window=300, timeout=10 * 60,
)
class SlowFastExtractor(_SlowFastExtractorBase):
    pass


# ── Cross-modal image decoder (Paper C) — its OWN A100 container ─────────────
# MindEye2 (~1.9B params: BrainNetwork backbone + BrainDiffusionPrior + SDXL
# unCLIP) + a GIT captioner. Its dependency pins (torch 2.1.0 / transformers
# 4.37.2 / diffusers 0.23.0 / sgm) are incompatible with the serve image
# (torch 2.4.1 / transformers 4.51.3), so — exactly like SlowFastExtractor — it
# runs in a SEPARATE container. The warm MaryServer calls
# `CrossModalDecoder().decode.remote(synth_fmri_bytes)` and gets back
# (png_bytes, caption). This is a faithful port of paper-C decode_mindeye2.py:
# the fsaverage5 synthetic fMRI is bridged to NSD-Subject01 voxel space by the
# adapter, then run through the frozen pretrained MindEye2 decoder.
#
# REQUIRED VOLUME ASSETS (staged before deploy; see report):
#   /data/crossmodal/adapters/pca_adapter.npz   — PCAAdapter.save() output
#       (training-free fsaverage5->NSD bridge; optionally carries NSD beta
#        per-voxel mu/sd for amplitude calibration). If the Ridge adapter is
#        preferred, stage /data/crossmodal/adapters/ridge_adapter.npz instead
#        and set CROSSMODAL_ADAPTER=ridge.
# MindEye2 decoder weights download from HF dataset pscotti/MindEyeV2 at
# runtime (needs the hf-token secret): train_logs/<model>/last.pth,
# unclip6_epoch0_step110000.ckpt, bigG_to_L_epoch8.pth. The unclip config +
# vendored MindEye2 source (models.py, modeling_git.py, unclip6_infer.yaml,
# utils.py) are baked into the image below from the local mindeye2_vendor dir.
_ME2_VENDOR = (_HERE.parents[3] / "sapient-research" / "Mary-Papers" /
               "paper-C-cross-modal" / "code" / "mindeye2_vendor") \
    if len(_HERE.parents) >= 4 else Path("/nonexistent-me2-vendor")
_ME2_ADAPTER_SRC = (_HERE.parents[3] / "sapient-research" / "Mary-Papers" /
                    "paper-C-cross-modal" / "code" / "adapter.py") \
    if len(_HERE.parents) >= 4 else Path("/nonexistent-adapter")

CROSSMODAL_HF_REPO = "pscotti/MindEyeV2"
CROSSMODAL_MODEL_NAME = "final_subj01_pretrained_40sess_24bs"
CROSSMODAL_HIDDEN_DIM = 4096
CROSSMODAL_ADAPTER_DIR = "/data/crossmodal/adapters"

crossmodal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # Pins match the official MindEye2 setup.sh (mirror paper-C decode_mindeye2.py).
    .pip_install(
        "torch==2.1.0", "torchvision==0.16.0", "xformers==0.0.22.post7",
        "numpy==1.26.4", "h5py==3.10.0", "tqdm", "matplotlib==3.8.2",
        "huggingface_hub==0.20.3", "safetensors>=0.4",
        "transformers==4.37.2", "diffusers==0.23.0", "accelerate==0.24.1",
        "open-clip-torch==2.24.0", "omegaconf==2.3.0", "einops",
        "kornia==0.7.1", "scipy", "Pillow", "scikit-image==0.22.0",
        "pytorch-lightning==2.0.1", "torchmetrics==1.3.0.post0",
        "webdataset==0.2.73", "ftfy", "regex",
    )
    .pip_install("dalle2-pytorch==1.15.6")
    .run_commands("pip install git+https://github.com/openai/CLIP.git --no-deps")
    .run_commands(
        "git clone --depth 1 https://github.com/MedARC-AI/MindEyeV2.git /opt/me2repo",
        "mkdir -p /opt/genroot && cp -r /opt/me2repo/src/generative_models /opt/genroot/generative_models",
        "touch /opt/genroot/generative_models/__init__.py",
        "pip install --no-deps -e /opt/genroot/generative_models || echo 'sgm editable (no-deps)'",
    )
    .env({"HF_HOME": "/cache/huggingface",
          "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"})
    # local-file adds MUST be the last image build steps (Modal constraint).
    .add_local_dir(str(_ME2_VENDOR), remote_path="/root/me2")
    .add_local_file(str(_ME2_ADAPTER_SRC), remote_path="/root/me2_adapter.py")
)


# NOTE: temporarily pointed at the lightweight serve `image` (not the heavy
# `crossmodal_image`) so `modal deploy` does NOT build the MindEye2 stack while
# cross_modal_image is DEFERRED (weights + pca_adapter.npz not staged yet). The
# class is never instantiated (run_job short-circuits the capability), so this is
# inert. Restore `image=crossmodal_image` when re-enabling cross-modal.
@app.cls(
    image=image, gpu="A100-40GB", volumes=VOLUMES,
    secrets=[modal.Secret.from_name("hf-token")],
    min_containers=0, scaledown_window=300, timeout=30 * 60,
)
class CrossModalDecoder:
    """Frozen MindEye2 + GIT decode of Mary's synthetic fMRI → (image, caption).

    Port of paper-C decode_mindeye2.py for a SINGLE synthetic-fMRI vector. The
    heavy decoder weights are lazily downloaded + assembled on first call inside
    `decode` (kept out of @modal.enter so the import-only `selfcheck` is cheap)."""

    def _hf_dl(self, filename: str) -> str:
        import os
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=CROSSMODAL_HF_REPO, filename=filename,
                               repo_type="dataset", token=os.environ.get("HF_TOKEN"))

    def _load_adapter(self):
        """Load the fsaverage5→NSD bridge from the volume. Returns a callable
        adapter (PCA or Ridge). Raises a clear error if the asset is missing."""
        import os
        import sys
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        import importlib.util
        spec = importlib.util.spec_from_file_location("me2_adapter", "/root/me2_adapter.py")
        me2_adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(me2_adapter)

        kind = os.environ.get("CROSSMODAL_ADAPTER", "pca").lower()
        if kind == "ridge":
            path = f"{CROSSMODAL_ADAPTER_DIR}/ridge_adapter.npz"
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"cross-modal Ridge adapter missing at {path} — stage it on the "
                    "sapient-data volume (RidgeAdapter.save) before running this read-out")
            return me2_adapter.RidgeAdapter.load(path)
        path = f"{CROSSMODAL_ADAPTER_DIR}/pca_adapter.npz"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"cross-modal PCA adapter missing at {path} — stage it on the "
                "sapient-data volume (PCAAdapter.save) before running this read-out")
        return me2_adapter.PCAAdapter.load(path)

    @modal.method()
    def decode(self, synth_fmri_bytes: bytes,
               prior_timesteps: int = 20, unclip_steps: int = 38,
               num_samples: int = 4) -> dict:
        """npy bytes of (T_TR, 20484) OR (20484,) Mary synthetic fMRI →
        {ok, image_b64, caption}. On any failure returns {ok: False, error}.

        Faithful port of paper-C decode_mindeye2.py::decode for one vector. The
        Mary prediction is adapted fsaverage5(20484)→NSD(15724), then the frozen
        MindEye2 backbone+prior+SDXL-unCLIP produce an image and the GIT
        captioner a caption. num_samples defaults lower than the paper's 16 to
        bound A100-40GB latency for the interactive product path."""
        import base64
        import io
        import sys

        try:
            import numpy as np
            import torch
            import torch.nn as nn
            from torchvision import transforms
            from omegaconf import OmegaConf

            sys.path.insert(0, "/root/me2")
            sys.path.append("/opt/genroot")
            sys.path.append("/opt/genroot/generative_models")
            torch.backends.cuda.matmul.allow_tf32 = True
            device = "cuda"

            import utils  # noqa: F401
            from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior
            from modeling_git import GitForCausalLMClipEmb
            from transformers import AutoProcessor
            from generative_models.sgm.models.diffusion import DiffusionEngine
            from generative_models.sgm.util import append_dims

            # ---- adapt Mary fsaverage5 → NSD voxel space ----
            verts = np.load(io.BytesIO(synth_fmri_bytes)).astype(np.float32)
            adapter = self._load_adapter()
            X = adapter(verts).astype(np.float32)            # (15724,)
            num_voxels = X.shape[0]
            clip_seq_dim, clip_emb_dim = 256, 1664

            # ---- MindEye2 model (RidgeRegression + BrainNetwork + prior) ----
            class MindEyeModule(nn.Module):
                def __init__(self): super().__init__()
                def forward(self, x): return x

            class RidgeRegression(nn.Module):
                def __init__(self, input_sizes, out_features):
                    super().__init__()
                    self.out_features = out_features
                    self.linears = nn.ModuleList([nn.Linear(s, out_features) for s in input_sizes])
                def forward(self, x, subj_idx):
                    return self.linears[subj_idx](x[:, 0]).unsqueeze(1)

            model = MindEyeModule()
            model.ridge = RidgeRegression([num_voxels], out_features=CROSSMODAL_HIDDEN_DIM)
            model.backbone = BrainNetwork(
                h=CROSSMODAL_HIDDEN_DIM, in_dim=CROSSMODAL_HIDDEN_DIM, seq_len=1,
                clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
                blurry_recon=False, clip_scale=1.)
            prior_network = PriorNetwork(
                dim=clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim // 52,
                causal=False, num_tokens=clip_seq_dim, learned_query_mode="pos_emb")
            model.diffusion_prior = BrainDiffusionPrior(
                net=prior_network, image_embed_dim=clip_emb_dim,
                condition_on_text_encodings=False, timesteps=100,
                cond_drop_prob=0.2, image_embed_scale=None)

            ckpt = torch.load(self._hf_dl(f"train_logs/{CROSSMODAL_MODEL_NAME}/last.pth"),
                              map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.to(device).eval().requires_grad_(False)

            # ---- GIT captioner + CLIPConverter ----
            processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
            clip_text_model = GitForCausalLMClipEmb.from_pretrained(
                "microsoft/git-large-coco").to(device).eval().requires_grad_(False)
            clip_text_seq_dim, clip_text_emb_dim = 257, 1024

            class CLIPConverter(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
                    self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
                def forward(self, x):
                    x = x.permute(0, 2, 1)
                    x = self.linear1(x)
                    return self.linear2(x.permute(0, 2, 1))

            clip_convert = CLIPConverter()
            cc_sd = torch.load(self._hf_dl("bigG_to_L_epoch8.pth"),
                               map_location="cpu")["model_state_dict"]
            clip_convert.load_state_dict(cc_sd, strict=True)
            clip_convert.to(device).eval().requires_grad_(False)

            # ---- SDXL unCLIP ----
            cfg = OmegaConf.to_container(
                OmegaConf.load("/root/me2/unclip6_infer.yaml"), resolve=True)
            p = cfg["model"]["params"]
            p["sampler_config"]["params"]["num_steps"] = unclip_steps
            p["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"
            de = DiffusionEngine(
                network_config=p["network_config"],
                denoiser_config=p["denoiser_config"],
                first_stage_config=p["first_stage_config"],
                conditioner_config=p["conditioner_config"],
                sampler_config=p["sampler_config"],
                scale_factor=p["scale_factor"],
                disable_first_stage_autocast=p["disable_first_stage_autocast"])
            de.eval().requires_grad_(False).to(device)
            unclip_ckpt = torch.load(self._hf_dl("unclip6_epoch0_step110000.ckpt"),
                                     map_location="cpu")
            de.load_state_dict(unclip_ckpt["state_dict"], strict=False)

            batch = {"jpg": torch.randn(1, 3, 1, 1).to(device),
                     "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
                     "crop_coords_top_left": torch.zeros(1, 2).to(device)}
            vector_suffix = de.conditioner(batch)["vector"].to(device)
            clip_img_embedder = de.conditioner.embedders[0]
            clip_img_embedder.to(device).eval()

            def embed_for_rerank(imgs01):
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                    return clip_img_embedder(imgs01)

            RECON_CHUNK = 2

            def unclip_recon(x, n=1, offset_noise_level=0.04):
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), de.ema_scope():
                    z = torch.randn(n, 4, 96, 96).to(device)
                    c = {"crossattn": x.repeat(n, 1, 1),
                         "vector": vector_suffix.repeat(n, 1)}
                    tokens = torch.randn_like(x)
                    uc = {"crossattn": tokens.repeat(n, 1, 1),
                          "vector": vector_suffix.repeat(n, 1)}
                    for k in c:
                        c[k], uc[k] = c[k][:n].to(device), uc[k][:n].to(device)
                    noise = torch.randn_like(z)
                    sigmas = de.sampler.discretization(de.sampler.num_steps)
                    sigma = sigmas[0].to(z.device)
                    if offset_noise_level > 0.0:
                        noise = noise + offset_noise_level * append_dims(
                            torch.randn(z.shape[0], device=z.device), z.ndim)
                    noised_z = (z + noise * append_dims(sigma, z.ndim)) / \
                        torch.sqrt(1.0 + sigmas[0] ** 2.0)
                    def denoiser(xx, ss, cc): return de.denoiser(de.model, xx, ss, cc)
                    samples_z = de.sampler(denoiser, noised_z, cond=c, uc=uc)
                    outs = []
                    for j in range(samples_z.shape[0]):
                        sx = de.decode_first_stage(samples_z[j:j + 1])
                        outs.append(torch.clamp((sx * .8 + .2), 0.0, 1.0))
                    return torch.cat(outs, dim=0)

            def unclip_recon_n(x, n):
                outs, done = [], 0
                while done < n:
                    b = min(RECON_CHUNK, n - done)
                    outs.append(unclip_recon(x, n=b))
                    done += b
                return torch.cat(outs, dim=0)

            # ---- run (single vector) ----
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                voxel = torch.from_numpy(X)[None, None].to(device)   # (1,1,15724)
                voxel_ridge = model.ridge(voxel, 0)
                backbone, clip_voxels, _ = model.backbone(voxel_ridge)
                prior_out = model.diffusion_prior.p_sample_loop(
                    backbone.shape, text_cond=dict(text_embed=backbone),
                    cond_scale=1., timesteps=prior_timesteps)
                pred_caption_emb = clip_convert(prior_out)
                gen_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
                caption = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

                samples = unclip_recon_n(prior_out, max(1, num_samples))
                if num_samples > 1:
                    cand_emb = embed_for_rerank(samples)
                    ce = torch.nn.functional.normalize(cand_emb.flatten(1).float(), dim=-1)
                    tv = torch.nn.functional.normalize(clip_voxels.flatten(1).float(), dim=-1)
                    sims = (ce @ tv.T).squeeze(-1)
                    best = int(torch.argmax(sims).item())
                    best_sample = samples[best:best + 1]
                else:
                    best_sample = samples
                img = transforms.Resize((256, 256))(best_sample).float().cpu().numpy()[0]

            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray((img.transpose(1, 2, 0) * 255).astype("uint8")).save(buf, format="PNG")
            return {"ok": True,
                    "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "caption": caption}
        except Exception as e:
            print(f"[cross-modal] decode failed: {e!r}")
            return {"ok": False, "error": str(e) or repr(e)}

    @modal.method()
    def selfcheck(self) -> dict:
        """Import-only smoke test — validates the heavy image without diffusion."""
        import sys
        import torch
        sys.path.insert(0, "/root/me2")
        sys.path.append("/opt/genroot")
        sys.path.append("/opt/genroot/generative_models")
        import utils  # noqa: F401
        from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior  # noqa: F401
        from modeling_git import GitForCausalLMClipEmb  # noqa: F401
        from generative_models.sgm.models.diffusion import DiffusionEngine  # noqa: F401
        return {"ok": True, "torch": torch.__version__,
                "cuda": torch.cuda.is_available()}


# ── Shared serving logic ──────────────────────────────────────────────────────
# ALL of Mary's warm-serving logic — load(), the network/KPI reducers, storage
# helpers, _build_artifact, run_job, probes — lives on this UNDECORATED base so a
# SECOND, independent Modal app (mary/modal/api_serve.py, app `mary-serve-api`)
# can reuse it byte-for-byte by simply subclassing + re-decorating, WITHOUT
# touching the live `mary-serve` deploy. Modal 1.x discovers @modal.enter/@method
# across the full MRO (modal._partial_function._find_partial_methods_for_user_cls
# walks reversed(cls.mro())), so the thin decorated subclasses below pick up every
# method here automatically.
class _MaryServerBase:
    @modal.enter()
    def load(self):
        import numpy as np
        _setup_path()
        from qualia.core.mary_engine import MaryEngine
        from extractors.qwen_ctx import QwenCtxExtractor
        from extractors.whisper import WhisperExtractor
        from extractors.beats import BeatsExtractor

        t0 = time.time()
        self.engine = MaryEngine.from_channel("improving", device="cuda")
        self.qwen = QwenCtxExtractor(device="cuda")
        # Audio stream extractors (whisper encoder + official BEATs). The
        # whisper-timestamped forced aligner is lazy-loaded on first audio request
        # (extractors.qwen_ctx.forced_align) to keep warm-load reasonable.
        self.whisper = WhisperExtractor(device="cuda")
        self.beats = BeatsExtractor(device="cuda")
        with np.load(YEO7_MAP_PATH, allow_pickle=True) as z:
            self.vertex_yeo = z["vertex_yeo"]
            self.yeo7_order = [str(x) for x in z["yeo7_order"]]
        # Per-vertex ISC noise ceiling — restrict network reductions to responsive
        # vertices. Guarded: missing → None → all-vertex fallback in the reducers.
        self.vertex_isc = None
        try:
            if os.path.exists(NOISE_CEILING_PATH):
                isc = np.load(NOISE_CEILING_PATH).astype(np.float64)   # (20484,)
                if isc.shape[0] == self.vertex_yeo.shape[0]:
                    self.vertex_isc = isc
                    n_resp = int((isc > NOISE_CEILING_THRESHOLD).sum())
                    print(f"[mary-serve] ISC ceiling loaded ({n_resp} responsive "
                          f"vertices @ ISC>{NOISE_CEILING_THRESHOLD}); "
                          f"network reductions restricted to responsive vertices.")
                else:
                    print(f"[mary-serve] ISC ceiling shape {isc.shape} != "
                          f"{self.vertex_yeo.shape}; ignoring (all-vertex fallback).")
            else:
                print(f"[mary-serve] {NOISE_CEILING_PATH} missing; all-vertex fallback.")
        except Exception as e:
            print(f"[mary-serve] ISC ceiling load failed ({e!r}); all-vertex fallback.")
        from supabase import create_client
        self.sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        print(f"[mary-serve] warm in {time.time()-t0:.1f}s "
              f"(d_model={self.engine.d_model}, ver={self.engine.version})")

    # — Supabase helpers —
    def _update(self, run_id: str, **fields):
        try:
            self.sb.table("mary_runs").update(fields).eq("id", run_id).execute()
        except Exception as e:  # never let a status write kill the job
            print(f"[mary-serve] mary_runs update failed for {run_id}: {e!r}")

    def _now(self) -> str:
        import datetime as _dt
        return _dt.datetime.now(_dt.timezone.utc).isoformat()

    # — Vertices (T,20484) → per-Yeo-7-network response —
    def _reduce_networks(self, verts) -> list[dict]:
        import numpy as np
        # Mean |predicted activation| over time per vertex — the same engagement
        # magnitude the research demo reports (~0.11–0.13), robust for short
        # inputs (temporal std collapses to ~0 on a few timesteps).
        per_vertex = np.abs(verts).mean(axis=0).astype(np.float64)
        isc = self.vertex_isc
        # NOTE: restricting to responsive (ISC>thr) vertices shifts the raw
        # magnitudes vs. the old all-vertex mean, so KPI_BASELINE (mean,std) may need
        # recomputation against this reduction. We do NOT change KPI_BASELINE here.
        out = []
        for i, name in enumerate(self.yeo7_order):
            net = self.vertex_yeo == i
            if isc is not None:
                m = net & (isc > NOISE_CEILING_THRESHOLD)
                if m.any():
                    # ISC-weighted mean over responsive vertices in this network.
                    w = isc[m]
                    val = float((per_vertex[m] * w).sum() / w.sum())
                else:
                    val = float(per_vertex[net].mean()) if net.any() else 0.0
            else:
                val = float(per_vertex[net].mean()) if net.any() else 0.0
            out.append({"network": name, "value": val, "plain": YEO7_PLAIN.get(name, name)})
        return out

    # — Vertices (T_TR,20484) → per-timestep × per-network (T_TR, 7) —
    def _network_timecourse(self, verts):
        """Reduce EACH timestep of the predicted fMRI to the 7 Yeo networks.
        Returns (T_TR, 7) float64 of |activation| averaged within each network's
        vertices — the temporal analogue of `_reduce_networks` (which is the
        time-mean of this). Reuses the cached self.vertex_yeo network map."""
        import numpy as np
        mag = np.abs(verts).astype(np.float64)          # (T_TR, 20484)
        T = mag.shape[0]
        n = len(self.yeo7_order)
        isc = self.vertex_isc
        tc = np.zeros((T, n), dtype=np.float64)
        # Restrict each network to responsive (ISC>thr) vertices, ISC-weighted. Falls
        # back to the all-vertex mean if the ISC map is missing or a network has no
        # responsive vertices. (KPI_BASELINE may need recomputation — see _reduce_networks.)
        for i in range(n):
            net = self.vertex_yeo == i
            if isc is not None:
                m = net & (isc > NOISE_CEILING_THRESHOLD)
                if m.any():
                    w = isc[m]                          # (n_resp,)
                    tc[:, i] = (mag[:, m] * w).sum(axis=1) / w.sum()
                    continue
            if net.any():
                tc[:, i] = mag[:, net].mean(axis=1)
        return tc

    # — (T_TR,7) network timecourse → per-second 10-KPI block —
    def _build_kpi_block(self, verts, duration_sec):
        """Translate Mary's network timecourse into the live 10-KPI surface the
        verdict-v3 components consume. Returns a dict with:
          kpi_time_series : {KPIname: [per-second float]}      (len == n_sec)
          kpi_summary     : {KPIname: {mean,std,min,max,argmin_sec,argmax_sec}}
          kpi_peaks       : {KPIname: {peaks:[...], lows:[...]}}  (~4 each)
          composites      : [{name,score,tier} x5]
          summary         : {score, grade, peak_multiplier}

        HONESTY: every KPI here is a fixed weighted blend of Mary's 7 predicted
        networks (KPI_FROM_NETWORKS) — a translation of one prediction into the
        product vocabulary, NOT 10 separate measurements."""
        import numpy as np

        tc = self._network_timecourse(verts)             # (T_TR, 7)
        T_tr = tc.shape[0]
        net_idx = {name: i for i, name in enumerate(self.yeo7_order)}

        # Resample the T_TR timecourse onto whole-second buckets so the per-second
        # series length matches duration_sec (the UI assumes ~1 Hz). When T_TR is
        # already close to duration we still bucket to integer seconds.
        n_sec = max(1, int(round(duration_sec or 0)) or T_tr)
        # map each second s -> the slice of TR rows covering it
        edges = np.linspace(0, T_tr, n_sec + 1).astype(int)
        net_per_sec = np.zeros((n_sec, len(self.yeo7_order)), dtype=np.float64)
        for s in range(n_sec):
            a, b = edges[s], max(edges[s] + 1, edges[s + 1])
            b = min(b, T_tr)
            net_per_sec[s] = tc[a:b].mean(axis=0) if b > a else tc[min(a, T_tr - 1)]

        # KPI per-second = weighted (sum-normalized) blend of networks per second.
        kpi_ts = {}
        for kpi, weights in KPI_FROM_NETWORKS.items():
            wsum = sum(weights.values()) or 1.0
            acc = np.zeros(n_sec, dtype=np.float64)
            for net, w in weights.items():
                j = net_idx.get(net)
                if j is not None:
                    acc += (w / wsum) * net_per_sec[:, j]
            kpi_ts[kpi] = acc

        # Per-KPI summary + peaks/lows.
        kpi_summary, kpi_peaks = {}, {}
        for kpi in KPI_NAMES:
            arr = kpi_ts[kpi]
            mean = float(arr.mean())
            std = float(arr.std())
            kpi_summary[kpi] = {
                "mean": round(mean, 6), "std": round(std, 6),
                "min": round(float(arr.min()), 6), "max": round(float(arr.max()), 6),
                "argmin_sec": int(arr.argmin()), "argmax_sec": int(arr.argmax()),
            }
            # z-score each second; pick top peaks + lows.
            z = (arr - mean) / (std + 1e-9)
            order = np.argsort(arr)
            lows_idx = order[: min(4, n_sec)]
            peaks_idx = order[::-1][: min(4, n_sec)]

            def _pt(t):
                rank = float((arr <= arr[t]).mean()) * 100.0
                return {"t_sec": int(t), "value": round(float(arr[t]), 6),
                        "zscore": round(float(z[t]), 4), "percentile": round(rank, 1)}

            kpi_peaks[kpi] = {
                "peaks": [_pt(int(t)) for t in peaks_idx],
                "lows": [_pt(int(t)) for t in lows_idx],
            }

        # Convert a raw KPI mean into a 0-100 score RELATIVE to the corpus
        # baseline (z-score). This is what makes content differentiate: a clip
        # that drives a network harder than the typical clip scores above 50,
        # weaker below 50, so A/B shows a real winner instead of everything ~51.
        # KPIs without a baseline fall back to the old absolute map.
        def _score_of(kpi, mean_val):
            base = KPI_BASELINE.get(kpi)
            if not base:
                return max(0, min(100, int(round(mean_val * 500))))
            base_mean, base_std = base
            z = (mean_val - base_mean) / (base_std + 1e-9)
            return max(KPI_SCORE_LO, min(KPI_SCORE_HI, int(round(50 + z * KPI_Z_SPREAD))))

        kpi_scores = {k: _score_of(k, kpi_summary[k]["mean"]) for k in KPI_NAMES}

        # Composites — weighted blend of KPI scores; tier per verdict banding.
        composites = []
        for cname, weights in COMPOSITE_FROM_KPIS.items():
            wsum = sum(weights.values()) or 1.0
            cscore = sum((w / wsum) * kpi_scores.get(k, 0) for k, w in weights.items())
            cscore = int(round(cscore))
            tier = "strength" if cscore >= 70 else "okay" if cscore >= 40 else "weakness"
            composites.append({"name": cname, "score": cscore, "tier": tier})

        # Overall summary score = mean of composite scores; grade + peak multiplier.
        overall = int(round(sum(c["score"] for c in composites) / max(1, len(composites))))
        grade = ("A" if overall >= 85 else "B" if overall >= 70 else
                 "C" if overall >= 55 else "D" if overall >= 40 else "F")
        # peak_multiplier: ratio of the single best per-second composite signal to
        # its mean across the video (how much the best moment beats the baseline).
        comp_per_sec = np.zeros(n_sec, dtype=np.float64)
        for cname, weights in COMPOSITE_FROM_KPIS.items():
            wsum = sum(weights.values()) or 1.0
            for k, w in weights.items():
                comp_per_sec += (w / wsum) * kpi_ts.get(k, np.zeros(n_sec))
        cmean = float(comp_per_sec.mean()) or 1e-9
        peak_mult = round(float(comp_per_sec.max()) / cmean, 2) if cmean > 0 else 1.0

        # Per-second 7-Yeo-network timecourse, bucketed on the SAME `net_per_sec`
        # edges as the KPIs so each network series lines up second-for-second with
        # kpiTimeSeries (same length == n_sec). Keys are the canonical Yeo-7 names.
        network_ts = {
            name: [round(float(net_per_sec[s, net_idx[name]]), 6) for s in range(n_sec)]
            for name in self.yeo7_order
        }

        # camelCase keys to match the MaryArtifact TS contract
        # (src/lib/maryWorkspaceClient.ts), which the API passes straight
        # through from mary_runs.result with no normalization.
        return {
            "kpiTimeSeries": {k: [round(float(v), 6) for v in kpi_ts[k]] for k in KPI_NAMES},
            "networkTimeSeries": network_ts,
            "kpiSummary": kpi_summary,
            "kpiPeaks": kpi_peaks,
            "composites": composites,
            "verdictSummary": {"score": overall, "grade": grade, "peak_multiplier": peak_mult},
        }

    # — Persist the downloaded clip so the verdict can PLAY + scrub it —
    UPLOAD_BUCKET = "kairo-uploads"

    def _store_playable(self, local_path: str, run_id: str) -> str | None:
        """Upload the downloaded media to storage and return a long-lived signed
        URL, so the verdict media frame can play + scrub the actual video (the
        live analysis page does the same). Best-effort: any failure → None and
        the UI falls back to the cover poster."""
        try:
            ext = Path(local_path).suffix or ".mp4"
            key = f"mary/{run_id}{ext}"
            with open(local_path, "rb") as f:
                data = f.read()
            ctype = ("video/mp4" if ext in (".mp4", ".m4v", ".mov") else
                     "audio/mpeg" if ext in (".mp3", ".m4a", ".aac") else
                     "application/octet-stream")
            try:
                self.sb.storage.from_(self.UPLOAD_BUCKET).upload(
                    key, data, {"content-type": ctype, "upsert": "true"})
            except Exception:
                # already exists / older client signature — try update, then re-sign
                self.sb.storage.from_(self.UPLOAD_BUCKET).update(
                    key, data, {"content-type": ctype})
            res = self.sb.storage.from_(self.UPLOAD_BUCKET).create_signed_url(key, 31536000)
            if isinstance(res, dict):
                return res.get("signedURL") or res.get("signedUrl") or res.get("signed_url")
            return None
        except Exception as e:
            print(f"[mary-serve] _store_playable failed: {e!r}")
            return None

    def _store_image(self, image_bytes: bytes, run_id: str, suffix: str = "field") -> str | None:
        """Upload a generated PNG to the upload bucket and return a long-lived
        signed URL — mirrors _store_playable for image read-out artifacts
        (visual_field / cross_modal_image). Best-effort: any failure → None."""
        try:
            key = f"mary/{run_id}_{suffix}.png"
            try:
                self.sb.storage.from_(self.UPLOAD_BUCKET).upload(
                    key, image_bytes, {"content-type": "image/png", "upsert": "true"})
            except Exception:
                self.sb.storage.from_(self.UPLOAD_BUCKET).update(
                    key, image_bytes, {"content-type": "image/png"})
            res = self.sb.storage.from_(self.UPLOAD_BUCKET).create_signed_url(key, 31536000)
            if isinstance(res, dict):
                return res.get("signedURL") or res.get("signedUrl") or res.get("signed_url")
            return None
        except Exception as e:
            print(f"[mary-serve] _store_image failed: {e!r}")
            return None

    def _store_fmri(self, verts, run_id: str) -> str | None:
        """Persist the rawest predicted signal — the (T_TR, 20484) vertex
        timecourse — to storage as a compressed .npz, returning the storage path
        `fmri/<run_id>.npz` (NOT a signed URL: the raw fMRI is large and consumed
        server-side by API clients, fetched on demand). Mirrors _store_image's
        upload/update fallback. Best-effort: any failure → None."""
        try:
            import io
            import numpy as np
            buf = io.BytesIO()
            np.savez_compressed(buf, verts=np.asarray(verts, dtype=np.float32))
            data = buf.getvalue()
            key = f"fmri/{run_id}.npz"
            try:
                self.sb.storage.from_(self.UPLOAD_BUCKET).upload(
                    key, data, {"content-type": "application/octet-stream", "upsert": "true"})
            except Exception:
                self.sb.storage.from_(self.UPLOAD_BUCKET).update(
                    key, data, {"content-type": "application/octet-stream"})
            return key
        except Exception as e:
            print(f"[mary-serve] _store_fmri failed: {e!r}")
            return None

    # — visual_field read-out (Paper D): time-mean verts → V1/V2/V3 splat image —
    def _load_benson_atlas(self) -> dict:
        """Lazy-load + cache the Benson retinotopic atlas off the volume. Raises
        a clear, user-facing error if the asset isn't staged."""
        import numpy as np
        cached = getattr(self, "_benson", None)
        if cached is not None:
            return cached
        if not Path(BENSON_ATLAS_PATH).exists():
            raise FileNotFoundError(
                f"retinotopic atlas missing at {BENSON_ATLAS_PATH} — stage "
                "benson_fsaverage5.npz on the sapient-data volume "
                "(produced by paper-D benson_atlas.py) before a visual_field read-out")
        with np.load(BENSON_ATLAS_PATH) as z:
            self._benson = {k: z[k] for k in z.files}
        return self._benson

    def _vf_field_positions(self, atlas):
        """Per-vertex (x,y) visual-field position in deg — port of Paper D
        render_analyze.field_positions (Benson polar-angle convention)."""
        import numpy as np
        e = atlas["eccen"]
        theta = np.radians(180.0 - atlas["angle"])
        sh = np.where(atlas["hemi_id"] == 0, 1.0, -1.0)
        x = e * np.sin(theta) * sh
        y = -e * np.cos(theta)
        return x, y

    def _vf_render(self, activation, atlas, mask=None):
        """Gaussian-splat a (20484,) activation into a VF_CANVAS² visual-field
        image. Port of Paper D render_analyze.render. Returns (img, wsum)."""
        import numpy as np
        if mask is None:
            mask = atlas["mask"]
        idx = np.where(mask)[0]
        x, y = self._vf_field_positions(atlas)
        sig = np.clip(atlas["sigma"], *VF_SIGMA_CLAMP)
        px = (x + VF_FIELD_DEG) / (2 * VF_FIELD_DEG) * (VF_CANVAS - 1)
        py = (VF_FIELD_DEG - y) / (2 * VF_FIELD_DEG) * (VF_CANVAS - 1)
        sig_px = sig / VF_DEG_PER_PX
        acc = np.zeros((VF_CANVAS, VF_CANVAS), np.float64)
        wsum = np.zeros((VF_CANVAS, VF_CANVAS), np.float64)
        gy, gx = np.mgrid[0:VF_CANVAS, 0:VF_CANVAS]
        for i in idx:
            s = sig_px[i]
            r = int(np.ceil(3 * s))
            cx, cy = px[i], py[i]
            x0, x1 = max(0, int(cx - r)), min(VF_CANVAS, int(cx + r) + 1)
            y0, y1 = max(0, int(cy - r)), min(VF_CANVAS, int(cy + r) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            ww = gx[y0:y1, x0:x1] - cx
            hh = gy[y0:y1, x0:x1] - cy
            g = np.exp(-(ww * ww + hh * hh) / (2 * s * s))
            acc[y0:y1, x0:x1] += activation[i] * g
            wsum[y0:y1, x0:x1] += g
        from scipy.ndimage import gaussian_filter
        img = np.where(wsum > 1e-8, acc / np.maximum(wsum, 1e-8), 0.0)
        img = gaussian_filter(img, VF_SMOOTH_PX)
        return img, wsum

    def _visual_field_readout(self, verts, run_id):
        """From the time-mean of verts, splat the Benson V1/V2/V3 vertices into a
        512² visual-field heatmap PNG (→ signed URL) and compute a small
        visualFieldMap summary (per-area mean + foveal/peripheral split).
        Returns (recon_image_url, visual_field_map)."""
        import io
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        atlas = self._load_benson_atlas()
        act = np.abs(verts).mean(axis=0).astype(np.float64)   # (20484,) time-mean magnitude
        mask = atlas["mask"]
        img, wsum = self._vf_render(act, atlas, mask)
        support = wsum > 1e-6

        # Per-pixel eccentricity → foveal / peripheral split (port of Paper D).
        gy, gx = np.mgrid[0:VF_CANVAS, 0:VF_CANVAS]
        xd = (gx / (VF_CANVAS - 1)) * (2 * VF_FIELD_DEG) - VF_FIELD_DEG
        yd = VF_FIELD_DEG - (gy / (VF_CANVAS - 1)) * (2 * VF_FIELD_DEG)
        ecc_px = np.sqrt(xd * xd + yd * yd)
        in_field = ecc_px <= VF_FIELD_DEG
        fov = (ecc_px < 5.0) & support
        peri = (ecc_px >= 5.0) & in_field & support

        def _safe_mean(arr, m):
            return float(arr[m].mean()) if m.any() else 0.0

        # Per-area V1/V2/V3 mean (vertex-space, restricted to the masked area).
        per_area = {}
        for v, nm in [(1, "V1"), (2, "V2"), (3, "V3")]:
            am = mask & (atlas["varea"] == v)
            per_area[nm] = round(_safe_mean(act, am), 6)

        visual_field_map = {
            "perArea": per_area,
            "foveal": round(_safe_mean(img, fov), 6),
            "peripheral": round(_safe_mean(img, peri), 6),
            "fieldMean": round(_safe_mean(img, support), 6),
            "canvasPx": VF_CANVAS,
            "fieldDeg": VF_FIELD_DEG,
        }

        # Render the heatmap PNG (RdBu_r over the V1-V3 support, NaN elsewhere).
        disp = np.where(support, img, np.nan)
        vlim = float(np.nanmax(np.abs(disp))) if np.isfinite(disp).any() else 1.0
        vlim = vlim or 1.0
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.imshow(disp, origin="upper", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                  extent=[0, VF_CANVAS, VF_CANVAS, 0])
        th = np.linspace(0, 2 * np.pi, 200)
        for d in (5, 10, 20):
            rpx = d / (2 * VF_FIELD_DEG) * (VF_CANVAS - 1)
            ax.plot(VF_CANVAS / 2 + rpx * np.cos(th),
                    VF_CANVAS / 2 + rpx * np.sin(th), color="k", lw=0.7, alpha=0.45)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Decoded visual-field response (V1/V2/V3)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        url = self._store_image(buf.getvalue(), run_id, suffix="field")
        return url, visual_field_map

    def _build_artifact(self, capability, modality, networks, duration_sec,
                        verts=None, media_url=None, run_id=None, include_fmri=False,
                        visual_used=None) -> dict:
        vals = [n["value"] for n in networks] or [0.0]
        mean_abs = sum(vals) / len(vals)
        score = max(0, min(100, round(min(1.0, mean_abs / 0.2) * 90)))
        top = sorted(networks, key=lambda n: n["value"], reverse=True)[:3]
        hi = max(vals) or 1.0
        top_regions = [{"label": t["plain"], "value": round(t["value"] / hi, 3)} for t in top]
        headline = f"Strongest response in {top[0]['plain'].lower()}" if top else "Brain response map"
        summary = (
            "Mary's decoded brain response is led by "
            + ", ".join(t["plain"].lower() for t in top[:2])
            + ". A directional read of how the brain responds, not a clinical scan."
        )
        artifact = {
            "capability": capability, "modality": modality,
            "headline": headline, "summary": summary, "score": score,
            "networks": networks, "topRegions": top_regions,
            "durationSec": duration_sec, "mediaUrl": media_url,
        }
        # VISIBILITY: did the SlowFast visual stream actually feed this prediction?
        # For video modality this MUST be true for a real visual read — if it's
        # false, the run silently degraded to audio-only (the exact failure that
        # collapsed two different IG reels to r=0.99). None for non-video modalities.
        if visual_used is not None:
            artifact["visual_used"] = bool(visual_used)
        # Enrich with the per-second 10-KPI translation when we have the raw
        # vertex timecourse (every real run does). Keeps networks/score intact.
        if verts is not None:
            try:
                artifact.update(self._build_kpi_block(verts, duration_sec))
                # Unify the headline score with the calibrated verdict score so
                # the hero number == the verdict number (no more 51-vs-56 split),
                # and so the top-level score also reflects RELATIVE performance.
                vs = artifact.get("verdictSummary") or {}
                if isinstance(vs.get("score"), (int, float)):
                    artifact["score"] = int(round(vs["score"]))
            except Exception as e:  # never let enrichment kill a good prediction
                print(f"[mary-serve] kpi-block enrichment failed: {e!r}")
        # Optional raw fMRI persist — only when explicitly requested (keeps the
        # default artifact/payload small). Stores the (T_TR, 20484) verts as a
        # compressed .npz to kairo-uploads/fmri/<run_id>.npz and records the
        # storage path on the artifact. Additive + guarded.
        if include_fmri and verts is not None and run_id:
            fmri_path = self._store_fmri(verts, run_id)
            if fmri_path:
                artifact["fmri_path"] = fmri_path
        # Neuro interpretation (neurosignal) — the cited construct/metric/buy-sell
        # layer computed from Mary's Yeo-7 activations. Additive + guarded: a missing
        # dep or any failure never breaks a run; existing artifact fields are untouched.
        # Requires `neurosignal` in the Modal image (see INTEGRATION_neurosignal.md).
        try:
            from neurosignal.inputs.model_networks import analyze_model_networks
            neuro = analyze_model_networks(
                {n["network"]: n["value"] for n in networks},
                source="Mary encoder",
                modalities=[modality],
            )
            artifact["neuro"] = neuro.to_dict()
        except Exception as e:
            print(f"[mary-serve] neurosignal interpretation skipped: {e!r}")
        return artifact

    # — Audio → feature streams (whisper + beats [+ qwen_ctx if speech]) —
    AUDIO_MAX_SEC = 60  # bound cost: only featurize the first 60 s

    def _audio_features(self, wav_path: str, streams=("whisper", "beats", "qwen_ctx")) -> tuple[dict, float]:
        """Decode + featurize an audio/video file → ({stream: (1,T,D) cuda}, dur_sec).

        `streams` selects which to build (analysis mode):
          full       → whisper + beats + qwen_ctx (sound + speech + narrative)
          audio      → whisper + beats             (acoustic only)
          transcript → qwen_ctx                    (words / narrative only)
        qwen_ctx needs speech; if forced alignment finds none it's dropped (and
        for transcript-only that surfaces a clear error). Streams trimmed to the
        common min T, mirroring MaryEngine.assemble_features."""
        import numpy as np
        import torch
        _setup_path()
        from extractors.audio import load_audio_16k_mono
        from extractors.qwen_ctx import forced_align

        audio = load_audio_16k_mono(wav_path)
        if audio.size == 0:
            raise ValueError("no decodable audio track in input")
        max_n = int(self.AUDIO_MAX_SEC * 16000)
        if audio.shape[0] > max_n:
            audio = audio[:max_n]
        duration_sec = round(audio.shape[0] / 16000.0, 1)

        feats_np: dict[str, np.ndarray] = {}
        if "whisper" in streams:
            feats_np["whisper"] = self.whisper.featurize(audio)   # (T,1280)
        if "beats" in streams:
            feats_np["beats"] = self.beats.featurize(audio)       # (T,768)
        # Forced alignment → real word timings. No speech (music/silence) → skip
        # qwen_ctx (acoustic streams carry it). Transcript-only with no speech errors.
        if "qwen_ctx" in streams:
            try:
                words = forced_align(wav_path)
            except Exception as e:
                print(f"[mary-serve] forced_align failed ({e!r}); proceeding without qwen_ctx")
                words = []
            if words:
                feats_np["qwen_ctx"] = self.qwen.encode_words(words)  # (T,4096)

        if not feats_np:
            raise ValueError("no usable streams for this analysis mode (no speech found for a transcript analysis?)")

        T = min(a.shape[0] for a in feats_np.values())
        if T < 2:
            raise ValueError(f"audio yielded too few timesteps (T={T})")
        features = {
            s: torch.from_numpy(a[:T]).float()[None].cuda()
            for s, a in feats_np.items()
        }
        return features, duration_sec

    # — Video → SlowFast visual stream (separate container) —
    VISUAL_MAX_SEC = 60  # bound cost/transfer: featurize the first 60 s

    def _slowfast_extractor_cls(self):
        """Return the SlowFast @app.cls bound to THIS server's Modal app.

        CRITICAL: a Modal `@app.cls` can only be `.remote()`-called from within the
        app it is registered on. `SlowFastExtractor` is registered on the live
        `mary-serve` app (below). A SEPARATE deployed app (e.g. `mary-serve-api`,
        which subclasses this base) does NOT have it registered, so calling it from
        there raises `ExecutionError: Function has not been hydrated ... the App it
        is defined on is not running` — and the visual stream silently falls back to
        audio-only for EVERY job. Each app must therefore override this to return its
        OWN SlowFast class. The default is the live app's extractor."""
        return SlowFastExtractor

    def _slowfast_features(self, video_path: str):
        """Trim+downscale the video, run SlowFast in its own container, return a
        (T_2Hz, 2304) float32 numpy array — or None if unavailable/too short."""
        import io
        import subprocess
        import tempfile
        import numpy as np

        # Trim to the first VISUAL_MAX_SEC, drop audio, downscale to the network's
        # 256px input, normalize fps → a tiny file that's cheap to ship + decode.
        # If the source container/codec is one decord can't read, this ffmpeg pass
        # ALSO re-muxes/transcodes it to a clean 16fps H.264-decodable mp4, so an
        # odd IG/social container can't starve the visual stream of frames.
        trimmed = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
               "-t", str(self.VISUAL_MAX_SEC), "-i", video_path,
               "-an", "-r", "16", "-vf", "scale=256:256", trimmed]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            trimmed = video_path  # fallback: ship the original file
        with open(trimmed, "rb") as f:
            video_bytes = f.read()
        raw = self._slowfast_extractor_cls()().featurize.remote(
            video_bytes, max_sec=self.VISUAL_MAX_SEC)
        if not raw:
            return None
        return np.load(io.BytesIO(raw)).astype(np.float32)

    @modal.method()
    def probe_audio(self, wav_path_on_volume: str, max_sec: int = 12) -> dict:
        """Audio file (path on the /data volume) → full MaryArtifact, no DB.
        For smoke-testing the audio compute path end-to-end."""
        import shutil
        import subprocess
        import tempfile

        # Trim to a short slice on a local temp wav so the smoke is fast/cheap.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-t", str(int(max_sec)),
               "-i", wav_path_on_volume, "-ac", "1", "-ar", "16000", tmp]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            shutil.copy(wav_path_on_volume, tmp)  # fallback: full file
        features, duration_sec = self._audio_features(tmp)
        verts = self.engine.encode(features, subject_idx=0)
        networks = self._reduce_networks(verts)
        artifact = self._build_artifact("brain_map", "audio", networks, duration_sec, verts=verts)
        artifact["streams"] = sorted(features.keys())
        return artifact

    @modal.method()
    def probe_video(self, input_url: str) -> dict:
        """Direct/signed media URL → full MaryArtifact, no DB. Mirrors run_job's
        video compute path (audio streams + slowfast visual + un-flatten) so the
        KPI baseline can be recomputed from REAL reels through production math."""
        import shutil, tempfile, urllib.parse, urllib.request
        import torch
        _setup_path()
        UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
        suffix = Path(urllib.parse.urlparse(input_url).path).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
        req = urllib.request.Request(input_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        features, duration_sec = self._audio_features(tmp, streams=("whisper", "beats", "qwen_ctx"))
        visual_used = False
        try:
            sf = self._slowfast_features(tmp)
            if sf is not None and sf.shape[0] >= 2:
                T = min(sf.shape[0], min(f.shape[1] for f in features.values()))
                features = {s: f[:, :T] for s, f in features.items()}
                features["slowfast"] = torch.from_numpy(sf[:T]).float()[None].cuda()
                visual_used = True
                print(f"[mary-serve] probe_video visual stream (slowfast) added: T={T}")
        except Exception as e:
            print(f"[mary-serve] probe_video slowfast failed ({e!r}); audio-only")
        verts = self.engine.encode(features, subject_idx=0)
        networks = self._reduce_networks(verts)
        return self._build_artifact("brain_map", "video", networks, duration_sec,
                                    verts=verts, visual_used=visual_used)

    @modal.method()
    def probe(self, text: str) -> dict:
        """Text → full MaryArtifact, no DB. For smoke-testing the compute path."""
        import torch
        _setup_path()
        from extractors.qwen_ctx import words_from_text
        words = words_from_text(text)
        feats = self.qwen.encode_words(words)
        features = {"qwen_ctx": torch.from_numpy(feats).float()[None].cuda()}
        verts = self.engine.encode(features, subject_idx=0)
        networks = self._reduce_networks(verts)
        return self._build_artifact("brain_map", "text", networks,
                                    round(float(words[-1]["offset"]), 1), verts=verts)

    @modal.method()
    def run_job(self, run_id: str, capability: str, modality: str,
                input_text: str | None = None, input_url: str | None = None,
                analysis_mode: str = "full", include_fmri: bool = False) -> dict:
        import torch
        _setup_path()
        try:
            self._update(run_id, status="extracting", progress=15, started_at=self._now())
            features: dict = {}
            duration_sec = None
            media_url = None  # set once the clip is downloaded → playable in the verdict
            # For video: did the visual (SlowFast) stream actually feed the model?
            # Stays False if the clip is audio-only, too short, or the extractor
            # failed — so an audio-only-degraded video run is detectable, not hidden.
            visual_used = False

            if modality == "text":
                from extractors.qwen_ctx import words_from_text
                words = words_from_text(input_text or "")
                if len(words) < 2:
                    raise ValueError("please provide at least a short sentence of text")
                feats = self.qwen.encode_words(words)  # (T_2Hz, 4096)
                features["qwen_ctx"] = torch.from_numpy(feats).float()[None].cuda()
                duration_sec = round(float(words[-1]["offset"]), 1)
            elif modality in ("audio", "video"):
                # Video is analyzed multimodally: _audio_features decodes the audio
                # track (whisper+beats+qwen_ctx) AND, for full/visual modes, the
                # SlowFast R101 visual stream is added below from the actual frames.
                if not input_url:
                    raise ValueError("audio/video modality requires input_url")
                import glob
                import os
                import re
                import shutil
                import subprocess
                import tempfile
                import urllib.parse
                import urllib.request

                UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
                # Normalize the pasted link. Users routinely paste a bare host
                # (`www.instagram.com/reel/…`, `tiktok.com/…`) with no scheme —
                # urlparse then puts the whole string in `path` (empty netloc), so
                # no platform matches and urlopen dies with "unknown url type".
                # Add a scheme and strip surrounding junk so the platform router +
                # downloaders see a well-formed https URL.
                input_url = (input_url or "").strip().strip('"').strip("'").strip()
                if input_url.startswith("//"):
                    input_url = "https:" + input_url
                elif not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", input_url):
                    input_url = "https://" + input_url.lstrip("/")
                parsed = urllib.parse.urlparse(input_url)
                path_lower = (parsed.path or "").lower()
                # A URL is a DIRECT media download only when it's our own signed
                # upload URL or it points straight at a media file. EVERYTHING
                # else is a web/page link (instagram, tiktok, youtube, loom,
                # vimeo, a blog embed, …) that serves HTML — those must go
                # through the download ladder, never a naive GET (which 404s or
                # grabs the HTML page).
                MEDIA_EXT = (".mp4", ".mov", ".m4a", ".webm", ".mkv", ".wav",
                             ".mp3", ".aac", ".ogg", ".flac", ".avi")
                is_direct_media = (
                    "/storage/v1/object/" in input_url          # supabase signed upload
                    or path_lower.endswith(MEDIA_EXT)            # direct media file
                )

                tmp = None
                if is_direct_media:
                    # Direct/signed URL (e.g. an uploaded file) — download as-is.
                    # If the host blocks the bare GET (403/404/timeout), fall back
                    # to the ladder (yt-dlp downloads direct URLs too, with its own
                    # headers) so a host-side block never fails an otherwise-valid
                    # media link.
                    try:
                        suffix = Path(path_lower).suffix or ".bin"
                        cand = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
                        req = urllib.request.Request(input_url, headers={"User-Agent": UA})
                        with urllib.request.urlopen(req, timeout=120) as r, open(cand, "wb") as f:
                            shutil.copyfileobj(r, f)
                        tmp = cand
                    except Exception as direct_err:
                        print(f"[mary-serve] direct download failed ({direct_err!r}); trying ladder")
                        _setup_path()
                        from extractors.rapidapi import download_via_ladder
                        tmp = download_via_ladder(input_url, tempfile.mkdtemp())
                else:
                    # Any web/page link → the SAME robust ladder the live pipeline
                    # uses: RapidAPI Auto-Download (generic, covers most sites) →
                    # platform-specific tiers → yt-dlp (which supports Loom, Vimeo,
                    # and hundreds more). Raises a clear, user-facing message if the
                    # link is genuinely unfetchable (private / login-gated / dead).
                    _setup_path()
                    from extractors.rapidapi import download_via_ladder
                    tmp = download_via_ladder(input_url, tempfile.mkdtemp())
                stream_sets = {
                    "audio": ("whisper", "beats"),          # acoustic only
                    "transcript": ("qwen_ctx",),            # words / narrative only
                    "full": ("whisper", "beats", "qwen_ctx"),
                    "visual": ("whisper", "beats", "qwen_ctx"),  # audio streams + slowfast (below)
                }
                features, duration_sec = self._audio_features(
                    tmp, streams=stream_sets.get(analysis_mode, stream_sets["full"]),
                )
                # Visual stream: SlowFast R101 on the actual video frames (its own
                # container). The slowfast stream-encoder weights are trained and
                # loaded, they were simply never fed — feeding them is what makes a
                # video's VISUALS (not just its audio track) drive the prediction,
                # and what makes two visually-different clips score differently.
                # 'full' and 'visual' modes include it; 'audio'/'transcript' don't.
                if analysis_mode in ("full", "visual"):
                    try:
                        self._update(run_id, status="extracting", progress=35)
                        sf = self._slowfast_features(tmp)  # (T_sf, 2304) float32 | None
                        if sf is not None and sf.shape[0] >= 2:
                            # Align all streams to a common T (every stream is 2 Hz).
                            T = min(sf.shape[0], min(f.shape[1] for f in features.values()))
                            features = {s: f[:, :T] for s, f in features.items()}
                            features["slowfast"] = (
                                torch.from_numpy(sf[:T]).float()[None].cuda()
                            )
                            visual_used = True
                            print(f"[mary-serve] visual stream (slowfast) added: T={T}")
                        else:
                            print("[mary-serve] no slowfast features (short/failed); audio-only")
                    except Exception as e:
                        print(f"[mary-serve] slowfast extraction failed ({e!r}); audio-only")
                # Persist the downloaded clip so the verdict can play + scrub it.
                media_url = self._store_playable(tmp, run_id)
            else:
                raise ValueError(f"modality '{modality}' not enabled in this serving slice")

            self._update(run_id, status="predicting", progress=60)
            verts = self.engine.encode(features, subject_idx=0)  # (T_TR, 20484)
            networks = self._reduce_networks(verts)
            # Base artifact — networks/score/topRegions + the 10-KPI block are
            # SHARED across all capabilities (the brain panel + summary always
            # work). `capability` then selects an additional read-out off the
            # SAME verts. The default (brain_map) branch is byte-identical to the
            # prior behavior; new read-outs only ADD fields.
            artifact = self._build_artifact(capability, modality, networks, duration_sec,
                                            verts=verts, media_url=media_url,
                                            run_id=run_id, include_fmri=include_fmri,
                                            visual_used=(visual_used if modality == "video" else None))

            if capability == "visual_field":
                # Paper D — lightweight CPU Gaussian-splat of the early-visual
                # (V1/V2/V3) vertices into a 512² visual-field heatmap. A failure
                # here marks the run errored (clear message) rather than silently
                # returning a bare brain_map under a visual_field label.
                try:
                    self._update(run_id, status="predicting", progress=80)
                    recon_url, vf_map = self._visual_field_readout(verts, run_id)
                    top = sorted(networks, key=lambda n: n["value"], reverse=True)[:2]
                    lead = ", ".join(t["plain"].lower() for t in top) if top else "early vision"
                    artifact["reconImageUrl"] = recon_url
                    artifact["visualFieldMap"] = vf_map
                    artifact["headline"] = "Decoded early-visual field response"
                    artifact["summary"] = (
                        "A directional read of how Mary's predicted early-visual "
                        f"(V1/V2/V3) response distributes across the visual field, led by {lead}. "
                        "This is a retinotopic projection of the decoded brain response, "
                        "not a literal image of what the eyes saw."
                    )
                except Exception as e:
                    raise RuntimeError(f"visual_field read-out failed: {e}") from e

            elif capability == "cross_modal_image":
                # DEFERRED for this deploy: MindEye2 weights + the fsaverage5->NSD
                # adapter (pca_adapter.npz) are not staged yet. Fail cleanly with a
                # friendly message rather than cold-starting the heavy container.
                raise RuntimeError(
                    "\"What the brain pictures\" isn't enabled yet — coming soon.")
                # Paper C — HEAVY MindEye2 decode in its own A100 container. A
                # controlled negative at this scale: framed as an evocative
                # impression, never a literal reconstruction. Missing weights /
                # adapter fail this capability cleanly (the decode container
                # returns ok=False) → run errored, no server crash.
                try:
                    import io as _io
                    import numpy as np
                    self._update(run_id, status="predicting", progress=75)
                    buf = _io.BytesIO()
                    np.save(buf, verts.astype(np.float32))
                    res = CrossModalDecoder().decode.remote(buf.getvalue())
                    if not isinstance(res, dict) or not res.get("ok"):
                        err = (res or {}).get("error", "unknown decoder error") \
                            if isinstance(res, dict) else "decoder returned no result"
                        raise RuntimeError(err)
                    import base64 as _b64
                    img_bytes = _b64.b64decode(res["image_b64"])
                    recon_url = self._store_image(img_bytes, run_id, suffix="crossmodal")
                    artifact["reconImageUrl"] = recon_url
                    artifact["reconCaption"] = res.get("caption")
                    artifact["headline"] = "What the brain pictures (neural impression)"
                    artifact["summary"] = (
                        "An evocative neural impression decoded from Mary's predicted "
                        "brain response via a frozen image decoder. This is NOT a literal "
                        "reconstruction — at this scale it is a controlled, impressionistic "
                        "read of the decoded signal, not a photograph of what was seen."
                    )
                except Exception as e:
                    raise RuntimeError(f"cross_modal_image read-out failed: {e}") from e

            self._update(run_id, status="complete", progress=100,
                         result=artifact, duration_sec=duration_sec, completed_at=self._now())
            return {"ok": True, "run_id": run_id, "score": artifact["score"]}
        except Exception as e:
            self._update(run_id, status="error", error=(str(e) or repr(e) or "analysis failed"),
                         error_step="run_job", completed_at=self._now())
            print(f"[mary-serve] run_job error {run_id}: {e!r}")
            return {"ok": False, "run_id": run_id, "error": str(e)}


@app.cls(
    image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=SECRETS,
    # A/B PARALLELISM: keep one container warm, but allow Modal to scale to a
    # second GPU container so the two runs of an A/B comparison execute
    # concurrently instead of the second waiting ~60s behind the first. Modal's
    # default concurrency is 1 input per container and each run_job saturates the
    # A100, so a 2-run compare fans out to 2 containers. Tradeoff: the second
    # container is a cold-ish spin-up on the first compare after scaledown;
    # min_containers=1 keeps at least one always warm for the common solo path.
    min_containers=1, max_containers=2, scaledown_window=300, timeout=20 * 60,
)
class MaryServer(_MaryServerBase):
    """Live UI-facing serving class — config UNCHANGED. All logic is inherited
    from _MaryServerBase; this thin subclass exists only so the WS-A API app can
    reuse the same logic on its own pool without altering this deploy."""
    pass


@app.function(image=image, secrets=SECRETS, timeout=60)
@modal.fastapi_endpoint(method="POST")
def submit(payload: dict) -> dict:
    """Express → Modal trigger. Verifies the shared secret, spawns the GPU job,
    returns the modal call id immediately (the job writes back to mary_runs)."""
    from fastapi import HTTPException

    expected = os.environ.get("MARY_PIPELINE_SECRET")
    if not expected or payload.get("auth_token") != expected:
        raise HTTPException(status_code=401, detail="bad auth_token")
    run_id = payload.get("run_id")
    capability = payload.get("capability", "brain_map")
    modality = payload.get("modality", "text")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id required")
    call = MaryServer().run_job.spawn(
        run_id=run_id, capability=capability, modality=modality,
        input_text=payload.get("input_text"), input_url=payload.get("input_url"),
        analysis_mode=payload.get("analysis_mode", "full"),
        include_fmri=bool(payload.get("include_fmri", False)),
    )
    return {"modal_call_id": call.object_id}


@app.local_entrypoint()
def smoke(text: str = "The waves crashed against the rocks as the sun set over the quiet harbor."):
    """End-to-end text smoke test against a synthetic run id (no DB write needed
    to read the return value)."""
    import uuid
    run_id = "mary_run_smoke_" + uuid.uuid4().hex[:8]
    res = MaryServer().probe.remote(text=text)
    print("smoke score:", res.get("score"), "duration_sec:", res.get("duration_sec"))
    print("networks (plain → value):")
    for n in res.get("networks", []):
        print(f"  {n['plain']:<22} {n['value']:.4f}")
    print("headline:", res.get("headline"))


@app.local_entrypoint()
def build_reel_baseline(urls_file: str = ""):
    """Recompute KPI_BASELINE from REAL reels through the deployed video path.
    --urls-file: a text file of direct/signed media URLs (one per line). Prints a
    paste-ready KPI_BASELINE (mean,std across reels) for serve.py + MaryPillars.tsx."""
    import json
    import statistics as st
    assert urls_file, "pass --urls-file (one media URL per line)"
    with open(urls_file) as f:
        urls = [u.strip() for u in f if u.strip()]
    srv = MaryServer()
    per_kpi = {k: [] for k in KPI_NAMES}
    n_ok = 0
    for i, u in enumerate(urls):
        try:
            art = srv.probe_video.remote(input_url=u)
            sm = art.get("kpiSummary") or {}
            if not sm:
                print(f"[{i}] no kpiSummary; skip"); continue
            for k in KPI_NAMES:
                per_kpi[k].append(float(sm.get(k, {}).get("mean", 0.0)))
            n_ok += 1
            print(f"[{i}] ok score={art.get('score')} dur={art.get('durationSec')}")
        except Exception as e:
            print(f"[{i}] FAILED {e!r}")
    print(f"\n# Reel-centered KPI_BASELINE from {n_ok} real reels")
    print("KPI_BASELINE = {")
    out = {}
    for k in KPI_NAMES:
        vals = per_kpi[k]
        m = round(st.mean(vals), 5) if vals else 0.0
        s = round(st.pstdev(vals), 5) if len(vals) > 1 else 0.0
        out[k] = (m, s)
        print(f'    "{k}":{" " * max(1, 36 - len(k))}({m}, {s}),')
    print("}")
    print("REEL_BASELINE_JSON " + json.dumps(out))


@app.function(image=image, volumes=VOLUMES, timeout=60)
def _first_story_wav() -> str:
    """Pick the first ds002345 speech story *_audio.wav on the volume."""
    stim = Path("/data/raw/ds002345/stimuli")
    wavs = sorted(stim.glob("*_audio.wav"))
    if not wavs:
        raise FileNotFoundError(f"no *_audio.wav under {stim}")
    return str(wavs[0])


@app.local_entrypoint()
def audio_smoke(max_sec: int = 12):
    """End-to-end AUDIO smoke test (no web/DB layer): take a short slice of a real
    ds002345 speech story off the volume and run the full audio compute path."""
    import json
    wav = _first_story_wav.remote()
    print(f"audio_smoke using: {wav}  (first {max_sec}s)", flush=True)
    res = MaryServer().probe_audio.remote(wav_path_on_volume=wav, max_sec=max_sec)
    print("audio smoke score:", res.get("score"), "duration_sec:", res.get("durationSec"), flush=True)
    print("streams used:", res.get("streams"), flush=True)
    print("networks (plain → value):", flush=True)
    for n in res.get("networks", []):
        print(f"  {n['plain']:<22} {n['value']:.4f}", flush=True)
    print("headline:", res.get("headline"), flush=True)
    # Single machine-parseable line (survives pipe buffering / log truncation).
    print("AUDIO_SMOKE_JSON " + json.dumps({
        "score": res.get("score"), "durationSec": res.get("durationSec"),
        "streams": res.get("streams"), "headline": res.get("headline"),
        "networks": {n["plain"]: round(n["value"], 4) for n in res.get("networks", [])},
    }), flush=True)
