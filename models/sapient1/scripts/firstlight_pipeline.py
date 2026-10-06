"""Self-contained first-light data pipeline for Sapient-1 (CNeuroMod Friends).

Mirrors what Sapient-2 did: build a small, clean (BOLD + V-JEPA2 video +
W2V-BERT audio) subset and write it to the exact paths train.py/dataset.py
expect, so we can launch a first-light H100 training run without the full
hundreds-of-GB CNeuroMod corpus.

Why a single orchestrator instead of the per-step data/*.py scripts:
  * The CNeuroMod Friends BOLD is per-SEGMENT (task-s01e01a / ...b), and the
    stimulus .mkv is also per-segment (friends_s01e01a.mkv). We pair them 1:1
    by the stem "s01e01a" — no episode/segment offset math needed.
  * Feature caches must land at  /data/features/<enc>/cneuromod/<stem>.npy
    (flat, stem-keyed) so data/manifest.py finds them. The general extractors
    mirror the stimulus directory tree, which doesn't match. Here we name
    outputs by stem directly.
  * We only touch the curated stems, so V-JEPA2 (the cost driver) runs on a
    bounded set of clips.

Steps (each a Modal function on the sapient-data volume):
  1. prepare_fmri_remote  — fsaverage5-project + 1 Hz resample the chosen
     segment BOLD runs for the 4 CC0 subjects → /data/fmri/cneuromod/<sub>/<run>.npy
  2. extract_video_remote — V-JEPA2 ViT-g @ 2 Hz over each segment .mkv (shared
     across subjects) → /data/features/vjepa2/cneuromod/<stem>.npy  (T, 1408)
  3. extract_audio_remote — W2V-BERT 2.0 @ 2 Hz over each segment .mkv audio →
     /data/features/w2vbert/cneuromod/<stem>.npy  (T, 1024)

Run each step:
  modal run scripts/firstlight_pipeline.py::prep_fmri
  modal run scripts/firstlight_pipeline.py::video
  modal run scripts/firstlight_pipeline.py::audio
"""

from __future__ import annotations

import modal

APP_NAME = "sapient-1-firstlight"

# Curated first-light subset: Friends season 1, episodes 1-6, both segments
# (a/b), across the 4 CC0 subjects. ~12 stimulus clips, shared across subjects.
SEASON = 1
EPISODES = list(range(1, 7))      # e01..e06
SEGMENTS = ["a", "b"]
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
TR_SECONDS = 1.49

FRIENDS_ROOT = "/data/raw/cneuromod/fmriprep/friends"
STIM_DIR = f"{FRIENDS_ROOT}/sourcedata/friends/stimuli"
FMRI_OUT = "/data/fmri/cneuromod"
VIDEO_OUT = "/data/features/vjepa2/cneuromod"
AUDIO_OUT = "/data/features/w2vbert/cneuromod"

BOLD_SUFFIX = "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"


def _stems() -> list[str]:
    return [f"s{SEASON:02d}e{ep:02d}{seg}" for ep in EPISODES for seg in SEGMENTS]


# ---------- Images (heavy deps live in the image; imports are INSIDE fns) ----------

fmri_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("nibabel>=5.2", "numpy>=1.26,<3", "scipy>=1.13", "nilearn>=0.10.4")
)

video_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        # V-JEPA 2 needs transformers>=4.52 (AutoVideoProcessor + VJEPA2Model +
        # model.get_vision_features). This is the proven sapient-2 stack.
        "torch==2.6.0", "torchvision==0.21.0", "transformers==4.53.0",
        "decord==0.6.0", "numpy>=1.26,<3", "huggingface_hub>=0.30,<2",
        "safetensors>=0.4",
    )
)

audio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "transformers==4.46.0",
        "soundfile==0.12.1", "librosa==0.10.2", "numpy>=1.26,<3",
        "huggingface_hub>=0.25,<2",
    )
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")
app = modal.App(APP_NAME)


# ============================ 1. fMRI prep ============================

@app.function(image=fmri_image, cpu=8.0, memory=32768,
              timeout=12 * 60 * 60, volumes={"/data": volume})
def prepare_fmri_remote(stems: list[str], subjects: list[str]) -> dict:
    from pathlib import Path
    import numpy as np
    import nibabel as nib
    from nilearn import datasets, surface
    from scipy.interpolate import interp1d

    N_VERTICES = 20484
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    out_root = Path(FMRI_OUT)
    out_root.mkdir(parents=True, exist_ok=True)

    def project(nii_path: Path) -> np.ndarray:
        img = nib.load(str(nii_path))
        lh = surface.vol_to_surf(img, fsaverage.pial_left, radius=3.0, kind="auto")
        rh = surface.vol_to_surf(img, fsaverage.pial_right, radius=3.0, kind="auto")
        out = np.concatenate([lh.T, rh.T], axis=1).astype(np.float32)
        if out.shape[1] != N_VERTICES:
            raise RuntimeError(f"got {out.shape[1]} vertices")
        return out

    def to_1hz(arr: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        t_in = np.arange(n) * TR_SECONDS
        t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0)
        f = interp1d(t_in, arr, axis=0, kind="linear", fill_value="extrapolate")
        return f(t_out).astype(np.float32)

    done, skip, missing = 0, 0, []
    for sub in subjects:
        sub_dir = Path(FRIENDS_ROOT) / sub
        out_sub = out_root / sub
        out_sub.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            # find the BOLD run whose task matches this stem
            matches = list(sub_dir.rglob(f"*task-{stem}{BOLD_SUFFIX}"))
            if not matches:
                missing.append(f"{sub}/{stem}")
                continue
            nii = matches[0]
            out = out_sub / f"{sub}_task-{stem}.npy"
            if out.exists():
                skip += 1
                continue
            print(f"  project {sub}/{stem}  ({nii.name})", flush=True)
            arr = to_1hz(project(nii))
            np.save(out, arr)
            done += 1
            volume.commit()
    volume.commit()
    return {"projected": done, "skipped": skip, "missing": missing}


@app.local_entrypoint()
def prep_fmri() -> None:
    print(prepare_fmri_remote.remote(_stems(), SUBJECTS))


# ============================ 2. Video (V-JEPA2) ============================

@app.function(image=video_image, gpu="A100-80GB", timeout=24 * 60 * 60,
              volumes={"/data": volume}, secrets=[hf_secret])
def extract_video_remote(stems: list[str]) -> dict:
    import math, os, time
    from pathlib import Path
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from transformers import AutoVideoProcessor, AutoModel

    MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
    CLIP_SECONDS = 4.0
    FRAMES_PER_CLIP = 64
    RATE_HZ = 2.0
    BATCH_STEPS = 8   # clips per GPU forward (A100-80GB fp16)

    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN",
                          os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    out_dir = Path(VIDEO_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...", flush=True)
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).eval().cuda()
    hidden_dim = int(getattr(model.config, "hidden_size", 1408))
    print(f"  encoder hidden_size = {hidden_dim}", flush=True)

    def find_mkv(stem: str) -> Path | None:
        # stimulus .mkv for s01e01a → s1/friends_s01e01a.mkv (skip annex copies)
        for p in Path(STIM_DIR).rglob(f"friends_{stem}.mkv"):
            if ".git" in str(p):
                continue
            return p
        return None

    results = {}
    for stem in stems:
        out = out_dir / f"{stem}.npy"
        if out.exists():
            results[stem] = "skip"
            continue
        vid = find_mkv(stem)
        if vid is None:
            results[stem] = "no_mkv"
            continue
        print(f"  extract {stem}  ({vid.name})", flush=True)
        vr = VideoReader(str(vid), ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        n_frames = len(vr)
        duration_s = n_frames / fps
        n_steps = int(math.floor(duration_s * RATE_HZ))
        if n_steps <= 0:
            results[stem] = "too_short"
            continue

        feats = np.empty((n_steps, hidden_dim), dtype=np.float16)
        t0 = time.time()
        for base in range(0, n_steps, BATCH_STEPS):
            steps = range(base, min(base + BATCH_STEPS, n_steps))
            clips = []
            for step in steps:
                t_end = (step + 1) / RATE_HZ
                t_start = max(0.0, t_end - CLIP_SECONDS)
                idx = np.linspace(t_start * fps, min(n_frames - 1, t_end * fps),
                                  FRAMES_PER_CLIP).astype(np.int64)
                frames = vr.get_batch(idx).asnumpy()    # (64, H, W, 3) uint8
                # AutoVideoProcessor wants each video as (T, C, H, W).
                clips.append(torch.from_numpy(frames).permute(0, 3, 1, 2))
            inputs = processor(clips, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out_h = model.get_vision_features(**inputs)   # (B, tokens, D)
            pooled = out_h.float().mean(dim=1).cpu().numpy()
            feats[base:base + pooled.shape[0]] = pooled.astype(np.float16)
            if base % (BATCH_STEPS * 20) == 0:
                rate = (base + len(list(steps))) / max(1e-6, time.time() - t0)
                print(f"    {stem} {base}/{n_steps} ({rate:.1f} steps/s)", flush=True)
        np.save(out, feats)
        volume.commit()
        results[stem] = f"ok {feats.shape}"
    volume.commit()
    return results


@app.local_entrypoint()
def video() -> None:
    print(extract_video_remote.remote(_stems()))


# ============================ 3. Audio (W2V-BERT) ============================

@app.function(image=audio_image, gpu="A100-40GB", timeout=24 * 60 * 60,
              volumes={"/data": volume}, secrets=[hf_secret])
def extract_audio_remote(stems: list[str]) -> dict:
    import subprocess, tempfile
    from pathlib import Path
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    MODEL_ID = "facebook/w2v-bert-2.0"
    TARGET_SR = 16000
    RATE_HZ = 2.0
    CHUNK_SECONDS = 30.0

    out_dir = Path(AUDIO_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().cuda()

    def find_mkv(stem: str) -> Path | None:
        for p in Path(STIM_DIR).rglob(f"friends_{stem}.mkv"):
            if ".git" in str(p):
                continue
            return p
        return None

    def load_wav(mkv: Path) -> np.ndarray:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mkv), "-ac", "1", "-ar",
                 str(TARGET_SR), "-vn", tmp.name],
                check=True, capture_output=True,
            )
            wav, _ = sf.read(tmp.name)
        return wav.astype(np.float32)

    results = {}
    for stem in stems:
        out = out_dir / f"{stem}.npy"
        if out.exists():
            results[stem] = "skip"
            continue
        mkv = find_mkv(stem)
        if mkv is None:
            results[stem] = "no_mkv"
            continue
        print(f"  extract {stem}  ({mkv.name})", flush=True)
        wav = load_wav(mkv)
        total_s = len(wav) / TARGET_SR
        n_steps = int(np.floor(total_s * RATE_HZ))
        chunk = int(CHUNK_SECONDS * TARGET_SR)

        step_feats: list[np.ndarray] = []
        for c0 in range(0, len(wav), chunk):
            seg = wav[c0:c0 + chunk]
            if len(seg) < TARGET_SR // 2:
                break
            inputs = fe(seg, sampling_rate=TARGET_SR, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = model(**inputs).last_hidden_state.squeeze(0).float().cpu().numpy()
            # h is ~50 Hz frames; pool to 2 Hz over 0.5 s windows
            seg_s = len(seg) / TARGET_SR
            seg_steps = max(1, int(np.floor(seg_s * RATE_HZ)))
            frames_per_step = max(1, h.shape[0] // seg_steps)
            for s in range(seg_steps):
                a = h[s * frames_per_step:(s + 1) * frames_per_step]
                if a.shape[0] == 0:
                    a = h[-1:]
                step_feats.append(a.mean(axis=0))
        feats = np.asarray(step_feats[:n_steps], dtype=np.float16)
        np.save(out, feats)
        volume.commit()
        results[stem] = f"ok {feats.shape}"
    volume.commit()
    return results


@app.local_entrypoint()
def audio() -> None:
    print(extract_audio_remote.remote(_stems()))
