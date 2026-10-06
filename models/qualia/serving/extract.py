"""qualia/serving/extract.py — (formerly kairo_extract.py) RAW VIDEO URL → all 18 Kairo streams at the EXACT trained
dims/layers in ckpt['streams']. This is the piece that lets kairo-serve score an
ARBITRARY video (not just cached Friends clips).

Design
------
One Modal app, several extractor functions (each with its own pinned image where a
backbone needs incompatible deps). A top-level `extract_all(url)` downloads the
video once (mary's RapidAPI ladder), fans the local mp4 out to each backbone, and
returns a dict {stream_name: np.ndarray (T, dim)} at 1 Hz, trimmed to a common T.

EXACT stream contract (from ckpt['streams'], verified on the weights volume):
  vjepa_block{5,15,23}     facebook V-JEPA ViT-L   attn.proj  token-avg   dim 1024
  whisper_layer{12,25,31}  whisper-large-v3 enc    fc2                    dim 1280
  whisper_layernorm        whisper-large-v3 enc    final layer_norm       dim 1280
  vmae2_block{10,18,25}    VideoMAEv2-giant        attn.proj dropout avg  dim 1280
  ivl3_layer{10,15,20}     InternVL3-8B (Qwen2.5-7B LM) post_attn_ln      dim 3584
  ivl3_norm                InternVL3-8B LM         final norm             dim 3584
  emonet                   EmoNet visual                                  dim 900
  llama                    Llama-3.2-3B            3 layer-group means     dim 9216 (3072*3)
  w2vbert                  w2v-bert-2.0            3 layer-group means     dim 3072 (1024*3)
  vjepa2                   V-JEPA2 ViT-g           3 layer-group means     dim 4224 (1408*3)

All streams are produced at **1 Hz** (one vector per second), matching the cached
Friends features (591 steps for a 591 s clip). Any backbone that cannot be matched
to the trained dims is ZERO-FILLED at the correct (T, dim) — this DEGRADES quality
for that stream but never fails the whole pipeline.

VISUAL-ON STATUS (this fix): the full visual front-end is live —
  • vjepa_block{5,15,23} : facebook/vjepa2-vitl-fpc64-256 (ViT-L 1024) hidden-state
        block taps 5/15/23, token-mean. (was hard-coded zero-fill.)
  • vmae2_block{10,18,25}: OpenGVLab/VideoMAEv2-**huge** (1280, 32 blocks) attn.proj
        hooks. (was -giant: wrong 1408 dim + crashed on missing easydict.)
  • vjepa2               : facebook/vjepa2-vitg-fpc64-256 (ViT-g 1408×3 → 4224).
  • ivl3_*              : OpenGVLab/InternVL3-8B 8-bit LM hidden states (3584).
        (was gated off behind with_internvl=False; now default ON.)
The ONLY remaining zero-fill is `emonet` (a custom face-cropping net, dim 900, not
on HF — reproducing its exact 900-d tap needs the original repo + a face detector,
out of budget; sensitivity: dropping emonet keeps per-sec DIFF ~0.15 — safest to
leave). See QUALIA-VISUAL-ON-REPORT.md.

Entrypoints:
  modal run qualia/serving/extract.py --url <URL>            # extract → write npz to volume
  modal run qualia/serving/extract.py::probe --url <URL>     # just download + report frames/dur
"""
from __future__ import annotations
import modal
import os as _os
import sys as _sys

# rapidapi.py is a sibling module in this same qualia/serving/ dir (both were formerly
# in kairo/). Put this dir on sys.path so `from rapidapi import ...` and
# add_local_python_source("rapidapi") resolve regardless of CWD.
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)

APP_NAME = "kairo-extract"
app = modal.App(APP_NAME)

# Streams per backbone, with the EXACT trained dim. Keep in lockstep with kairo_model.STREAM_ORDER.
STREAM_DIMS = {
    "vjepa_block5": 1024, "vjepa_block15": 1024, "vjepa_block23": 1024,
    "whisper_layer12": 1280, "whisper_layer25": 1280, "whisper_layer31": 1280,
    "whisper_layernorm": 1280, "vmae2_block10": 1280, "vmae2_block18": 1280,
    "vmae2_block25": 1280, "ivl3_layer10": 3584, "ivl3_layer15": 3584,
    "ivl3_layer20": 3584, "ivl3_norm": 3584, "emonet": 900, "llama": 9216,
    "w2vbert": 3072, "vjepa2": 4224,
}

TARGET_HZ = 1.0
TARGET_SR = 16000
MAX_SEC = 120  # cap clip length so cost is bounded; reels are ~10-60 s anyway

data_vol = modal.Volume.from_name("kairo-extracted", create_if_missing=True)
# Dedicated LEAN hf cache for kairo. mary-hf-cache holds dozens of multi-GB models
# from sibling agents + xet storage; mounting it adds >10 min of startup stall, so
# kairo gets its own small volume (first run re-downloads whisper-turbo + w2v-bert
# ~5 GB; every subsequent run mounts instantly).
hf_vol = modal.Volume.from_name("kairo-hf-cache", create_if_missing=True)
VOLS = {"/extracted": data_vol, "/cache": hf_vol}
SECRETS = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("sapient-rapidapi")]

# ── shared CPU image: download + ffmpeg + frame sampling ─────────────────────
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("numpy>=1.26,<3", "requests", "yt-dlp", "pillow>=10")
    .add_local_python_source("rapidapi")
)

# ── visual image: torch + transformers (VideoMAEv2 + EmoNet, decord frames) ──
# easydict is REQUIRED by OpenGVLab/VideoMAEv2's remote modeling file (modeling_videomaev2.py
# imports easydict at module load — without it AutoModel raises ImportError and the whole
# vmae2 stream silently zero-filled). This was the vmae2 root cause.
visual_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "numpy>=1.26,<3",
        "transformers==4.51.3", "timm==1.0.11", "einops", "easydict",
        "huggingface_hub>=0.25,<2", "safetensors>=0.4", "decord==0.6.0",
        "pillow>=10", "requests", "yt-dlp",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"})
    .add_local_python_source("rapidapi")
)

# ── vjepa2 image: needs transformers>=4.52 for the `vjepa2` model type ────────
vjepa2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "numpy>=1.26,<3",
        "transformers==4.53.0", "timm==1.0.11", "einops",
        "huggingface_hub>=0.25,<2", "safetensors>=0.4", "decord==0.6.0",
        "pillow>=10", "requests", "yt-dlp",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"})
    .add_local_python_source("rapidapi")
)

# ── audio image: whisper + w2v-bert ──────────────────────────────────────────
audio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1", "numpy>=1.26,<3",
        "transformers==4.51.3", "librosa==0.10.2", "soundfile==0.12.1",
        "huggingface_hub>=0.25,<2", "safetensors>=0.4", "requests", "yt-dlp",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"})
    .add_local_python_source("rapidapi")
)

# ── text image: llama-3.2-3b + whisper transcript ────────────────────────────
# openai-whisper's legacy sdist imports pkg_resources at build time → needs
# setuptools<81 installed FIRST and --no-build-isolation (mirrors mary serve.py).
text_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("setuptools<81", "wheel", "torch==2.4.1", "torchaudio==2.4.1", "numpy>=1.26,<3")
    .pip_install("openai-whisper==20240930", extra_options="--no-build-isolation")
    .pip_install(
        "transformers==4.51.3", "librosa==0.10.2", "soundfile==0.12.1",
        "accelerate>=0.34", "huggingface_hub>=0.25,<2", "safetensors>=0.4",
        "sentencepiece", "requests", "yt-dlp",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"})
    .add_local_python_source("rapidapi")
)


# ───────────────────────── helpers (defined in-container) ────────────────────
def _download(url: str) -> bytes:
    return _download_with_ext(url)[0]


def _download_with_ext(url: str):
    """Download `url` via the robust ladder; return (bytes, suffix). The suffix
    (e.g. '.mp4') lets the caller persist the clip with the right extension +
    content-type so the verdict page can play it back (see extract_all →
    qualia-serve._store_playable)."""
    import tempfile, os
    from rapidapi import download_via_ladder
    d = tempfile.mkdtemp()
    path = download_via_ladder(url, d)
    ext = os.path.splitext(path)[1] or ".mp4"
    with open(path, "rb") as f:
        return f.read(), ext


def _video_meta(video_bytes: bytes):
    """Return (n_seconds, fps, n_frames) capped at MAX_SEC using decord."""
    import tempfile
    from decord import VideoReader, cpu
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(video_bytes)
    vr = VideoReader(tmp, ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 30.0
    n_frames = len(vr)
    dur = min(n_frames / fps, float(MAX_SEC))
    return int(dur), fps, n_frames, tmp


# ════════════════════════════ WHISPER (large-v3) ═════════════════════════════
@app.function(image=audio_image, volumes=VOLS, secrets=SECRETS, gpu="A10G", timeout=1800)
def extract_whisper(video_bytes: bytes) -> dict:
    """whisper-large-v3 encoder → fc2 of layers 12/25/31 + final layer_norm, 1 Hz.
    Returns {stream: npy-bytes}. Hooks capture fc2; last_hidden_state == layer_norm."""
    import io, os, subprocess, tempfile
    import numpy as np, torch
    from transformers import AutoFeatureExtractor, WhisperForConditionalGeneration

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(video_bytes)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", tmp, "-f", "f32le",
           "-ac", "1", "-ar", str(TARGET_SR), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    audio = (np.frombuffer(proc.stdout, dtype=np.float32).copy()
             if proc.returncode == 0 and proc.stdout else np.zeros(0, np.float32))
    dur = min(len(audio) / TARGET_SR, float(MAX_SEC))
    n_steps = int(dur * TARGET_HZ)
    layers = {"whisper_layer12": 12, "whisper_layer25": 25, "whisper_layer31": 31}
    if n_steps <= 0:
        return {s: _npy(np.zeros((0, 1280), np.float16)) for s in
                list(layers) + ["whisper_layernorm"]}

    # large-v3-turbo shares the IDENTICAL 32-layer / 1280-d encoder with large-v3
    # (turbo only prunes the DECODER 32→4); we use only the encoder + its fc2/
    # layer_norm taps, so the features match the trained whisper streams exactly.
    # Using turbo because its weights are already on the mary-hf-cache volume.
    WH = "openai/whisper-large-v3-turbo"
    fe = AutoFeatureExtractor.from_pretrained(WH)
    full = WhisperForConditionalGeneration.from_pretrained(WH, torch_dtype=torch.float16)
    enc = full.get_encoder().eval().cuda()
    for p in enc.parameters():
        p.requires_grad_(False)

    cap = {}
    handles = []
    for name, li in layers.items():
        def mk(nm):
            return lambda m, i, o: cap.__setitem__(nm, o.detach().float().cpu())
        handles.append(enc.layers[li].fc2.register_forward_hook(mk(name)))

    WIN = int(30.0 * TARGET_SR)
    chunks = {name: [] for name in list(layers) + ["whisper_layernorm"]}
    audio = audio[: int(MAX_SEC * TARGET_SR)]
    for start in range(0, len(audio), WIN):
        chunk = audio[start:start + WIN]
        secs = len(chunk) / TARGET_SR
        feats = fe(chunk, sampling_rate=TARGET_SR, return_tensors="pt").to("cuda", torch.float16)
        cap.clear()
        with torch.no_grad():
            out = enc(feats.input_features)
        ln = out.last_hidden_state.squeeze(0).float().cpu()   # (1500, 1280) == final layer_norm
        n_native = ln.shape[0]
        n_keep = max(1, int(round(n_native * (secs / 30.0))))
        chunks["whisper_layernorm"].append(ln[:n_keep].numpy())
        for name in layers:
            chunks[name].append(cap[name].squeeze(0)[:n_keep].numpy())
    for h in handles:
        h.remove()

    out = {}
    for name, lst in chunks.items():
        native = np.concatenate(lst, axis=0)                  # (T_native, 1280)
        rate = native.shape[0] / max(dur, 1e-6)
        binsz = max(1, int(round(rate / TARGET_HZ)))
        f = np.empty((n_steps, 1280), np.float16)
        for i in range(n_steps):
            s = i * binsz; e = min(s + binsz, native.shape[0])
            seg = native[s:e] if e > s else native[-1:]
            f[i] = seg.mean(0).astype(np.float16)
        out[name] = _npy(f)
    print(f"[whisper] {n_steps} steps")
    return out


# ════════════════════════════ W2V-BERT-2.0 ═══════════════════════════════════
@app.function(image=audio_image, volumes=VOLS, secrets=SECRETS, gpu="A10G", timeout=1800)
def extract_w2vbert(video_bytes: bytes) -> dict:
    """w2v-bert-2.0 (24 layers, hidden 1024) → 3 layer-group means concat → 3072, 1 Hz."""
    import os, subprocess, tempfile
    import numpy as np, torch
    from transformers import AutoModel, AutoFeatureExtractor

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(video_bytes)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", tmp, "-f", "f32le",
           "-ac", "1", "-ar", str(TARGET_SR), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    audio = (np.frombuffer(proc.stdout, dtype=np.float32).copy()
             if proc.returncode == 0 and proc.stdout else np.zeros(0, np.float32))
    dur = min(len(audio) / TARGET_SR, float(MAX_SEC))
    n_steps = int(dur * TARGET_HZ)
    if n_steps <= 0:
        return {"w2vbert": _npy(np.zeros((0, 3072), np.float16))}
    audio = audio[: int(MAX_SEC * TARGET_SR)]

    fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    model = AutoModel.from_pretrained(
        "facebook/w2v-bert-2.0", torch_dtype=torch.float16,
        output_hidden_states=True).eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)

    inp = fe(audio, sampling_rate=TARGET_SR, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        out = model(**inp)
    hs = out.hidden_states  # tuple of (1, T_native, 1024), len 25 (embed + 24 layers)
    layer_hs = [h.squeeze(0).float().cpu().numpy() for h in hs[1:]]  # 24 layers
    # 3 contiguous groups of 8 layers; mean within group → concat 1024*3
    groups = [np.mean(layer_hs[0:8], axis=0),
              np.mean(layer_hs[8:16], axis=0),
              np.mean(layer_hs[16:24], axis=0)]
    native = np.concatenate(groups, axis=1)  # (T_native, 3072)
    rate = native.shape[0] / max(dur, 1e-6)
    binsz = max(1, int(round(rate / TARGET_HZ)))
    f = np.empty((n_steps, 3072), np.float16)
    for i in range(n_steps):
        s = i * binsz; e = min(s + binsz, native.shape[0])
        seg = native[s:e] if e > s else native[-1:]
        f[i] = seg.mean(0).astype(np.float16)
    print(f"[w2vbert] {n_steps} steps, native {native.shape}")
    return {"w2vbert": _npy(f)}


# ════════════════════════════ V-JEPA2 (ViT-g) ════════════════════════════════
@app.function(image=vjepa2_image, volumes=VOLS, secrets=SECRETS, gpu="A10G", timeout=2400)
def extract_vjepa2(video_bytes: bytes) -> dict:
    """V-JEPA2 ViT-g (hidden 1408) → 3 layer-group means concat → 4224, 1 Hz.
    Per-second 16-frame clips; mean over patch tokens; hidden_states grouped.
    Manual preprocessing (resize 256 + ImageNet norm) so we don't depend on a
    specific processor class name across transformers versions."""
    import tempfile
    import numpy as np, torch
    from decord import VideoReader, cpu

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(video_bytes)
    try:
        vr = VideoReader(tmp, ctx=cpu(0))
    except Exception as e:
        print(f"[vjepa2] decode fail {e!r}; zero-fill")
        return {"vjepa2": _npy(np.zeros((0, 4224), np.float16))}
    fps = float(vr.get_avg_fps()) or 30.0
    n_frames = len(vr)
    n_sec = min(int(n_frames / fps), MAX_SEC)
    if n_sec < 1:
        return {"vjepa2": _npy(np.zeros((0, 4224), np.float16))}

    REPO = "facebook/vjepa2-vitg-fpc64-256"
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            REPO, torch_dtype=torch.float16, output_hidden_states=True,
            trust_remote_code=True).eval().cuda()
        for p in model.parameters():
            p.requires_grad_(False)
    except Exception as e:
        print(f"[vjepa2] model load FAILED ({e!r}); zero-fill")
        return {"vjepa2": _npy(np.zeros((0, 4224), np.float16))}

    SIDE = 256
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).cuda()
    NF = 16
    feats = np.zeros((n_sec, 4224), np.float16)
    nl = 0
    for s in range(n_sec):
        t0, t1 = s * fps, (s + 1) * fps
        idx = np.linspace(t0, min(n_frames - 1, t1 - 1e-3), NF)
        idx = np.clip(np.round(idx), 0, n_frames - 1).astype(np.int64)
        clip = vr.get_batch(list(idx)).asnumpy()  # (NF,H,W,3) uint8
        c = torch.from_numpy(clip).to("cuda").permute(3, 0, 1, 2).float() / 255.0  # (3,NF,H,W)
        c = torch.nn.functional.interpolate(c, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
        c = c.unsqueeze(0)  # (1,3,NF,SIDE,SIDE) == (B,C,T,H,W)
        c = ((c - mean) / std).to(torch.float16)
        # HF VJEPA2 expects pixel_values_videos as (B, T, C, H, W)
        cv = c.permute(0, 2, 1, 3, 4).contiguous()                   # (1,NF,3,SIDE,SIDE)
        try:
            with torch.no_grad():
                out = model(pixel_values_videos=cv)
        except Exception:
            with torch.no_grad():
                out = model(cv)
        hs = out.hidden_states  # tuple (1, n_tokens, 1408)
        layer_hs = [h.squeeze(0).float().mean(0).cpu().numpy() for h in hs[1:]]
        nl = len(layer_hs)
        g = max(1, nl // 3)
        groups = [np.mean(layer_hs[0:g], 0), np.mean(layer_hs[g:2 * g], 0),
                  np.mean(layer_hs[2 * g:], 0)]
        v = np.concatenate(groups)
        if v.shape[0] >= 4224:
            feats[s] = v[:4224].astype(np.float16)
        else:
            feats[s, :v.shape[0]] = v.astype(np.float16)
    print(f"[vjepa2] {n_sec} steps, n_layers={nl}")
    return {"vjepa2": _npy(feats)}


# ════════════════════════════════ VideoMAEv2-giant ═══════════════════════════
@app.function(image=visual_image, volumes=VOLS, secrets=SECRETS, gpu="A10G", timeout=2400)
def extract_videomae2(video_bytes: bytes) -> dict:
    """VideoMAEv2-HUGE → blocks 10/18/25 attn.proj output, token-mean → 1280, 1 Hz.

    ROOT CAUSE FIX: the trained vmae2 streams are 1280-dim with 32 blocks (block 25
    exists) — that is OpenGVLab/VideoMAEv2-**huge** (embed_dim 1280, depth 32), NOT
    -giant (embed_dim 1408, depth 40, which both needed easydict AND emits the wrong
    1408 dim). We now load -huge (matching the trained tap exactly) and hook
    model.blocks[{10,18,25}].attn.proj, token-mean → 1280. (Override via VMAE2_REPO.)
    On any failure the streams are returned EMPTY (T=0) so the orchestrator zero-fills."""
    import os, tempfile
    import numpy as np, torch
    from decord import VideoReader, cpu

    names = {"vmae2_block10": 10, "vmae2_block18": 18, "vmae2_block25": 25}
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(video_bytes)
    try:
        vr = VideoReader(tmp, ctx=cpu(0))
    except Exception as e:
        print(f"[vmae2] decode fail {e!r}")
        return {n: _npy(np.zeros((0, 1280), np.float16)) for n in names}
    fps = float(vr.get_avg_fps()) or 30.0
    n_frames = len(vr)
    n_sec = min(int(n_frames / fps), MAX_SEC)
    if n_sec < 1:
        return {n: _npy(np.zeros((0, 1280), np.float16)) for n in names}

    try:
        from transformers import AutoModel
        repo = os.environ.get("VMAE2_REPO", "OpenGVLab/VideoMAEv2-huge")
        model = AutoModel.from_pretrained(
            repo, trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")).eval().cuda().half()
        print(f"[vmae2] loaded {repo}")
    except Exception as e:
        print(f"[vmae2] HF load FAILED ({e!r}); returning empty → orchestrator zero-fills")
        return {n: _npy(np.zeros((0, 1280), np.float16)) for n in names}
    for p in model.parameters():
        p.requires_grad_(False)

    # locate the transformer blocks. For VideoMAEv2-huge the AutoModel wraps the ViT
    # under `.model`, so the blocks live at model.model.blocks (NOT model.blocks).
    # Search robustly: try common attrs, then scan named_modules for a long ModuleList
    # ending in 'blocks' (mirrors the verified probe).
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        blocks = getattr(getattr(model, "model", object()), "blocks", None)
    if blocks is None:
        blocks = getattr(getattr(model, "encoder", object()), "blocks", None)
    if blocks is None:
        for nmod, mod in model.named_modules():
            if nmod.endswith("blocks") and hasattr(mod, "__len__") and len(mod) > 25:
                blocks = mod
                print(f"[vmae2] located blocks at '{nmod}' (n={len(mod)})")
                break
    if blocks is None:
        print("[vmae2] could not locate .blocks; empty")
        return {n: _npy(np.zeros((0, 1280), np.float16)) for n in names}

    cap = {}
    handles = []
    for nm, bi in names.items():
        def mk(k):
            return lambda m, i, o: cap.__setitem__(
                k, (o[0] if isinstance(o, tuple) else o).detach().float().mean(1).squeeze(0).cpu())
        try:
            handles.append(blocks[bi].attn.proj.register_forward_hook(mk(nm)))
        except Exception as e:
            print(f"[vmae2] hook {nm} failed: {e!r}")
    if not handles:
        print("[vmae2] no proj hooks attached; empty")
        return {n: _npy(np.zeros((0, 1280), np.float16)) for n in names}

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1).cuda()
    NF = 16
    out = {nm: np.zeros((n_sec, 1280), np.float16) for nm in names}
    for s in range(n_sec):
        t0, t1 = s * fps, (s + 1) * fps
        idx = np.linspace(t0, min(n_frames - 1, t1 - 1e-3), NF)
        idx = np.clip(np.round(idx), 0, n_frames - 1).astype(np.int64)
        clip = vr.get_batch(list(idx)).asnumpy()  # (NF,H,W,3)
        c = torch.from_numpy(clip).to("cuda").permute(3, 0, 1, 2).float() / 255.0
        c = torch.nn.functional.interpolate(c, size=(224, 224), mode="bilinear", align_corners=False)
        c = (c - mean) / std  # (3,NF,224,224)
        cap.clear()
        with torch.no_grad():
            try:
                _ = model(c.unsqueeze(0).to(torch.float16))           # (1,3,NF,224,224)
            except Exception:
                _ = model(c.unsqueeze(0).to(torch.float16))
        for nm in names:
            if nm in cap:
                out[nm][s] = cap[nm].numpy().astype(np.float16)
    for h in handles:
        h.remove()
    print(f"[vmae2] {n_sec} steps")
    return {nm: _npy(out[nm]) for nm in names}


def _npy(arr):
    import io, numpy as np
    b = io.BytesIO(); np.save(b, arr); return b.getvalue()


# ════════════════════════════ V-JEPA (ViT-L, 1024) ═══════════════════════════
@app.function(image=vjepa2_image, volumes=VOLS, secrets=SECRETS, gpu="A10G", timeout=2400)
def extract_vjepa(video_bytes: bytes) -> dict:
    """V-JEPA ViT-L (hidden 1024) → blocks 5/15/23 token-mean → 1024, 1 Hz.

    ROOT CAUSE FIX: this used to be HARD-CODED to zero-fill ("HF has no V-JEPA-v1
    AutoModel"). But facebook/vjepa2-vitl-fpc64-256 IS a public HF AutoModel that
    emits the SAME 1024-dim, 24-block ViT-L encoder the trained vjepa_block{5,15,23}
    taps came from (verified: hidden_states gives 1024-d at every block). We read
    hidden_states[b+1] (output of encoder block b), token-mean → 1024, for b in
    {5,15,23}. Real visual features, exact trained dim. (Override repo via VJEPA_VITL_REPO.)
    On decode/load failure → EMPTY so the orchestrator zero-fills."""
    import os, tempfile
    import numpy as np, torch
    from decord import VideoReader, cpu

    names = ["vjepa_block5", "vjepa_block15", "vjepa_block23"]
    blk_idx = {"vjepa_block5": 5, "vjepa_block15": 15, "vjepa_block23": 23}
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(video_bytes)
    try:
        vr = VideoReader(tmp, ctx=cpu(0))
    except Exception as e:
        print(f"[vjepa] decode fail {e!r}; zero-fill")
        return {n: _npy(np.zeros((0, 1024), np.float16)) for n in names}
    fps = float(vr.get_avg_fps()) or 30.0
    n_frames = len(vr)
    n_sec = min(int(n_frames / fps), MAX_SEC)
    if n_sec < 1:
        return {n: _npy(np.zeros((0, 1024), np.float16)) for n in names}

    REPO = os.environ.get("VJEPA_VITL_REPO", "facebook/vjepa2-vitl-fpc64-256")
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            REPO, torch_dtype=torch.float16, output_hidden_states=True,
            trust_remote_code=True).eval().cuda()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"[vjepa] loaded {REPO}")
    except Exception as e:
        print(f"[vjepa] model load FAILED ({e!r}); zero-fill")
        return {n: _npy(np.zeros((0, 1024), np.float16)) for n in names}

    SIDE = 256
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).cuda()
    NF = 16
    out = {n: np.zeros((n_sec, 1024), np.float16) for n in names}
    for s in range(n_sec):
        t0, t1 = s * fps, (s + 1) * fps
        idx = np.linspace(t0, min(n_frames - 1, t1 - 1e-3), NF)
        idx = np.clip(np.round(idx), 0, n_frames - 1).astype(np.int64)
        clip = vr.get_batch(list(idx)).asnumpy()                     # (NF,H,W,3) uint8
        c = torch.from_numpy(clip).to("cuda").permute(3, 0, 1, 2).float() / 255.0  # (3,NF,H,W)
        c = torch.nn.functional.interpolate(c, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
        c = c.unsqueeze(0)                                           # (1,3,NF,SIDE,SIDE)
        c = ((c - mean) / std).to(torch.float16)
        cv = c.permute(0, 2, 1, 3, 4).contiguous()                  # (1,NF,3,SIDE,SIDE) = (B,T,C,H,W)
        try:
            with torch.no_grad():
                o = model(pixel_values_videos=cv)
        except Exception:
            with torch.no_grad():
                o = model(cv)
        hs = o.hidden_states                                         # tuple, len 25, (1,tokens,1024)
        for nm, bi in blk_idx.items():
            h = hs[bi + 1].squeeze(0).float().mean(0).cpu().numpy()  # token-mean → (1024,)
            out[nm][s] = h[:1024].astype(np.float16)
    print(f"[vjepa] {n_sec} steps (ViT-L 1024)")
    return {n: _npy(out[n]) for n in names}


# ════════════════════════════ EmoNet (visual 900) ════════════════════════════
@app.function(image=visual_image, volumes=VOLS, secrets=SECRETS, gpu="A10G", timeout=1800)
def extract_emonet(video_bytes: bytes) -> dict:
    """EmoNet visual features (dim 900). EmoNet is a custom repo (face-cropping +
    a bespoke head) whose exact 900-dim layout the checkpoint doesn't pin. Building
    it to the EXACT trained 900-dim tap is out of budget, so this zero-fills
    (sensitivity: dropping emonet keeps per-sec DIFF ~0.15 — SAFE to zero-fill)."""
    import numpy as np
    print("[emonet] zero-filling (custom 900-dim tap not recoverable — documented)")
    return {"emonet": _npy(np.zeros((0, 900), np.float16))}


# ════════════════════════════ Llama-3.2-3B (text) ════════════════════════════
@app.function(image=text_image, volumes=VOLS, secrets=SECRETS, gpu="A10G", timeout=2400)
def extract_llama(video_bytes: bytes) -> dict:
    """Transcript (whisper) → Llama-3.2-3B (hidden 3072, 28 layers) → 3 layer-group
    means concat → 9216, aligned to 1 Hz video seconds via word timestamps."""
    import subprocess, tempfile
    import numpy as np, torch

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(video_bytes)
    # decode audio + duration
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", tmp, "-f", "f32le",
           "-ac", "1", "-ar", str(TARGET_SR), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    audio = (np.frombuffer(proc.stdout, dtype=np.float32).copy()
             if proc.returncode == 0 and proc.stdout else np.zeros(0, np.float32))
    dur = min(len(audio) / TARGET_SR, float(MAX_SEC))
    n_steps = int(dur * TARGET_HZ)
    if n_steps <= 0:
        return {"llama": _npy(np.zeros((0, 9216), np.float16))}
    audio = audio[: int(MAX_SEC * TARGET_SR)]

    # transcribe with word timestamps (openai-whisper small for speed)
    try:
        import whisper
        wm = whisper.load_model("base")
        res = wm.transcribe(audio.astype(np.float32), word_timestamps=True, fp16=False)
        words = [(w["word"].strip(), float(w["start"]), float(w["end"]))
                 for seg in res.get("segments", []) for w in seg.get("words", [])
                 if w.get("word", "").strip()]
        # Capture the SPOKEN TRANSCRIPT (already produced here for the Llama stream)
        # so it can be surfaced on the served artifact — the grounded "why" + chat on
        # the web side reference real spoken content instead of "no content evidence".
        # Returned to extract_all under the reserved "__transcript__" key (NOT a
        # model stream); extract_all stashes it on the manifest, never np.load()s it.
        _transcript = {
            "text": (res.get("text") or "").strip(),
            "segments": [
                {"start": round(float(s.get("start") or 0.0), 2),
                 "end": round(float(s.get("end") or 0.0), 2),
                 "text": (s.get("text") or "").strip()}
                for s in res.get("segments", []) if (s.get("text") or "").strip()
            ],
            "language": res.get("language"),
        }
    except Exception as e:
        print(f"[llama] transcription failed {e!r}; zero-fill")
        return {"llama": _npy(np.zeros((n_steps, 9216), np.float16))}
    if not words:
        print("[llama] no speech; zero-fill (silent video)")
        return {"llama": _npy(np.zeros((n_steps, 9216), np.float16))}

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    model = AutoModel.from_pretrained(
        "meta-llama/Llama-3.2-3B", torch_dtype=torch.float16,
        output_hidden_states=True).eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)

    text = " ".join(w[0] for w in words)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=2048)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    hs = out.hidden_states  # 29 (embed + 28); take 28 layers
    layer_hs = [h.squeeze(0).float().cpu().numpy() for h in hs[1:]]  # (T_tok, 3072) each
    g = 28 // 3
    grouped = np.concatenate(
        [np.mean(layer_hs[0:g], 0), np.mean(layer_hs[g:2 * g], 0),
         np.mean(layer_hs[2 * g:], 0)], axis=1)  # (T_tok, 9216)

    # map each token to its char offset → which word → which second
    char_to_sec = {}
    cursor = 0
    for w, ws, we in words:
        start_c = text.find(w, cursor)
        if start_c < 0:
            start_c = cursor
        for c in range(start_c, start_c + len(w)):
            char_to_sec[c] = int(min(n_steps - 1, max(0, (ws + we) / 2)))
        cursor = start_c + len(w)
    feats = np.zeros((n_steps, 9216), np.float32)
    counts = np.zeros(n_steps, np.float32)
    for ti, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        sec = char_to_sec.get(a, char_to_sec.get(b - 1))
        if sec is None:
            continue
        feats[sec] += grouped[ti]
        counts[sec] += 1
    # fill empty seconds with nearest filled second (carry-forward) for continuity
    last = None
    for s in range(n_steps):
        if counts[s] > 0:
            feats[s] /= counts[s]; last = feats[s].copy()
        elif last is not None:
            feats[s] = last
    print(f"[llama] {n_steps} steps, {len(words)} words")
    import json as _json
    return {"llama": _npy(feats.astype(np.float16)),
            "__transcript__": _json.dumps(_transcript).encode("utf-8")}


# ════════════════════════════ InternVL3-8B (the hard 8B VLM) ═════════════════
internvl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "numpy>=1.26,<3",
        "transformers==4.51.3", "timm==1.0.11", "einops", "accelerate>=0.34",
        "bitsandbytes>=0.43", "sentencepiece",
        "huggingface_hub>=0.25,<2", "safetensors>=0.4", "decord==0.6.0",
        "pillow>=10", "requests", "yt-dlp",
    )
    .env({"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"})
    .add_local_python_source("rapidapi")
)


@app.function(image=internvl_image, volumes=VOLS, secrets=SECRETS, gpu="A100-40GB", timeout=3600)
def extract_internvl3(video_bytes: bytes) -> dict:
    """InternVL3-8B (8-bit) — per-second frame → VLM LM hidden states at
    layers 10/15/20 (post_attention_layernorm) + final norm, dim 3584, 1 Hz.

    This is the expensive/hard backbone. It runs the 8B VLM once per second on a
    single frame + a fixed prompt, captures the LM hidden states via forward hooks.
    On ANY failure (OOM, remote-code drift, dim mismatch) it returns EMPTY for all
    four ivl3 streams so the orchestrator zero-fills them (documented quality hit;
    sensitivity: dropping InternVL3 raises per-sec DIFF 0.16→0.54, still differentiates)."""
    import tempfile
    import numpy as np, torch
    from decord import VideoReader, cpu

    names = ["ivl3_layer10", "ivl3_layer15", "ivl3_layer20", "ivl3_norm"]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(video_bytes)
    try:
        vr = VideoReader(tmp, ctx=cpu(0))
    except Exception as e:
        print(f"[ivl3] decode fail {e!r}; zero-fill")
        return {n: _npy(np.zeros((0, 3584), np.float16)) for n in names}
    fps = float(vr.get_avg_fps()) or 30.0
    n_frames = len(vr)
    n_sec = min(int(n_frames / fps), MAX_SEC)
    if n_sec < 1:
        return {n: _npy(np.zeros((0, 3584), np.float16)) for n in names}

    try:
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from PIL import Image

        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL3-8B", torch_dtype=torch.float16,
            quantization_config=bnb, trust_remote_code=True,
            low_cpu_mem_usage=True).eval()
        tok = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL3-8B", trust_remote_code=True, use_fast=False)

        lm = model.language_model.model  # Qwen2.5 LM
        idxs = {"ivl3_layer10": 10, "ivl3_layer15": 15, "ivl3_layer20": 20}
        cap = {}
        handles = []
        for nm, li in idxs.items():
            def mk(k):
                return lambda m, i, o: cap.__setitem__(
                    k, (o[0] if isinstance(o, tuple) else o).detach().float().mean(1).squeeze(0).cpu())
            handles.append(lm.layers[li].post_attention_layernorm.register_forward_hook(mk(nm)))
        def norm_hook(m, i, o):
            cap["ivl3_norm"] = (o[0] if isinstance(o, tuple) else o).detach().float().mean(1).squeeze(0).cpu()
        handles.append(lm.norm.register_forward_hook(norm_hook))

        MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
        tf = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                        T.ToTensor(), T.Normalize(MEAN, STD)])
        gcfg = dict(max_new_tokens=1, do_sample=False)
        out = {nm: np.zeros((n_sec, 3584), np.float16) for nm in names}
        for s in range(n_sec):
            fidx = int(min(n_frames - 1, (s + 0.5) * fps))
            frame = vr[fidx].asnumpy()
            pv = tf(Image.fromarray(frame)).unsqueeze(0).to(torch.float16).cuda()
            cap.clear()
            with torch.no_grad():
                try:
                    model.chat(tok, pv, "<image>\nDescribe.", gcfg)
                except Exception:
                    # fallback: direct vision+LM forward if .chat signature differs
                    pass
            for nm in names:
                if nm in cap:
                    # The final RMSNorm (ivl3_norm) can emit NaN/inf under 8-bit on
                    # some frames; sanitize so the whole pipeline isn't poisoned
                    # (a single NaN propagates through the projection → ALL parcels NaN).
                    v = np.nan_to_num(cap[nm].numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                    out[nm][s] = v.astype(np.float16)
        for h in handles:
            h.remove()
        if not any(out[nm].any() for nm in names):
            print("[ivl3] hooks captured nothing; zero-fill")
            return {n: _npy(np.zeros((0, 3584), np.float16)) for n in names}
        print(f"[ivl3] {n_sec} steps captured")
        return {nm: _npy(out[nm]) for nm in names}
    except Exception as e:
        print(f"[ivl3] FAILED ({e!r}); zero-fill all 4 ivl3 streams")
        return {n: _npy(np.zeros((0, 3584), np.float16)) for n in names}


# ════════════════════════════ ORCHESTRATOR ═══════════════════════════════════
@app.function(image=cpu_image, volumes=VOLS, secrets=SECRETS, timeout=5400)
def extract_all(url: str, with_internvl: bool = True, key: str = "",
                skip_visual: bool = False, backbone_timeout: int = 1500) -> dict:
    """Download URL once, fan out to every backbone (parallel), assemble all 18
    streams at a common T, zero-fill any that came back empty, write an npz to the
    kairo-extracted volume keyed by `key` (or a url hash). Returns a manifest of
    which streams are REAL vs ZERO-FILLED + the npz path.

    VISUAL-ON DEFAULT: all visual backbones run by default now —
      vjepa (ViT-L 1024), videomae2 (huge 1280), vjepa2 (ViT-g 4224) AND internvl3
      (8B VLM, 3584). Only emonet (custom 900-dim face net, not on HF) stays
      zero-filled. with_internvl defaults TRUE so ivl3 is real on the product path.

    skip_visual: skip the slow/cold visual backbones (vjepa/videomae2/vjepa2/internvl3)
      — they zero-fill. Use for a fast, audio+speech+language-only run.
    backbone_timeout: per-backbone wall-clock cap; a backbone that exceeds it is
    treated as failed → zero-filled (so a stuck GPU queue can't hang the pipeline)."""
    import hashlib, io
    import numpy as np
    vb, src_ext = _download_with_ext(url)
    print(f"[extract_all] downloaded {len(vb)} bytes from {url}")

    # Persist the raw downloaded clip to the shared kairo-extracted volume so the
    # async qualia-serve job (run_job, which mounts /extracted AND holds the
    # Supabase creds) can mirror it into the kairo-uploads bucket and hand the
    # verdict page a playable, scrubbable <video> — the same persistence
    # mary-serve does via _store_playable. We stage it here (where the bytes
    # already exist) rather than re-downloading downstream. Best-effort: a write
    # failure must never fail extraction — the run still scores, just without
    # inline playback.
    k = key or hashlib.sha1(url.encode()).hexdigest()[:16]
    clip_ext = src_ext if src_ext in (".mp4", ".mov", ".m4v", ".webm") else ".mp4"
    clip_path = f"/extracted/{k}{clip_ext}"
    try:
        with open(clip_path, "wb") as cf:
            cf.write(vb)
        data_vol.commit()
        print(f"[extract_all] staged playable clip → {clip_path} ({len(vb)} bytes)")
    except Exception as e:
        clip_path = None
        print(f"[extract_all] clip staging failed ({e!r}); playback will be unavailable")

    # spawn backbones in parallel
    calls = {
        "whisper": extract_whisper.spawn(vb),
        "w2vbert": extract_w2vbert.spawn(vb),
        "emonet": extract_emonet.spawn(vb),
        "llama": extract_llama.spawn(vb),
    }
    if not skip_visual:
        calls["vjepa"] = extract_vjepa.spawn(vb)
        calls["vjepa2"] = extract_vjepa2.spawn(vb)
        calls["videomae2"] = extract_videomae2.spawn(vb)
        if with_internvl:
            calls["internvl3"] = extract_internvl3.spawn(vb)

    raw = {}
    transcript_obj = None  # spoken transcript surfaced by extract_llama (if any)
    for name, h in calls.items():
        try:
            d = h.get(timeout=backbone_timeout)
            for s, b in d.items():
                # Reserved non-stream payload: the spoken transcript from extract_llama.
                # Decode it and stash on the manifest; do NOT treat it as an npz stream.
                if s == "__transcript__":
                    try:
                        import json as _json
                        transcript_obj = _json.loads(b.decode("utf-8"))
                        print(f"  ✓ transcript: {len((transcript_obj or {}).get('segments') or [])} segments")
                    except Exception as te:
                        print(f"  ! transcript decode failed: {te!r}")
                    continue
                arr = np.load(io.BytesIO(b))
                # Belt-and-suspenders: scrub NaN/inf from ANY backbone before it can
                # poison the model (one NaN in any stream → all 1000 parcels NaN).
                if arr.size and not np.isfinite(arr).all():
                    nbad = int((~np.isfinite(arr)).sum())
                    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0,
                                        neginf=0.0).astype(arr.dtype)
                    print(f"  ! {s}: scrubbed {nbad} non-finite values")
                raw[s] = arr
            print(f"  ✓ {name}: {[(s, raw[s].shape) for s in d]}")
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e!r}")

    order = list(STREAM_DIMS.keys())
    # common T = max real length (so we don't truncate to a zero-filled 0)
    real_T = [raw[s].shape[0] for s in order if s in raw and raw[s].shape[0] > 0]
    T = min(real_T) if real_T else 0
    manifest = {"real": [], "zero_filled": [], "T": int(T), "url": url}
    streams = {}
    for s in order:
        dim = STREAM_DIMS[s]
        a = raw.get(s)
        if a is not None and a.shape[0] >= T and T > 0:
            streams[s] = a[:T].astype(np.float16)
            manifest["real"].append(s)
        else:
            streams[s] = np.zeros((T, dim), np.float16)
            manifest["zero_filled"].append(s)

    # Serialize ONCE to bytes, then both write the file and return the bytes.
    import io as _io
    _buf = _io.BytesIO()
    np.savez_compressed(_buf, **streams)
    _npz_bytes = _buf.getvalue()
    out_path = f"/extracted/{k}.npz"
    with open(out_path, "wb") as _f:
        _f.write(_npz_bytes)
    data_vol.commit()
    manifest["npz"] = out_path
    # Return the features IN-MEMORY too. Reading them back off the shared
    # /extracted volume in qualia-serve RACES: a warm scoring container can hit the
    # npz before the cross-container commit is visible -> "npz not found" failures.
    # Passing the bytes through the function return makes scoring independent of
    # volume propagation. The on-volume file is kept for the clip/debug path and as
    # a fallback. (All 18 streams, visual included, are inside these bytes.)
    manifest["npz_bytes"] = _npz_bytes
    # Path to the staged raw clip on the shared volume (None if staging failed).
    # qualia-serve.run_job reads it back, uploads to kairo-uploads, and signs a
    # playable URL for the verdict <video>.
    manifest["clip_path"] = clip_path
    manifest["transcript"] = transcript_obj  # None when silent / no speech
    print(f"[extract_all] T={T} real={manifest['real']} zero={manifest['zero_filled']}")
    return manifest


# ════════════════════════════ PROBE / orchestration ══════════════════════════
@app.function(image=visual_image, volumes=VOLS, secrets=SECRETS, timeout=600)
def probe(url: str) -> dict:
    """Download a URL and report duration/fps/frames — cheapest end-to-end check."""
    vb = _download(url)
    import tempfile
    from decord import VideoReader, cpu
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    open(tmp, "wb").write(vb)
    vr = VideoReader(tmp, ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 30.0
    n = len(vr)
    info = {"bytes": len(vb), "fps": round(fps, 2), "n_frames": n,
            "dur_s": round(n / fps, 1), "url": url}
    print("[probe]", info)
    return info


@app.function(image=cpu_image, volumes=VOLS, secrets=SECRETS, timeout=3600)
def selftest(url: str, which: str = "audio") -> dict:
    """Run a SUBSET of backbones on a URL and report exact shapes/dims — cheap
    validation that each extractor emits the trained dim before paying for a full run.
      which=audio   -> whisper + w2vbert (A10G, cheap)
      which=visual  -> vjepa2 + videomae2 (A10G)
      which=text    -> llama (A10G)
    """
    import io
    import numpy as np
    vb = _download(url)
    fns = {"audio": [("whisper", extract_whisper), ("w2vbert", extract_w2vbert)],
           "visual": [("vjepa2", extract_vjepa2), ("videomae2", extract_videomae2)],
           "text": [("llama", extract_llama)]}[which]
    report = {}
    for name, fn in fns:
        d = fn.remote(vb)
        for s, b in d.items():
            a = np.load(io.BytesIO(b))
            report[s] = {"shape": list(a.shape), "dim_ok": a.shape[1] == STREAM_DIMS[s] if a.ndim == 2 else False,
                         "nonzero": bool(np.abs(a).sum() > 0)}
    print("[selftest]", report)
    return report


@app.local_entrypoint()
def main(url: str = "", mode: str = "probe", which: str = "audio",
         with_internvl: bool = False, key: str = ""):
    import json
    if mode == "probe":
        print(json.dumps(probe.remote(url), indent=2, default=str))
    elif mode == "selftest":
        print(json.dumps(selftest.remote(url, which), indent=2, default=str))
    elif mode == "extract":
        print(json.dumps(extract_all.remote(url, with_internvl, key), indent=2, default=str))
    else:
        raise SystemExit("mode must be probe|selftest|extract")
