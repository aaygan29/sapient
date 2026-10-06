"""Final test: did the visual-forced fine-tune raise visual's output contribution,
and do two visually-different / audio-similar videos now diverge?

Compares SERVED checkpoint vs new mary_visual_force_v1 on:
  1. slowfast on-vs-off prediction divergence on real video val windows
     (the original ~0.1% headline number) — for BOTH checkpoints.
  2. Two-video divergence: pick two val video windows with DIFFERENT visual content
     but matched on audio-presence; swap ONLY visual between them and measure how much
     the predicted brain curve changes. Higher = visual drives output more.

Audio-similar / visually-different real reels (DZX0pkERJsE, DZF8MpCJz1y) require the
full serve.py download+extract pipeline; this script uses cached val video features as
the controlled equivalent (true paired visual swap, no audio confound).

Runs on A10G. Read-only on both checkpoints.
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

_REPO = Path(__file__).parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "scipy>=1.13", "pyyaml>=6",
                 "tqdm>=4.66", "transformers==4.46.0", "huggingface_hub>=0.25,<2",
                 "safetensors>=0.4")
    .add_local_dir(str(_REPO), remote_path="/root/maryrepo",
                   ignore=["**/__pycache__", "**/.venv", "**/*.pt", "**/*.npy",
                           "**/.git", "**/logs/**"])
)
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token")]
app = modal.App("mary-test-divergence")

SERVED = "/data/checkpoints/mary_multi_stable_resp_s13/best.pt"
NEW = "/data/checkpoints/mary_visual_force_v1/best.pt"
VISUAL = ["slowfast", "qwen_vl", "got_ocr"]


@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={"/data": volume}, secrets=secrets)
def test() -> dict:
    import os, sys
    os.chdir("/root/maryrepo"); sys.path.insert(0, "/root/maryrepo")
    import numpy as np, torch
    from mary.model import MaryConfig, MaryModel
    from data.dataset import MaryDataset, collate_mary

    device = torch.device("cuda")

    def load(path):
        ck = torch.load(path, map_location=device)
        m = MaryModel(MaryConfig.from_yaml(ck["config"])).to(device).eval()
        m.load_state_dict(ck["state_dict"])
        return m, ck["config"]

    served, cfg = load(SERVED)
    new, _ = load(NEW)

    manifest = "/data/" + cfg.get("manifest_file", "manifest_mary_multi.json")
    val_ds = MaryDataset(manifest, split="val", sequence_length=200)
    items = [val_ds[i] for i in range(len(val_ds)) if "slowfast" in val_ds[i]["features"]]
    batch = collate_mary(items)
    feats = {s: t.to(device) for s, t in batch["features"].items()}
    subj = batch["subject_idx"].to(device)
    B = subj.shape[0]

    def pred(model, fd):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(fd, subj).float().cpu()

    def per_sample_corr(a, b):
        rs = []
        for i in range(a.shape[0]):
            fa = a[i].reshape(-1).clone(); fb = b[i].reshape(-1).clone()
            fa -= fa.mean(); fb -= fb.mean()
            d = (fa.norm() * fb.norm()).clamp(min=1e-8)
            rs.append(float((fa * fb).sum() / d))
        return float(np.mean(rs))

    out = {}
    for name, model in [("served", served), ("visual_force_v1", new)]:
        full = pred(model, feats)
        # slowfast OFF
        no_sf = {s: t for s, t in feats.items() if s != "slowfast"}
        off = pred(model, no_sf)
        r_sf = per_sample_corr(full, off)
        # ALL visual OFF
        no_vis = {s: t for s, t in feats.items() if s not in VISUAL}
        off_v = pred(model, no_vis)
        r_vis = per_sample_corr(full, off_v)
        # Two-video visual swap: roll visual by B/2 (pair different videos), keep audio
        perm = torch.tensor([(i + B // 2) % B for i in range(B)])
        swapped = dict(feats)
        for s in VISUAL:
            if s in swapped:
                swapped[s] = swapped[s][perm].contiguous()
        sw = pred(model, swapped)
        r_swap = per_sample_corr(full, sw)
        out[name] = {
            "slowfast_on_vs_off_corr": r_sf,
            "slowfast_pct_output_change": (1 - r_sf) * 100,
            "all_visual_on_vs_off_corr": r_vis,
            "all_visual_pct_output_change": (1 - r_vis) * 100,
            "two_video_visual_swap_corr": r_swap,
            "two_video_visual_swap_pct_change": (1 - r_swap) * 100,
        }
    out["n_video_windows"] = B
    print(json.dumps(out, indent=2, default=str))
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(test.remote(), indent=2, default=str))
