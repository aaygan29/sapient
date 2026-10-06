"""Decisive test: does the visual CONTENT carry brain signal, or is slowfast just a
'present-stream' placeholder the conjunctive fusion needs?

The ablation test conflates two things: (1) removing slowfast renormalizes the fusion
mean (off-distribution shift) and (2) losing visual information. This script keeps ALL
streams PRESENT (no renormalization) but PERMUTES the slowfast content across video
windows (shuffle which video's frames go with which fMRI target). If visual content
carries real, target-specific brain signal, accuracy should DROP under the shuffle.
If slowfast is just a 'something is here' placeholder, shuffling its content won't
hurt (the fusion only needed *a* present slowfast tensor, not the *right* one).

Same for audio (whisper) as a positive control — audio is known to carry signal, so
shuffling whisper SHOULD hurt.

Runs on the served checkpoint, video windows only. Read-only.
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
app = modal.App("mary-diagnose-swap")
CKPT = "/data/checkpoints/mary_multi_stable_resp_s13/best.pt"


@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={"/data": volume}, secrets=secrets)
def diagnose() -> dict:
    import os, sys
    os.chdir("/root/maryrepo"); sys.path.insert(0, "/root/maryrepo")
    import numpy as np, torch
    from mary.model import MaryConfig, MaryModel
    from mary.metrics import vertex_pearson
    from data.dataset import MaryDataset, collate_mary
    from torch.utils.data import DataLoader

    device = torch.device("cuda")
    ckpt = torch.load(CKPT, map_location=device)
    cfg = ckpt["config"]
    model = MaryModel(MaryConfig.from_yaml(cfg)).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    isc = np.load("/data/noise_ceiling_isc.npy").astype("float32")
    rmask = torch.from_numpy(isc > 0.05)
    key = "vertex_pearson_mean_responsive"

    manifest = "/data/" + cfg.get("manifest_file", "manifest_mary_multi.json")
    val_ds = MaryDataset(manifest, split="val", sequence_length=200)

    # Collect ALL val windows that have slowfast (video windows) as individual items
    # so we can permute streams across them at equal T.
    items = []
    for idx in range(len(val_ds)):
        it = val_ds[idx]
        if "slowfast" in it["features"]:
            items.append(it)
    print(f"video windows (have slowfast): {len(items)}")
    if len(items) < 4:
        return {"error": "too few video windows", "n": len(items)}

    batch = collate_mary(items)
    feats = {s: t.to(device) for s, t in batch["features"].items()}
    subj = batch["subject_idx"].to(device)
    tgt = batch["fmri"]
    B = subj.shape[0]

    def acc(feature_dict):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = model(feature_dict, subj).float().cpu()
        return vertex_pearson(p, tgt, responsive_mask=rmask)[key]

    base = acc(feats)

    # deterministic derangement-ish permutation (roll by half) to avoid identity
    perm = torch.tensor([(i + B // 2) % B for i in range(B)])

    def with_shuffled(stream):
        f = dict(feats)
        if stream in f:
            f[stream] = f[stream][perm].contiguous()
        return f

    res = {"n_video_windows": B, "base_acc_all_streams": base}
    for s in ["slowfast", "qwen_vl", "whisper", "beats", "qwen_ctx", "got_ocr"]:
        if s in feats:
            a = acc(with_shuffled(s))
            res[f"acc_shuffle_{s}"] = a
            res[f"delta_shuffle_{s}"] = a - base
    # also: shuffle slowfast AND qwen_vl AND got_ocr (all visual) together
    f = dict(feats)
    for s in ["slowfast", "qwen_vl", "got_ocr"]:
        if s in f:
            f[s] = f[s][perm].contiguous()
    res["acc_shuffle_ALL_VISUAL"] = acc(f)
    res["delta_shuffle_ALL_VISUAL"] = res["acc_shuffle_ALL_VISUAL"] - base
    # shuffle all audio+text together (positive control)
    f = dict(feats)
    for s in ["whisper", "beats", "qwen_ctx"]:
        if s in f:
            f[s] = f[s][perm].contiguous()
    res["acc_shuffle_ALL_AUDIO_TEXT"] = acc(f)
    res["delta_shuffle_ALL_AUDIO_TEXT"] = res["acc_shuffle_ALL_AUDIO_TEXT"] - base

    # ---- permutation null ----------------------------------------------------
    # The roll-by-half above is ONE permutation, so it yields a point estimate with
    # no error bar. That is fine for a 55%-of-base effect (audio+text) and useless
    # for a ~5% one (visual): without knowing the spread the permutation itself
    # induces, "visual is about zero" is not a claim, it is a single number.
    #
    # Here each condition is re-shuffled under K random permutations. The reported
    # mean_delta estimates the cost of destroying that group's content/target
    # alignment; ci95 is the 2.5/97.5 percentile across permutations. A group whose
    # ci95 straddles 0 carries no detectable target-specific signal at this K.
    #
    # NOTE on interpretation: a near-zero delta is consistent with BOTH "this stream
    # is non-predictive" AND "the model learned to route around it". This test
    # cannot separate those. finetune_visual_force.py is the test that can.
    K = int(os.environ.get("SWAP_NULL_K", "20"))
    g = torch.Generator().manual_seed(13)
    GROUPS = {
        "slowfast": ["slowfast"], "qwen_vl": ["qwen_vl"], "got_ocr": ["got_ocr"],
        "whisper": ["whisper"], "beats": ["beats"], "qwen_ctx": ["qwen_ctx"],
        "ALL_VISUAL": ["slowfast", "qwen_vl", "got_ocr"],
        "ALL_AUDIO_TEXT": ["whisper", "beats", "qwen_ctx"],
    }
    null = {}
    for name, streams in GROUPS.items():
        if not any(s in feats for s in streams):
            continue
        deltas = []
        for _ in range(K):
            p = torch.randperm(B, generator=g)
            f = dict(feats)
            for s in streams:
                if s in f:
                    f[s] = f[s][p].contiguous()
            deltas.append(float(acc(f)) - float(base))
        d = np.asarray(deltas, dtype=np.float64)
        null[name] = {
            "k": K,
            "mean_delta": float(d.mean()),
            "sd_delta": float(d.std(ddof=1)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "frac_negative": float((d < 0).mean()),
            "mean_pct_of_base": float(d.mean() / float(base) * 100.0),
        }
        print(f"{name:16s} mean Δ={d.mean():+.5f} sd={d.std(ddof=1):.5f} "
              f"ci95=[{np.percentile(d, 2.5):+.5f},{np.percentile(d, 97.5):+.5f}] "
              f"({d.mean() / float(base) * 100.0:+.2f}% of base)")
    res["permutation_null"] = null
    res["permutation_null_seed"] = 13

    print(json.dumps(res, indent=2, default=str))
    return res


@app.local_entrypoint()
def main():
    print(json.dumps(diagnose.remote(), indent=2, default=str))
