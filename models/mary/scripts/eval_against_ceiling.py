"""Evaluate a Mary checkpoint against the ISC noise ceiling (proof + honest baseline).

Reports, on the val split:
  - raw whole-brain mean r            (reproduces the misleading "~0.009")
  - mean r over responsive vertices   (ISC > 0.05 and ISC > 0.1)
  - ceiling-normalized r (% of ISC)   on responsive vertices — the honest dial
  - parcel-free top-k mean r          (top 10% vertices by ISC)

Inference only — no training. Run on a cheap GPU.
  modal run scripts/eval_against_ceiling.py --ckpt /data/checkpoints/mary_multi_v2_s23/latest.pt
"""
from __future__ import annotations

import modal

app = modal.App("mary-eval-ceiling")
_REPO = "/Users/robertgutierrez/Desktop/sapient-models/mary"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "transformers==4.46.0", "pyyaml>=6")
    .add_local_dir(_REPO, remote_path="/root/maryrepo",
                   ignore=["**/__pycache__", "**/*.pt", "**/*.npy", "**/.git"])
)
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=30 * 60, volumes={"/data": vol})
def run(ckpt: str, manifest: str = "manifest_mary_multi.json") -> dict:
    import os, sys
    os.chdir("/root/maryrepo")
    sys.path.insert(0, "/root/maryrepo")
    import json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from mary.model import MaryConfig, MaryModel
    from data.dataset import MaryDataset, collate_mary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device)
    cfg = state.get("config", {})
    mcfg = MaryConfig.from_yaml(cfg)
    model = MaryModel(mcfg).to(device).eval()
    # Drop shape-mismatched keys (stream-encoder dim drift for streams absent from
    # this checkpoint's training data, e.g. qwen_vl version bump 3584->4096), then
    # load non-strict so the trained streams (audio/text) load cleanly.
    msd = model.state_dict()
    sd = state["state_dict"]
    filtered = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    dropped = [k for k in sd if k not in filtered]
    incompat = model.load_state_dict(filtered, strict=False)
    n_dropped = len(dropped)

    seq_len = int(cfg.get("train", {}).get("sequence_length", 200))
    val = MaryDataset(f"/data/{manifest}", "val", sequence_length=seq_len)
    loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_mary)

    n_subj_model = int(mcfg.n_subjects)
    preds, targs = [], []
    with torch.no_grad():
        for b in loader:
            feats = {k: v.to(device) for k, v in b["features"].items()}
            # checkpoint was trained on fewer subjects than the current manifest;
            # clamp the per-subject head index so the embedding lookup is in range.
            sidx = b["subject_idx"].clamp(max=n_subj_model - 1).to(device)
            p = model(feats, sidx)
            preds.append(p.float().cpu())
            targs.append(b["fmri"])
    P = torch.cat(preds).reshape(-1, 20484).numpy()
    T = torch.cat(targs).reshape(-1, 20484).numpy()

    # per-vertex r along the concatenated time axis
    p = P - P.mean(0, keepdims=True)
    t = T - T.mean(0, keepdims=True)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    den = np.where(den < 1e-8, 1e-8, den)
    r = np.nan_to_num((p * t).sum(0) / den)          # (20484,)

    ceil = np.load("/data/noise_ceiling_isc.npy")    # (20484,)

    def stats(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        rr = r[mask]
        cc = ceil[mask]
        norm = np.clip(rr, 0, None) / np.clip(cc, 1e-3, None)
        return {"n": n, "r_mean": float(rr.mean()),
                "r_median": float(np.median(rr)),
                "pct_of_ceiling": float(np.median(norm))}

    return {
        "ckpt": ckpt,
        "n_dropped_mismatch_keys": n_dropped,
        "n_val_timepoints": int(P.shape[0]),
        "whole_brain": stats(np.ones(20484, bool)),
        "responsive_isc>0.05": stats(ceil > 0.05),
        "responsive_isc>0.1": stats(ceil > 0.1),
        "top10pct_by_isc": stats(ceil >= np.quantile(ceil, 0.90)),
    }


@app.local_entrypoint()
def main(ckpt: str = "/data/checkpoints/mary_multi_v2_s23/latest.pt",
         manifest: str = "manifest_mary_multi.json"):
    import json
    print(json.dumps(run.remote(ckpt, manifest), indent=2))
