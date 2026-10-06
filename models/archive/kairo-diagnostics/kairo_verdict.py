"""kairo_verdict.py — the DECISIVE, correctly-framed differentiation test (CPU, cents).

Key realization from the sweep: time-AVERAGED parcel pattern is ~0.98 across any two
Friends clips (both average toward 'a brain watching TV'), but the PER-SECOND time
series are nearly uncorrelated (timewise ~0.03). 'Differentiate two videos' means the
per-second readout must DIFFER between videos and MATCH for the same video. That's a
temporal-pattern test, not a grand-mean test.

This script runs the proper controls with the canonical wiring (crossstream_mix):

  Control 1 (SAME):  clip A vs clip A  -> per-second corr must be ~1.0
  Control 2 (DIFF):  clip A vs clip B  -> per-second corr must be LOW
  Control 3 (DIFF):  clip A vs clip C  -> per-second corr must be LOW
  Plus per-network per-second time-series correlation (the served artifact).

Verdict YES iff: same==~1.0 AND different clearly < same (large margin).
"""
from __future__ import annotations
import modal

app = modal.App("kairo-verdict")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "h5py>=3.10")
    .add_local_python_source("kairo_model", "kairo_diff2")
)
weights_vol = modal.Volume.from_name("kairo-weights")
features_vol = modal.Volume.from_name("kairo-features")
VOLUMES = {"/weights": weights_vol, "/features": features_vol}
CKPT = "/weights/v4_seed_0/best_model.pt"

from kairo_diff2 import _load_clip_feats, _yeo7, _forward  # reuse corrected loaders/wiring


def _run_clip(model, clip, subj):
    import torch, numpy as np
    feats = _load_clip_feats(clip, standardize=False)
    t = {s: torch.from_numpy(v)[None] for s, v in feats.items()}
    with torch.no_grad():
        parcels = _forward(model, t, subj, "crossstream_mix")   # (T,1000)
    nets = _yeo7(parcels)                                         # (T,7) signed
    return parcels, nets


def _ts_corr(x, y):
    """Mean per-timestep Pearson corr across the 1000-parcel pattern."""
    import numpy as np
    T = min(x.shape[0], y.shape[0])
    cs = []
    for t in range(T):
        a, b = x[t], y[t]
        if a.std() > 1e-8 and b.std() > 1e-8:
            cs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(cs)) if cs else float("nan")


def _net_ts_corr(na, nb):
    """Per-network correlation of the per-second time series (the served artifact)."""
    import numpy as np
    T = min(na.shape[0], nb.shape[0])
    out = []
    for i in range(7):
        a, b = na[:T, i], nb[:T, i]
        if a.std() > 1e-8 and b.std() > 1e-8:
            out.append(float(np.corrcoef(a, b)[0, 1]))
        else:
            out.append(float("nan"))
    return out


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=32768, timeout=2400)
def verdict(clip_a: str = "s01e01a", clip_b: str = "s01e15a", clip_c: str = "s02e01a", subject: int = 0):
    import numpy as np, torch
    from kairo_model import load_kairo
    model, meta, report = load_kairo(CKPT)
    model.eval()
    subj = torch.tensor([subject], dtype=torch.long)
    YEO = ["Visual","Somatomotor","DorsAttn","VentAttn","Limbic","Cont","Default"]

    pa, na = _run_clip(model, clip_a, subj)
    pa2, na2 = _run_clip(model, clip_a, subj)   # determinism control
    pb, nb = _run_clip(model, clip_b, subj)
    pc, nc = _run_clip(model, clip_c, subj)

    same_parcel = _ts_corr(pa, pa2)
    diff_ab = _ts_corr(pa, pb)
    diff_ac = _ts_corr(pa, pc)
    diff_bc = _ts_corr(pb, pc)

    net_same = _net_ts_corr(na, na2)
    net_ab = _net_ts_corr(na, nb)
    net_ac = _net_ts_corr(na, nc)

    print("\n================= KAIRO DIFFERENTIATION VERDICT =================")
    print(f"val_pearson={meta.get('val_pearson'):.4f}  load_clean={not report['missing'] and not report['unexpected']}")
    print(f"clips: A={clip_a}(T={pa.shape[0]})  B={clip_b}(T={pb.shape[0]})  C={clip_c}(T={pc.shape[0]})")
    print("\n--- per-second 1000-parcel pattern correlation ---")
    print(f"  SAME  (A vs A again):  {same_parcel:.4f}   (must be ~1.000)")
    print(f"  DIFF  (A vs B):        {diff_ab:.4f}")
    print(f"  DIFF  (A vs C):        {diff_ac:.4f}")
    print(f"  DIFF  (B vs C):        {diff_bc:.4f}")
    print("\n--- per-network per-second time-series corr (served artifact) ---")
    print("  net          A-vs-A     A-vs-B     A-vs-C")
    for i, n in enumerate(YEO):
        print(f"  {n:<11}{net_same[i]:>9.3f}{net_ab[i]:>11.3f}{net_ac[i]:>11.3f}")

    diff_mean = float(np.nanmean([diff_ab, diff_ac, diff_bc]))
    margin = same_parcel - diff_mean
    differentiates = (same_parcel > 0.999) and (diff_mean < 0.6) and (margin > 0.3)
    print(f"\n  same={same_parcel:.4f}  mean-different={diff_mean:.4f}  margin={margin:.4f}")
    print(f"  >>> KAIRO PROVABLY DIFFERENTIATES VIDEOS: {'YES' if differentiates else 'NO'} <<<")
    return {
        "val_pearson": meta.get("val_pearson"),
        "same_parcel_corr": same_parcel,
        "diff_ab": diff_ab, "diff_ac": diff_ac, "diff_bc": diff_bc,
        "diff_mean": diff_mean, "margin": margin,
        "net_same": net_same, "net_ab": net_ab, "net_ac": net_ac,
        "differentiates": bool(differentiates),
        "clips": {"a": clip_a, "b": clip_b, "c": clip_c},
    }


@app.local_entrypoint()
def main(clip_a: str = "s01e01a", clip_b: str = "s01e15a", clip_c: str = "s02e01a", subject: int = 0):
    import json
    print(json.dumps(verdict.remote(clip_a, clip_b, clip_c, subject), indent=2, default=str))
