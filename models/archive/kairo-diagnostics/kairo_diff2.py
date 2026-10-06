"""kairo_diff2.py — corrected feature loading + multi-wiring differentiation sweep (CPU, cents).

Fixes vs run_kairo_diff.py:
  - loads features strictly via ckpt['streams'] (subdir, exact h5 layer, dim)
  - handles emonet nesting <clip>/visual correctly
  - squeezes (T,1,1,D)->(T,D) and (T,1,D)->(T,D)
  - normalizes each stream (z-score per feature) optionally — training likely standardized inputs

Then sweeps candidate forward wirings to find one that DIFFERENTIATES two clips while
keeping SAME-clip-twice identical. Reports parcel corr / net corr / leading nets for each.
"""
from __future__ import annotations
import modal

app = modal.App("kairo-diff2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "h5py>=3.10")
    .add_local_python_source("kairo_model")
)

weights_vol = modal.Volume.from_name("kairo-weights")
features_vol = modal.Volume.from_name("kairo-features")
VOLUMES = {"/weights": weights_vol, "/features": features_vol}
CKPT = "/weights/v4_seed_0/best_model.pt"


def _clip_to_episode(clip):
    # s01e01a -> s01e01  (emonet group name)
    return clip[:-1] if clip and clip[-1].isalpha() else clip


def _load_clip_feats(clip, standardize=False):
    import glob, os, h5py, numpy as np, torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    streams = ck["streams"]
    out = {}
    for s, (subdir, layer, dim) in streams.items():
        base = f"/features/{subdir}"
        cands = glob.glob(os.path.join(base, "**", f"friends_{clip}.h5"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(base, "**", f"*{clip}*.h5"), recursive=True)
        if not cands:
            raise FileNotFoundError(f"{s}: no h5 for {clip} under {base}")
        path = sorted(cands)[0]
        with h5py.File(path, "r") as f:
            if layer in f:
                arr = f[layer][:]
            else:
                # nested under clip group (emonet: <s01e01>/visual)
                ep = _clip_to_episode(clip)
                grp = None
                for gk in (ep, clip):
                    if gk in f:
                        grp = f[gk]; break
                if grp is None:
                    grp = f[list(f.keys())[0]]
                lk = layer.split("/")[-1]
                if lk in grp:
                    arr = grp[lk][:]
                else:
                    arr = grp[list(grp.keys())[0]][:]
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr.squeeze(1) if arr.shape[1] == 1 else arr.reshape(arr.shape[0], -1)
        if arr.shape[-1] != dim:
            arr = arr[:, :dim] if arr.shape[-1] > dim else np.pad(arr, ((0,0),(0,dim-arr.shape[-1])))
        if standardize:
            mu = arr.mean(0, keepdims=True); sd = arr.std(0, keepdims=True) + 1e-6
            arr = (arr - mu) / sd
        out[s] = arr
    T = min(a.shape[0] for a in out.values())
    return {s: a[:T] for s, a in out.items()}


def _yeo7(parcels):
    import numpy as np, glob
    csv = glob.glob("/features/**/Schaefer_1000Parcel_7Networks_order_Yeo7.csv", recursive=True)
    if not csv:
        csv = glob.glob("/features/parcel_network_divisions/*Yeo7*.csv")
    labels = np.loadtxt(csv[0], dtype=int)
    p = parcels
    nets = np.zeros((p.shape[0], 7))
    for i in range(7):
        m = labels == (i+1)
        if m.any():
            nets[:, i] = p[:, m].mean(1)
    return nets


# ---- forward wirings -------------------------------------------------------
def _project_streams(model, feats):
    """Return list of (B,T,256) per stream in STREAM_ORDER."""
    import torch
    from kairo_model import STREAM_ORDER
    return [model.encoder.projections[s](feats[s]) for s in STREAM_ORDER]


def _forward(model, feats, subj, wiring):
    """feats: {s:(1,T,D)}; returns parcels (T,1000) numpy."""
    import torch
    from kairo_model import STREAM_ORDER, D_MODEL
    projected = _project_streams(model, feats)          # 18 x (1,T,256)
    stk = torch.stack(projected, dim=2)                 # (1,T,18,256)
    B, T, S, C = stk.shape
    subj_e = model.encoder.subject_emb(subj)            # (1,256)

    if wiring == "crossstream_mix":   # current default in run_kairo_diff
        mixed = model.encoder.transformer(stk.reshape(B*T, S, C)).reshape(B, T, S, C)
        subj_t = subj_e[:, None, None, :].expand(B, T, 1, C)
        cat = torch.cat([mixed, subj_t], dim=2).reshape(B, T, (S+1)*C)
    elif wiring == "no_transformer":  # skip encoder.transformer entirely
        subj_t = subj_e[:, None, None, :].expand(B, T, 1, C)
        cat = torch.cat([stk, subj_t], dim=2).reshape(B, T, (S+1)*C)
    elif wiring == "mix_with_subj_token":  # subject as 19th token THROUGH transformer
        subj_t = subj_e[:, None, None, :].expand(B, T, 1, C)
        seq = torch.cat([stk, subj_t], dim=2)           # (1,T,19,256)
        mixed = model.encoder.transformer(seq.reshape(B*T, S+1, C)).reshape(B, T, S+1, C)
        cat = mixed.reshape(B, T, (S+1)*C)
    else:
        raise ValueError(wiring)

    h = model.bottleneck(cat)
    parcels = model.predictor(h)
    return parcels.squeeze(0).detach().numpy()


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=32768, timeout=2400)
def sweep(clip_a: str = "s01e01a", clip_b: str = "s01e15a", subject: int = 0, standardize: bool = False):
    import numpy as np, torch, json
    from kairo_model import load_kairo
    model, meta, report = load_kairo(CKPT)
    model.eval()
    clean = not report["missing"] and not report["unexpected"]
    print("load clean:", clean, "| val_pearson", meta.get("val_pearson"))

    fa = _load_clip_feats(clip_a, standardize)
    fb = _load_clip_feats(clip_b, standardize)
    ta = {s: torch.from_numpy(v)[None] for s, v in fa.items()}
    tb = {s: torch.from_numpy(v)[None] for s, v in fb.items()}
    subj = torch.tensor([subject], dtype=torch.long)
    YEO = ["Visual","Somatomotor","DorsAttn","VentAttn","Limbic","Cont","Default"]

    results = {}
    for wiring in ("crossstream_mix", "no_transformer", "mix_with_subj_token"):
        with torch.no_grad():
            pa = _forward(model, ta, subj, wiring)
            pb = _forward(model, tb, subj, wiring)
            pa2 = _forward(model, ta, subj, wiring)   # same clip twice -> must be identical
        T = min(pa.shape[0], pb.shape[0])
        na, nb = _yeo7(np.abs(pa)), _yeo7(np.abs(pb))
        parcel_corr = float(np.corrcoef(pa[:T].mean(0), pb[:T].mean(0))[0,1])
        same_corr = float(np.corrcoef(pa.mean(0), pa2.mean(0))[0,1])
        tw = float(np.mean([np.corrcoef(pa[t], pb[t])[0,1] for t in range(min(T,80))]))
        net_l1 = float(np.abs(na.mean(0)-nb.mean(0)).sum())
        lead_a, lead_b = YEO[int(np.argmax(na.mean(0)))], YEO[int(np.argmax(nb.mean(0)))]
        # raw (non-abs) parcel pattern spread
        raw_std_a = float(pa.std()); raw_std_b = float(pb.std())
        diff = (parcel_corr < 0.9) and same_corr > 0.999
        print(f"\n[{wiring}] standardize={standardize}")
        print(f"  same-clip-twice corr: {same_corr:.5f}  (must be ~1.0)")
        print(f"  A-vs-B parcel-mean corr: {parcel_corr:.4f}  (lower=more differentiation)")
        print(f"  A-vs-B per-timestep corr: {tw:.4f}")
        print(f"  net L1: {net_l1:.4f}  lead A->{lead_a} B->{lead_b}  raw std A={raw_std_a:.3f} B={raw_std_b:.3f}")
        print(f"  >>> DIFFERENTIATES: {'YES' if diff else 'no'}")
        results[wiring] = {"parcel_corr": parcel_corr, "same_corr": same_corr,
                           "timewise": tw, "net_l1": net_l1, "lead_a": lead_a,
                           "lead_b": lead_b, "raw_std_a": raw_std_a, "raw_std_b": raw_std_b,
                           "differentiates": diff}
    return {"clean_load": clean, "results": results,
            "clip_a": clip_a, "clip_b": clip_b, "standardize": standardize}


@app.local_entrypoint()
def main(clip_a: str = "s01e01a", clip_b: str = "s01e15a", subject: int = 0, standardize: bool = False):
    import json
    print(json.dumps(sweep.remote(clip_a, clip_b, subject, standardize), indent=2, default=str))
