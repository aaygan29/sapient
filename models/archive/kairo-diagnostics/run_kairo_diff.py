"""run_kairo_diff.py — Kairo 2-video DIFFERENTIATION TEST (prepared; runs on Modal).

⚠️ SUPERSEDED (2026-06-10): this file's `differentiate` entrypoint compares the
TIME-AVERAGED parcel pattern, which is ~0.98 for ANY two Friends clips (both
average toward a generic 'watching-TV' brain) — it FALSELY reads as collapse.
The model is NOT collapsed. The correct test compares the PER-SECOND readout:
same clip twice = 1.000, two different clips = ~0.16–0.21 (margin ~0.81).
Use instead:
    modal run kairo/kairo_verdict.py     # decisive per-second verdict (CPU)
    modal run kairo/kairo_serve.py::gate # served-path 2-video gate (CPU)
Feature loading here was also buggy; the corrected loader lives in kairo_diff2.py.


Two entrypoints:

  (A) FREE, CPU, ~$0 — the architecture proof:
        modal run kairo/run_kairo_diff.py::dryrun
      Loads v4_seed_0/best_model.pt into the reconstructed KairoEncoder and
      reports missing/unexpected keys. If both are empty (modulo recomputed RoPE
      buffers) the architecture is PROVEN correct.

  (B) CHEAP, CPU, ~$0.02 — the decisive 2-video test (DO NOT run until greenlit):
        modal run kairo/run_kairo_diff.py::differentiate
      Loads cached features for TWO different Friends clips (no backbones, no
      GPU), runs Kairo, and reports whether the 1000-parcel / 7-network outputs
      DIFFER between the two videos. This is the whole point: prove the model
      perceives different content differently.

Why CPU is enough: the backbones are NOT run (features are pre-extracted on the
kairo-features volume). Kairo itself is ~130M params; a forward pass on ~600
timesteps × 2 clips is seconds on CPU. No GPU spend required for the test.
"""
from __future__ import annotations
import modal

app = modal.App("kairo-diff")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "h5py>=3.10")
    .add_local_python_source("kairo_model")
)

weights_vol = modal.Volume.from_name("kairo-weights")
features_vol = modal.Volume.from_name("kairo-features")
VOLUMES = {"/weights": weights_vol, "/features": features_vol}

CKPT = "/weights/v4_seed_0/best_model.pt"

# Two DIFFERENT videos that have features across ALL 8 backbones (from probe:
# 95 friends clips are common). Pick two visually/emotionally distinct ones.
CLIP_A = "s01e01a"   # Friends S1E1 part a
CLIP_B = "s01e15a"   # Friends S1E15 part a  (different episode/content)


def _load_clip_feats(clip: str):
    """Return {stream: np.ndarray (T, in_dim)} for one clip, from cached .h5.
    Slices 4D (T,1,1,D)->(T,D) and 3D (T,1,D)->(T,D); emonet nests under <clip>/visual."""
    import glob, os
    import h5py
    import numpy as np
    from kairo_model import STREAM_ORDER

    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    streams = ck["streams"]
    out = {}
    for s in STREAM_ORDER:
        subdir, layer, dim = streams[s]
        base = f"/features/{subdir}/friends"
        cands = glob.glob(os.path.join(base, "**", f"*{clip}*.h5"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(base, "**", f"friends_{clip}.h5"), recursive=True)
        if not cands:
            raise FileNotFoundError(f"{s}: no .h5 for clip {clip} under {base}")
        path = sorted(cands)[0]
        with h5py.File(path, "r") as f:
            if layer in f:
                arr = f[layer][:]
            elif clip in f and layer in f[clip]:
                arr = f[clip][layer][:]
            else:
                # emonet-style nesting: <clip>/visual
                grp = f[clip] if clip in f else f
                key = layer.split("/")[-1]
                arr = grp[key][:] if key in grp else grp[list(grp.keys())[0]][:]
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr.squeeze(1) if arr.shape[1] == 1 else arr.reshape(arr.shape[0], -1)
        if arr.shape[-1] != dim:
            arr = arr[:, :dim] if arr.shape[-1] > dim else np.pad(arr, ((0, 0), (0, dim - arr.shape[-1])))
        out[s] = arr
    # align all streams to the common min T
    T = min(a.shape[0] for a in out.values())
    return {s: a[:T] for s, a in out.items()}


def _yeo7_from_parcels(parcels):
    """(T,1000) -> (T,7) Yeo network means using the shipped Schaefer→Yeo7 CSV."""
    import numpy as np
    csv = "/features/parcel_network_divisions/Schaefer_1000Parcel_7Networks_order_Yeo7.csv"
    labels = np.loadtxt(csv, dtype=int)            # (1000,) network id 1..7
    p = np.abs(parcels)
    T = p.shape[0]
    nets = np.zeros((T, 7), dtype=np.float64)
    for i in range(7):
        m = labels == (i + 1)
        if m.any():
            nets[:, i] = p[:, m].mean(axis=1)
    return nets


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=16384, timeout=900)
def dryrun():
    """FREE architecture proof — load_state_dict into the reconstructed module."""
    import torch
    from kairo_model import load_kairo
    model, meta, report = load_kairo(CKPT)
    print("=== KAIRO v4 DRY-RUN (CPU, architecture proof) ===")
    print("meta:", {k: meta[k] for k in ("epoch", "seed", "val_pearson", "version",
                                         "n_streams", "n_backbones") if k in meta})
    print("dropped rope buffers:", report["dropped_rope_buffers"])
    print("MISSING keys   (model wants, ckpt lacks):", report["missing"])
    print("UNEXPECTED keys(ckpt has, model lacks)  :", report["unexpected"])
    ok = not report["missing"] and not report["unexpected"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print(f">>> ARCHITECTURE MATCH: {'YES — load is clean' if ok else 'PARTIAL — inspect keys above'} <<<")
    return {"clean": ok, "missing": report["missing"], "unexpected": report["unexpected"],
            "n_params": n_params, "meta": {k: str(v) for k, v in meta.items() if k != "streams"}}


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=32768, timeout=1800)
def differentiate(clip_a: str = CLIP_A, clip_b: str = CLIP_B, subject: int = 0):
    """THE DECISIVE TEST — do two different videos give different brain outputs?"""
    import numpy as np
    import torch
    from kairo_model import load_kairo

    model, meta, report = load_kairo(CKPT)
    if report["missing"] or report["unexpected"]:
        print("WARN: state_dict not clean:", report["missing"], report["unexpected"])
    model.eval()

    results = {}
    for tag, clip in (("A", clip_a), ("B", clip_b)):
        feats_np = _load_clip_feats(clip)
        feats = {s: torch.from_numpy(v)[None].float() for s, v in feats_np.items()}  # (1,T,D)
        subj = torch.tensor([subject], dtype=torch.long)
        with torch.no_grad():
            parcels = model(feats, subj).squeeze(0).numpy()    # (T, 1000)
        nets = _yeo7_from_parcels(parcels)                     # (T, 7)
        results[tag] = {
            "clip": clip, "T": int(parcels.shape[0]),
            "parcels_mean": parcels.mean(0), "nets_mean": nets.mean(0),
            "parcels_full": parcels,
        }

    A, B = results["A"], results["B"]
    YEO = ["Visual", "Somatomotor", "DorsAttn", "VentAttn", "Limbic", "Cont", "Default"]
    # Are the outputs different?
    T = min(A["parcels_full"].shape[0], B["parcels_full"].shape[0])
    pa, pb = A["parcels_full"][:T], B["parcels_full"][:T]
    parcel_corr = float(np.corrcoef(A["parcels_mean"], B["parcels_mean"])[0, 1])
    net_l1 = float(np.abs(A["nets_mean"] - B["nets_mean"]).sum())
    net_corr = float(np.corrcoef(A["nets_mean"], B["nets_mean"])[0, 1])
    timewise_corr = float(np.mean([np.corrcoef(pa[t], pb[t])[0, 1] for t in range(min(T, 50))]))
    lead_a = YEO[int(np.argmax(A["nets_mean"]))]
    lead_b = YEO[int(np.argmax(B["nets_mean"]))]

    print("\n================ KAIRO 2-VIDEO DIFFERENTIATION ================")
    print(f"meta: val_pearson={meta.get('val_pearson')} epoch={meta.get('epoch')} version={meta.get('version')}")
    print(f"clip A = {A['clip']} (T={A['T']})   clip B = {B['clip']} (T={B['T']})")
    print("\n7-network mean (abs) per clip:")
    print("  net        A         B        |A-B|")
    for i, n in enumerate(YEO):
        a, b = A["nets_mean"][i], B["nets_mean"][i]
        print(f"  {n:<10}{a:>9.4f}{b:>9.4f}{abs(a-b):>9.4f}")
    print(f"\n  leading network:  A->{lead_a}   B->{lead_b}")
    print(f"  parcel-mean corr (A vs B):   {parcel_corr:.4f}   (1.0 = identical = COLLAPSE)")
    print(f"  network-mean corr (A vs B):  {net_corr:.4f}")
    print(f"  network L1 distance:         {net_l1:.4f}")
    print(f"  per-timestep parcel corr:    {timewise_corr:.4f}")
    differentiates = (parcel_corr < 0.98) and (net_l1 > 0.02 or lead_a != lead_b)
    print(f"\n  >>> KAIRO DIFFERENTIATES THE TWO VIDEOS: {'YES' if differentiates else 'NO (collapse)'} <<<")
    return {
        "clip_a": A["clip"], "clip_b": B["clip"],
        "nets_A": A["nets_mean"].tolist(), "nets_B": B["nets_mean"].tolist(),
        "parcel_mean_corr": parcel_corr, "net_corr": net_corr, "net_l1": net_l1,
        "timewise_corr": timewise_corr, "lead_a": lead_a, "lead_b": lead_b,
        "differentiates": bool(differentiates),
    }


@app.local_entrypoint()
def main(mode: str = "dryrun", clip_a: str = CLIP_A, clip_b: str = CLIP_B):
    import json
    if mode == "dryrun":
        print(json.dumps(dryrun.remote(), indent=2, default=str))
    elif mode == "differentiate":
        print(json.dumps(differentiate.remote(clip_a, clip_b), indent=2, default=str))
    else:
        raise SystemExit("mode must be 'dryrun' or 'differentiate'")
