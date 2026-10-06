"""_visual_groundtruth.py — DEFINITIVE read-only investigation of whether the
SERVED Qualia (Kairo v4) checkpoint is trained on + actually USES visual streams.

Runs on Modal against the EXACT served checkpoint (/weights/v4_seed_0/best_model.pt
on kairo-weights) and the SAME production scoring code (kairo_model + the _forward
math copied from kairo_serve). Does NOT touch the racy gate_url harness.

Three pieces of evidence:
  (1) STREAM CONTRACT: list ckpt['streams'] / ckpt['feature_dims'] and which visual
      streams are part of the trained contract.
  (2) WEIGHT STATS: per-stream input-projection L2 norm / std / near-zero fraction.
      Compare visual heads vs audio/language heads.
  (3) CLEAN ABLATION: score ONE real Friends clip with visuals REAL, then with the
      visual streams ZEROED (distinct in-memory tensors, checksummed so caching can't
      fake identity). Compare per-second Yeo-7 networkTimeSeries.
"""
from __future__ import annotations
import modal

app = modal.App("qualia-visual-groundtruth")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "h5py>=3.10")
    .add_local_python_source("kairo_model")
)

weights_vol = modal.Volume.from_name("kairo-weights")
features_vol = modal.Volume.from_name("kairo-features")
VOLUMES = {"/weights": weights_vol, "/features": features_vol}
CKPT = "/weights/v4_seed_0/best_model.pt"

VISUAL = {"vjepa_block5", "vjepa_block15", "vjepa_block23",
          "vmae2_block10", "vmae2_block18", "vmae2_block25",
          "ivl3_layer10", "ivl3_layer15", "ivl3_layer20", "ivl3_norm",
          "vjepa2", "emonet"}
AUDIO = {"whisper_layer12", "whisper_layer25", "whisper_layer31",
         "whisper_layernorm", "w2vbert"}
LANG = {"llama"}

YEO7_ORDER = ["Visual", "Somatomotor", "Dorsal Attention",
              "Ventral Attention", "Limbic", "Frontoparietal", "Default Mode"]
YEO7_CSV = "/features/parcel_network_divisions/Schaefer_1000Parcel_7Networks_order_Yeo7.csv"


def _load_clip_feats(clip, streams):
    import glob, os, h5py, numpy as np

    def _clip_to_episode(c):
        return c[:-1] if c and c[-1].isalpha() else c

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
                ep = _clip_to_episode(clip)
                grp = None
                for gk in (ep, clip):
                    if gk in f:
                        grp = f[gk]; break
                if grp is None:
                    grp = f[list(f.keys())[0]]
                lk = layer.split("/")[-1]
                arr = grp[lk][:] if lk in grp else grp[list(grp.keys())[0]][:]
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr.squeeze(1) if arr.shape[1] == 1 else arr.reshape(arr.shape[0], -1)
        if arr.shape[-1] != dim:
            arr = arr[:, :dim] if arr.shape[-1] > dim else np.pad(arr, ((0, 0), (0, dim - arr.shape[-1])))
        out[s] = arr
    T = min(a.shape[0] for a in out.values())
    return {s: a[:T] for s, a in out.items()}


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=32768, timeout=1800)
def investigate(clip_id: str = "s01e01a"):
    import numpy as np, torch
    from kairo_model import load_kairo, STREAM_ORDER, FEATURE_DIMS

    report = {}

    # ── load served checkpoint exactly as serve does ────────────────────────────
    model, meta, load_report = load_kairo(CKPT)
    model.eval()
    report["load_report"] = {k: list(v) if isinstance(v, list) else v for k, v in load_report.items()}
    report["meta_keys"] = sorted([k for k in meta.keys()])
    report["meta_scalars"] = {k: meta[k] for k in meta
                              if isinstance(meta.get(k), (int, float, str, bool))}

    # ── (1) STREAM CONTRACT ─────────────────────────────────────────────────────
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    streams = ckpt.get("streams", {})
    report["ckpt_streams"] = {s: list(v) for s, v in streams.items()}
    report["ckpt_feature_dims"] = ckpt.get("feature_dims", {})
    report["ckpt_stream_names"] = sorted(list(streams.keys()))
    report["visual_streams_in_contract"] = sorted([s for s in streams if s in VISUAL])
    report["audio_streams_in_contract"] = sorted([s for s in streams if s in AUDIO])
    report["lang_streams_in_contract"] = sorted([s for s in streams if s in LANG])
    report["projection_keys"] = sorted(list(model.encoder.projections.keys()))

    # ── (2) WEIGHT STATS per input projection ───────────────────────────────────
    def stats(W):
        Wf = W.detach().float()
        l2 = float(Wf.norm().item())
        std = float(Wf.std().item())
        mean_abs = float(Wf.abs().mean().item())
        nz = float((Wf.abs() < 1e-6).float().mean().item())
        # per-input-feature column norm: a DEAD input feature has ~0 column norm.
        col_norm = Wf.norm(dim=0)  # (in_dim,)
        frac_dead_inputs = float((col_norm < 1e-4 * (col_norm.mean() + 1e-12)).float().mean().item())
        return {"l2": l2, "std": std, "mean_abs": mean_abs,
                "frac_near_zero": nz, "frac_dead_input_cols": frac_dead_inputs,
                "in_dim": int(Wf.shape[1]), "rms_per_elem": float(Wf.pow(2).mean().sqrt().item())}

    wstats = {}
    for s in STREAM_ORDER:
        proj = model.encoder.projections[s]
        wstats[s] = stats(proj.weight)
        wstats[s]["bias_l2"] = float(proj.bias.detach().float().norm().item()) if proj.bias is not None else None
        wstats[s]["group"] = ("VISUAL" if s in VISUAL else "AUDIO" if s in AUDIO
                              else "LANG" if s in LANG else "OTHER")
    report["weight_stats"] = wstats

    def grp_mean(metric, grp):
        vals = [wstats[s][metric] for s in STREAM_ORDER if wstats[s]["group"] == grp]
        return float(np.mean(vals)) if vals else None

    report["group_summary"] = {
        g: {"mean_rms_per_elem": grp_mean("rms_per_elem", g),
            "mean_std": grp_mean("std", g),
            "mean_frac_near_zero": grp_mean("frac_near_zero", g),
            "mean_frac_dead_input_cols": grp_mean("frac_dead_input_cols", g),
            "n_streams": sum(1 for s in STREAM_ORDER if wstats[s]["group"] == g)}
        for g in ("VISUAL", "AUDIO", "LANG")
    }

    # ── (3) CLEAN ABLATION on the production scoring path ───────────────────────
    labels = np.loadtxt(YEO7_CSV, dtype=int)

    def forward(feats):
        # EXACT copy of kairo_serve.Kairo._forward math.
        t = {s: torch.from_numpy(feats[s])[None] for s in STREAM_ORDER}
        subj = torch.tensor([0], dtype=torch.long)
        projected = [model.encoder.projections[s](t[s]) for s in STREAM_ORDER]
        stk = torch.stack(projected, dim=2)
        B, T, S, C = stk.shape
        mixed = model.encoder.transformer(stk.reshape(B * T, S, C)).reshape(B, T, S, C)
        subj_e = model.encoder.subject_emb(subj)
        subj_t = subj_e[:, None, None, :].expand(B, T, 1, C)
        cat = torch.cat([mixed, subj_t], dim=2).reshape(B, T, (S + 1) * C)
        with torch.no_grad():
            h = model.bottleneck(cat)
            parcels = model.predictor(h)
        return parcels.squeeze(0).numpy()

    def to_networks(parcels):
        n_sec = parcels.shape[0]
        nps = np.zeros((n_sec, 7))
        for i in range(7):
            m = labels == (i + 1)
            if m.any():
                nps[:, i] = parcels[:, m].mean(1)
        return nps  # (T,7)

    feats_real = _load_clip_feats(clip_id, streams)
    # checksum visual arrays in REAL input
    def vis_checksum(fd):
        h = 0.0
        for s in sorted(fd):
            if s in VISUAL:
                h += float(np.abs(np.asarray(fd[s], dtype=np.float64)).sum())
        return h

    feats_zero = {s: (np.zeros_like(a) if s in VISUAL else a.copy())
                  for s, a in feats_real.items()}

    report["ablation_inputs"] = {
        "clip_id": clip_id,
        "T": int(next(iter(feats_real.values())).shape[0]),
        "visual_checksum_REAL": vis_checksum(feats_real),
        "visual_checksum_ZEROED": vis_checksum(feats_zero),
        "non_visual_identical": all(
            np.array_equal(feats_real[s], feats_zero[s])
            for s in feats_real if s not in VISUAL),
        "visual_streams_zeroed": sorted([s for s in feats_zero if s in VISUAL]),
    }

    par_real = forward(feats_real)
    par_zero = forward(feats_zero)
    net_real = to_networks(par_real)  # (T,7)
    net_zero = to_networks(par_zero)

    T = min(net_real.shape[0], net_zero.shape[0])
    nr, nz = net_real[:T], net_zero[:T]

    # parcel-level deltas
    pr, pz = par_real[:T], par_zero[:T]
    abs_parcel_delta = float(np.abs(pr - pz).mean())
    rel_parcel_delta = float(np.abs(pr - pz).mean() / (np.abs(pr).mean() + 1e-12))
    max_parcel_delta = float(np.abs(pr - pz).max())

    # per-network timecourse correlation + relative delta
    net_corr = []
    net_reldelta = []
    for i in range(7):
        x, y = nr[:, i], nz[:, i]
        c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 and y.std() > 1e-12 else float("nan")
        net_corr.append(c)
        net_reldelta.append(float(np.abs(x - y).mean() / (np.abs(x).mean() + 1e-12)))

    # per-second 7-vector correlation (the "shape changes" measure)
    ps_corr = []
    for tt in range(T):
        x, y = nr[tt], nz[tt]
        if x.std() > 1e-12 and y.std() > 1e-12:
            ps_corr.append(float(np.corrcoef(x, y)[0, 1]))
    persec_corr_mean = float(np.mean(ps_corr)) if ps_corr else float("nan")

    report["ablation_result"] = {
        "abs_parcel_delta_mean": abs_parcel_delta,
        "rel_parcel_delta_mean": rel_parcel_delta,
        "max_parcel_delta": max_parcel_delta,
        "byte_identical_parcels": bool(np.array_equal(pr, pz)),
        "per_network_timecourse_corr_real_vs_zeroed": {YEO7_ORDER[i]: net_corr[i] for i in range(7)},
        "per_network_rel_delta": {YEO7_ORDER[i]: net_reldelta[i] for i in range(7)},
        "mean_network_corr": float(np.nanmean(net_corr)),
        "persec_7vec_corr_mean": persec_corr_mean,
        "sample_network_real_sec0": {YEO7_ORDER[i]: float(nr[0, i]) for i in range(7)},
        "sample_network_zeroed_sec0": {YEO7_ORDER[i]: float(nz[0, i]) for i in range(7)},
    }

    return report


@app.local_entrypoint()
def main(clip_id: str = "s01e01a"):
    import json
    r = investigate.remote(clip_id)
    print(json.dumps(r, indent=2, default=str))
