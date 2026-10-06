"""Recompute KPI_BASELINE for a Mary checkpoint (live-score calibration).

The live score z-scores each KPI's mean activation against a corpus baseline
(mean,std). When the served checkpoint changes, that baseline must be recomputed
from the NEW model's output distribution. The raw KPI mean is baseline-INDEPENDENT
(it's |predicted activation| reduced to Yeo-7 networks, then a fixed KPI blend), and
serve's engine.encode is just model.forward()[0] with subject_idx=0 — so this offline
computation on the val corpus matches production exactly.

Per (subject,story) val entry: mean the per-window KPI means -> one KPI mean per entry
(clip-level granularity, matching the original 28-run baseline). KPI_BASELINE[kpi] =
(mean, std) across entries.

  modal run scripts/build_kpi_baseline.py --ckpt /data/checkpoints/mary_multi_stable_resp_s13/best.pt

The reduction here mirrors serve.py's responsive un-flatten EXACTLY (ISC>0.05,
ISC-weighted Yeo-7 means); change one without the other and live scores miscalibrate.
"""
from __future__ import annotations

import modal

app = modal.App("mary-build-kpi-baseline")
_REPO = "/Users/robertgutierrez/Desktop/sapient-models/mary"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "transformers==4.46.0", "pyyaml>=6")
    .add_local_dir(_REPO, remote_path="/root/maryrepo",
                   ignore=["**/__pycache__", "**/*.pt", "**/*.npy", "**/.git"])
)
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)

# Copied verbatim from mary/modal/serve.py (keep in sync).
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
KPI_NAMES = list(KPI_FROM_NETWORKS.keys())


@app.function(image=image, gpu="A10G", timeout=30 * 60, volumes={"/data": vol})
def run(ckpt: str, manifest: str = "manifest_mary_multi.json") -> dict:
    import os, sys
    os.chdir("/root/maryrepo"); sys.path.insert(0, "/root/maryrepo")
    import numpy as np, torch
    from collections import defaultdict
    from mary.model import MaryConfig, MaryModel
    from data.dataset import MaryDataset, collate_mary
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device)
    cfg = state.get("config", {})
    model = MaryModel(MaryConfig.from_yaml(cfg)).to(device).eval()
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in state["state_dict"].items()
                           if k in msd and v.shape == msd[k].shape}, strict=False)

    z = np.load("/data/cache/mary_yeo7_vertex_map.npz", allow_pickle=True)
    vertex_yeo = z["vertex_yeo"]
    yeo7_order = [str(x) for x in z["yeo7_order"]]
    net_idx = {n: i for i, n in enumerate(yeo7_order)}

    # Responsive un-flatten — MUST mirror serve.py _reduce_networks EXACTLY: restrict
    # each network to ISC>0.05 vertices, ISC-weighted mean. The reduction is what fixes
    # the flat ~0.075 all-vertex collapse, so the baseline must be built through it or
    # production scores miscalibrate. Guarded: missing ISC → all-vertex fallback.
    NOISE_CEILING_PATH = "/data/noise_ceiling_isc.npy"
    NOISE_CEILING_THRESHOLD = 0.05
    vertex_isc = None
    if os.path.exists(NOISE_CEILING_PATH):
        _isc = np.load(NOISE_CEILING_PATH).astype(np.float64)
        if _isc.shape[0] == vertex_yeo.shape[0]:
            vertex_isc = _isc
            print(f"[build_kpi_baseline] ISC ceiling loaded "
                  f"({int((_isc > NOISE_CEILING_THRESHOLD).sum())} responsive vertices); "
                  f"reductions restricted to ISC>{NOISE_CEILING_THRESHOLD}.")
        else:
            print(f"[build_kpi_baseline] ISC shape {_isc.shape} != {vertex_yeo.shape}; all-vertex fallback.")
    else:
        print(f"[build_kpi_baseline] {NOISE_CEILING_PATH} missing; all-vertex fallback.")

    seq_len = int(cfg.get("train", {}).get("sequence_length", 200))
    val = MaryDataset(f"/data/{manifest}", "val", sequence_length=seq_len)
    # map each eval window -> its entry index, so we can aggregate to clip level
    win_entry = [ew[0] for ew in val._eval_windows]
    loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_mary)

    # per-entry accumulators of per-KPI means
    entry_kpi = defaultdict(lambda: {k: [] for k in KPI_NAMES})
    wi = 0
    with torch.no_grad():
        for b in loader:
            feats = {k: v.to(device) for k, v in b["features"].items()}
            sidx = torch.zeros(feats[next(iter(feats))].shape[0], dtype=torch.long, device=device)  # subject 0
            verts = model(feats, sidx).float().cpu().numpy()        # (B, T, 20484)
            for n in range(verts.shape[0]):
                mag = np.abs(verts[n]).astype(np.float64)           # (T, 20484)
                per_vertex = mag.mean(axis=0)                        # time-mean per vertex
                net_mean = {}
                for name in yeo7_order:
                    net = vertex_yeo == net_idx[name]
                    if vertex_isc is not None:
                        m = net & (vertex_isc > NOISE_CEILING_THRESHOLD)
                        if m.any():
                            w_isc = vertex_isc[m]                    # ISC-weighted mean
                            net_mean[name] = float((per_vertex[m] * w_isc).sum() / w_isc.sum())
                            continue
                    net_mean[name] = float(per_vertex[net].mean()) if net.any() else 0.0
                ent = win_entry[wi] if wi < len(win_entry) else 0
                for kpi, w in KPI_FROM_NETWORKS.items():
                    wsum = sum(w.values()) or 1.0
                    val_k = sum((ww / wsum) * net_mean.get(nm, 0.0) for nm, ww in w.items())
                    entry_kpi[ent][kpi].append(val_k)
                wi += 1

    # clip-level: mean per entry, then (mean,std) across entries
    baseline = {}
    for kpi in KPI_NAMES:
        per_entry = [float(np.mean(entry_kpi[e][kpi])) for e in entry_kpi if entry_kpi[e][kpi]]
        arr = np.array(per_entry, dtype=np.float64)
        baseline[kpi] = (round(float(arr.mean()), 5), round(float(arr.std()), 5))

    return {"ckpt": ckpt, "n_entries": len(entry_kpi), "n_windows": wi, "baseline": baseline}


@app.local_entrypoint()
def main(ckpt: str = "/data/checkpoints/mary_multi_stable_resp_s13/best.pt",
         manifest: str = "manifest_mary_multi.json"):
    import json
    out = run.remote(ckpt, manifest)
    print("ENTRIES:", out["n_entries"], "WINDOWS:", out["n_windows"])
    print("KPI_BASELINE = {")
    for k, (m, s) in out["baseline"].items():
        print(f'    "{k}":{" " * (36 - len(k))}({m}, {s}),')
    print("}")
