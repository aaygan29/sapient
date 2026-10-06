"""Diagnose WHY Mary's visual (slowfast) stream contributes ~0.1% to the output.

Runs on Modal against the SERVED checkpoint (mary_multi_stable_resp_s13/best.pt) and
the multi-dataset manifest on the sapient-data volume. Read-only.

Measures, for the served model:
  A. Per-stream PREDICTION contribution: ablate (zero+mask) each stream on the
     val windows that HAVE that stream, correlate the per-vertex prediction with the
     full-streams prediction. Low r = stream matters a lot; r≈1 = stream ignored.
  B. Per-stream ACCURACY contribution: val responsive-vertex pearson with the full
     stream set vs with each stream ablated, ON the entries that have that stream.
     This answers the make-or-break question: does dropping visual HURT accuracy
     (visual carries brain signal) or NOT (visual is noise for this content)?
  C. Encoded-feature variance per stream (is slowfast low-variance at the fusion input?)
  D. Fusion attention / modality-embed norms (is the fusion structurally ignoring it?)

Usage:
  modal run scripts/diagnose_visual_weight.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

_REPO = Path(__file__).parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1", "numpy>=1.26,<3", "scipy>=1.13", "pyyaml>=6", "tqdm>=4.66",
        "transformers==4.46.0", "huggingface_hub>=0.25,<2", "safetensors>=0.4",
    )
    .add_local_dir(
        str(_REPO), remote_path="/root/maryrepo",
        ignore=["**/__pycache__", "**/.venv", "**/*.pt", "**/*.npy",
                "**/.git", "**/logs/**"],
    )
)
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token")]
app = modal.App("mary-diagnose-visual")

CKPT = "/data/checkpoints/mary_multi_stable_resp_s13/best.pt"
VISUAL = {"slowfast", "qwen_vl", "got_ocr"}
AUDIO = {"beats", "whisper"}
TEXT = {"qwen_ctx"}


@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={"/data": volume}, secrets=secrets)
def diagnose() -> dict:
    import os, sys
    os.chdir("/root/maryrepo")
    sys.path.insert(0, "/root/maryrepo")
    import numpy as np
    import torch
    from mary.model import MaryConfig, MaryModel
    from mary.metrics import vertex_pearson
    from data.dataset import MaryDataset, collate_mary
    from torch.utils.data import DataLoader

    device = torch.device("cuda")
    ckpt = torch.load(CKPT, map_location=device)
    cfg = ckpt["config"]
    cfg.setdefault("model", {})
    # n_subjects from checkpoint config
    model = MaryModel(MaryConfig.from_yaml(cfg)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {CKPT} epoch={ckpt.get('epoch')} streams={model.stream_order}")

    manifest = "/data/" + cfg.get("manifest_file", "manifest_mary_multi.json")
    # ISC responsive mask
    isc_path = Path("/data/noise_ceiling_isc.npy")
    responsive_mask = None
    if isc_path.exists():
        isc = np.load(isc_path).astype("float32")
        responsive_mask = torch.from_numpy(isc > 0.05)
        print(f"responsive vertices: {int(responsive_mask.sum())} / {isc.shape[0]}")

    val_ds = MaryDataset(manifest, split="val", sequence_length=200)
    print(f"val windows: {len(val_ds)}  n_subjects={val_ds.n_subjects}")

    # Which streams appear in val entries, and how many windows have slowfast.
    from collections import Counter
    stream_window_counts = Counter()
    for (ei, _start) in val_ds._eval_windows:
        for s in val_ds.entries[ei].get("feature_paths", {}):
            stream_window_counts[s] += 1
    print("val windows per stream:", dict(stream_window_counts))

    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_mary)

    streams = list(model.stream_order)

    def run(batch_features, subj, drop=None):
        feats = {}
        for s, t in batch_features.items():
            if drop is not None and s == drop:
                continue
            feats[s] = t
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(feats, subj).float()

    # Accumulators
    # Prediction-divergence: only count a sample for stream s if s was actually
    # present (non-zero) for that sample, else ablating it is a no-op.
    pred_corr_sum = {s: 0.0 for s in streams}
    pred_corr_n = {s: 0 for s in streams}
    enc_var_sum = {s: 0.0 for s in streams}
    enc_var_n = {s: 0 for s in streams}

    # For accuracy: collect preds (full + per-ablation) and targets, but ONLY over
    # samples/entries that contain the ablated stream. We bucket by stream.
    full_preds, full_targets, full_masks = [], [], []
    # store per-stream ablated preds aligned to the same sample order, plus a
    # boolean "stream present in this sample".
    ablate_preds = {s: [] for s in streams}
    present_flags = {s: [] for s in streams}

    n_batches = 0
    for batch in loader:
        feats = {s: t.to(device) for s, t in batch["features"].items()}
        subj = batch["subject_idx"].to(device)
        fmri = batch["fmri"]
        mask = batch["mask"]

        # presence: a sample has stream s if its feature tensor is non-zero
        # (collate zero-fills missing streams).
        pres = {}
        for s in streams:
            if s in feats:
                nz = feats[s].abs().sum(dim=(1, 2)) > 0  # (B,)
                pres[s] = nz.cpu()
            else:
                pres[s] = torch.zeros(subj.shape[0], dtype=torch.bool)

        full = run(feats, subj)  # (B,T,V)
        full_preds.append(full.cpu())
        full_targets.append(fmri)
        full_masks.append(mask)

        # encoded-feature variance per present stream (fusion input scale)
        with torch.no_grad():
            for s in streams:
                if s in feats and pres[s].any():
                    enc = model.stream_encoders[s](feats[s])  # (B,T,D)
                    idx = pres[s].to(device)
                    v = enc[idx].float().var().item()
                    enc_var_sum[s] += v * int(idx.sum())
                    enc_var_n[s] += int(idx.sum())

        for s in streams:
            present_flags[s].append(pres[s])
            if s in feats and pres[s].any():
                ab = run(feats, subj, drop=s)
                ablate_preds[s].append(ab.cpu())
                # prediction divergence per-sample over present samples
                idx = pres[s]
                a = full[idx.to(device)].float()
                b = ab[idx.to(device)].float()
                # flatten time+vertex per sample, correlate
                for i in range(a.shape[0]):
                    fa = a[i].reshape(-1)
                    fb = b[i].reshape(-1)
                    fa = fa - fa.mean(); fb = fb - fb.mean()
                    denom = (fa.norm() * fb.norm()).clamp(min=1e-8)
                    r = float((fa * fb).sum() / denom)
                    pred_corr_sum[s] += r
                    pred_corr_n[s] += 1
            else:
                ablate_preds[s].append(None)
        n_batches += 1

    # ---- A. prediction divergence ----
    pred_divergence = {}
    for s in streams:
        if pred_corr_n[s] > 0:
            r = pred_corr_sum[s] / pred_corr_n[s]
            pred_divergence[s] = {"mean_corr_full_vs_ablated": r,
                                  "n_samples": pred_corr_n[s],
                                  "pct_output_change": (1 - r) * 100}

    # ---- C. encoded variance ----
    enc_var = {s: (enc_var_sum[s] / enc_var_n[s]) for s in streams if enc_var_n[s] > 0}

    # ---- B. accuracy with vs without each stream (on entries that have it) ----
    full_p = torch.cat(full_preds)      # (N,T,V)
    full_t = torch.cat(full_targets)
    rmask = responsive_mask
    base = vertex_pearson(full_p, full_t, responsive_mask=rmask)

    accuracy = {"full_all_val": base}
    for s in streams:
        flags = torch.cat(present_flags[s])  # (N,)
        if flags.sum() == 0:
            continue
        # gather ablated preds for present samples; full preds for same samples
        ab_list = []
        sample_ptr = 0
        # reconstruct per-batch sizes
        # easier: rebuild by iterating stored ablate_preds list aligned to batches
        # We need ablated preds for present samples only.
        # ablate_preds[s][b] is (n_present_in_batch, T, V) or None
        present_full = full_p[flags]
        present_tgt = full_t[flags]
        for b_ab in ablate_preds[s]:
            if b_ab is not None:
                ab_list.append(b_ab)
        if not ab_list:
            continue
        ab_cat = torch.cat(ab_list)  # (n_present_total, T, V)
        # sanity: counts match
        if ab_cat.shape[0] != present_full.shape[0]:
            print(f"WARN {s}: ablate count {ab_cat.shape[0]} != present {present_full.shape[0]}")
            m = min(ab_cat.shape[0], present_full.shape[0])
            ab_cat = ab_cat[:m]; present_full = present_full[:m]; present_tgt = present_tgt[:m]
        with_s = vertex_pearson(present_full, present_tgt, responsive_mask=rmask)
        without_s = vertex_pearson(ab_cat, present_tgt, responsive_mask=rmask)
        key = "vertex_pearson_mean_responsive" if rmask is not None else "vertex_pearson_mean"
        accuracy[s] = {
            "n_windows_with_stream": int(flags.sum()),
            "acc_with_stream": with_s.get(key, with_s.get("vertex_pearson_mean")),
            "acc_without_stream": without_s.get(key, without_s.get("vertex_pearson_mean")),
            "delta_acc_from_dropping": (without_s.get(key, without_s.get("vertex_pearson_mean"))
                                        - with_s.get(key, with_s.get("vertex_pearson_mean"))),
        }

    # ---- D. fusion structural inspection ----
    me = model.fusion.modality_embed.detach().float()  # (S, D)
    me_norms = {streams[i]: float(me[i].norm()) for i in range(len(streams))}

    out = {
        "checkpoint": CKPT,
        "epoch": ckpt.get("epoch"),
        "streams": streams,
        "val_windows_per_stream": dict(stream_window_counts),
        "A_prediction_divergence": pred_divergence,
        "B_accuracy": accuracy,
        "C_encoded_feature_variance": enc_var,
        "D_modality_embed_norms": me_norms,
    }
    print(json.dumps(out, indent=2, default=str))
    return out


@app.local_entrypoint()
def main():
    res = diagnose.remote()
    Path("/root/maryrepo/_diag_out.json")
    print("=== DIAGNOSIS COMPLETE ===")
    print(json.dumps(res, indent=2, default=str))
