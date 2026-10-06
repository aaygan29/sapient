"""Held-out evaluation for sapient-1. Per spec §5.

Loads a trained checkpoint, runs the test split of data/manifest.json, computes
vertex + Schaefer-1000 parcel Pearson, renders a brain-surface plot of vertex
Pearson on fsaverage5, and dumps eval/results.json.

Modal app: sapient-1-eval (A100-40GB per spec §12).

Usage:
  # Modal:
  modal run eval.py --checkpoint /data/checkpoints/sapient1_llama/best.pt
  # Local:
  python eval.py --checkpoint checkpoints/sapient1_llama/best.pt --local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

APP_NAME = "sapient-1-eval"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "numpy>=1.26,<3",
        "scipy>=1.13",
        "nilearn>=0.10.4",
        "nibabel>=5.2",
        "matplotlib>=3.8",
        "pyyaml>=6",
        "huggingface_hub>=0.25,<2",
    )
    # Mount local code so the remote eval fn can import sapient1 + data.dataset.
    .workdir("/root")
    .add_local_dir(".", remote_path="/root")
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token")]

app = modal.App(APP_NAME)


def main_eval(
    checkpoint_path: str,
    *,
    data_root: str = "data",
    parcellation_path: str | None = None,
    out_dir: str = "eval",
    manifest_name: str = "manifest.json",
) -> dict:
    """Run held-out eval. Returns the metrics dict (also written as JSON)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, "/root")  # the add_local_dir mount (container)
    from sapient1 import (
        SapientConfig, SapientModel,
        vertex_pearson, parcel_pearson, load_parcellation,
    )
    from data.dataset import SapientDataset

    data_root_p = Path(data_root)
    parcellation_path = parcellation_path or str(data_root_p / "parcellate.npz")
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = SapientConfig.from_yaml(cfg)
    model = SapientModel(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  variant: {cfg.get('variant_name')}  |  device: {device}")
    print(f"  trained {ckpt.get('epoch', '?')} epochs  |  val metrics: {ckpt.get('metrics')}")

    # ---- Held-out set ----
    # First-light AV manifests have no Movie10 'test' holdout; fall back to the
    # 'val' split (held-out Friends episodes — stimulus-novel, subject-seen).
    manifest_path = data_root_p / manifest_name
    import json as _json
    splits_present = {e["split"] for e in
                      _json.loads(manifest_path.read_text())["entries"]}
    eval_split = "test" if "test" in splits_present else "val"
    text_dim = cfg["model"].get("d_text", 3072)
    test_ds = SapientDataset(manifest_path, split=eval_split, text_dim=text_dim)
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False,
    )
    print(f"Eval split '{eval_split}': {len(test_ds)} windows")

    # ---- Forward pass over the entire test set ----
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                p = model(batch["video"], batch["audio"], batch["text"], batch["subject"])
            preds.append(p.float().cpu())
            targets.append(batch["fmri"].cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    print(f"Aggregated: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

    # ---- Vertex Pearson ----
    vmetrics = vertex_pearson(pred, target)
    print(f"  {json.dumps(vmetrics)}")

    # ---- Parcel Pearson (Schaefer-1000) ----
    pmetrics: dict = {}
    if Path(parcellation_path).exists():
        M = load_parcellation(parcellation_path)
        pmetrics = parcel_pearson(pred, target, M)
        print(f"  {json.dumps(pmetrics)}")
    else:
        print(f"  ⚠️ parcellation_path missing: {parcellation_path} — skipping parcel metrics")

    # ---- Brain-surface plot (spec §5.3) ----
    figure_path = None
    try:
        figure_path = _render_brain_surface(
            pred.reshape(-1, pred.shape[-1]),
            target.reshape(-1, target.shape[-1]),
            out_dir_p,
        )
    except Exception as e:
        print(f"  ⚠️ surface plot failed: {e!r}; metrics still written")

    # ---- Dump JSON ----
    results = {
        "checkpoint": str(checkpoint_path),
        "variant": cfg.get("variant_name"),
        **vmetrics, **pmetrics,
        "figure": str(figure_path) if figure_path else None,
        "eval_split": eval_split,
        "n_eval_windows": len(test_ds),
    }
    out_path = out_dir_p / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return results


def _render_brain_surface(pred, target, out_dir: Path) -> Path:
    """Render fsaverage5 brain-surface plot of per-vertex Pearson r."""
    import numpy as np
    from scipy.stats import pearsonr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    # Compute vertex Pearson (vectorized).
    p = pred.numpy()
    t = target.numpy()
    pm = p - p.mean(0, keepdims=True)
    tm = t - t.mean(0, keepdims=True)
    num = (pm * tm).sum(0)
    denom = np.sqrt((pm ** 2).sum(0) * (tm ** 2).sum(0)).clip(min=1e-8)
    r = num / denom
    r = np.nan_to_num(r, nan=0.0)

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    lh = r[:10242]
    rh = r[10242:]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                             subplot_kw={"projection": "3d"})
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, lh, hemi="left", view="lateral",
        bg_map=fsaverage.sulc_left, axes=axes[0, 0], title="LH lateral", colorbar=False,
    )
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, lh, hemi="left", view="medial",
        bg_map=fsaverage.sulc_left, axes=axes[0, 1], title="LH medial", colorbar=False,
    )
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, rh, hemi="right", view="lateral",
        bg_map=fsaverage.sulc_right, axes=axes[1, 0], title="RH lateral", colorbar=False,
    )
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, rh, hemi="right", view="medial",
        bg_map=fsaverage.sulc_right, axes=axes[1, 1], title="RH medial", colorbar=True,
    )
    out = out_dir / "brain_surface_pearson.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  brain-surface plot → {out}")
    return out


@app.function(image=image, gpu="A100-40GB", timeout=4 * 60 * 60,
              volumes={"/data": volume}, secrets=secrets)
def eval_remote(checkpoint: str, parcellation: str = "/data/parcellate.npz",
                manifest_name: str = "manifest.json") -> None:
    main_eval(checkpoint, data_root="/data", parcellation_path=parcellation,
              out_dir="/data/eval/sapient1", manifest_name=manifest_name)


@app.local_entrypoint()
def main(
    checkpoint: str,
    parcellation: str = "/data/parcellate.npz",
    manifest: str = "manifest_sapient1.json",
) -> None:
    eval_remote.remote(checkpoint, parcellation, manifest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    if args.local:
        main_eval(args.checkpoint, data_root=args.data_root,
                  parcellation_path=args.parcellation, out_dir=args.out_dir)
    else:
        print("Run via 'modal run eval.py --checkpoint ...' to launch on A100.\n"
              "Use --local to run on the current machine.", file=sys.stderr)
        sys.exit(2)
