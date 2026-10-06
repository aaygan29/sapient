"""Validate MaryDataset end-to-end on the volume + a MaryModel forward pass.

modal run scripts/validate_dataset.py
"""

from __future__ import annotations

import modal

APP_NAME = "mary-validate-dataset"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26,<3", "torch==2.4.1")
    .add_local_dir(
        "/Users/robertgutierrez/Desktop/sapient-models/mary",
        remote_path="/root/mary_pkg",
    )
)
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(image=image, cpu=2.0, memory=8192, timeout=20 * 60,
              volumes={"/data": volume})
def run() -> dict:
    import sys
    sys.path.insert(0, "/root/mary_pkg")
    import torch
    from data.dataset import MaryDataset, collate_mary

    out = {}
    for split in ("train", "val", "test"):
        try:
            ds = MaryDataset("/data/manifest_mary.json", split, sequence_length=200)
        except Exception as e:
            out[split] = {"error": f"init: {e}"}
            continue
        info = {"len": len(ds)}
        if len(ds) > 0:
            item = ds[0]
            info["fmri_shape"] = list(item["fmri"].shape)
            info["mask_sum"] = float(item["mask"].sum())
            info["subject_idx"] = int(item["subject_idx"])
            info["streams"] = {k: list(v.shape) for k, v in item["features"].items()}
            info["fmri_finite"] = bool(torch.isfinite(item["fmri"]).all())
        out[split] = info

    # collate + MaryModel forward (only if torch model imports cleanly)
    try:
        ds = MaryDataset("/data/manifest_mary.json", "test", sequence_length=200)
        if len(ds) >= 2:
            batch = collate_mary([ds[0], ds[1]])
            out["collate"] = {
                "fmri": list(batch["fmri"].shape),
                "subject_idx": list(batch["subject_idx"].shape),
                "streams": {k: list(v.shape) for k, v in batch["features"].items()},
            }
            from mary.model import MaryConfig, MaryModel
            cfg = MaryConfig(n_subjects=ds.n_subjects)
            model = MaryModel(cfg).eval()
            with torch.no_grad():
                pred = model(batch["features"], batch["subject_idx"])
            out["model_forward"] = {
                "pred_shape": list(pred.shape),
                "pred_finite": bool(torch.isfinite(pred).all()),
                "n_params": model.n_trainable_params(),
            }
    except Exception as e:
        import traceback
        out["model_forward"] = {"error": str(e), "tb": traceback.format_exc()[-1500:]}
    return out


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(run.remote(), indent=2))
