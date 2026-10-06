"""Single-file training entry point for sapient-1. Per spec §4.

Optimizer + schedule (spec §4.2):
  AdamW, lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999), grad_clip=1.0,
  OneCycleLR (pct_start=0.1), 15 epochs, batch=8, grad_accum=2 → effective 16,
  bfloat16 mixed precision (A100/H100 native).

Loss: masked_mse_loss (decision #15).
Selection metric: vertex_pearson_mean on the val manifest split.

Runs as a Modal H100-80GB job named `sapient-1-train-llama` per spec §12.
Can also run locally for debugging via `python train.py --config ... --local`.
No Modal job is launched until you `modal run train.py` (or `make train`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import modal

# ---- Modal app definition ----
APP_NAME = "sapient-1-train-llama"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.46.0",
        "numpy>=1.26,<3",
        "scipy>=1.13",
        "nilearn>=0.10.4",
        "nibabel>=5.2",
        "pyyaml>=6",
        "tqdm>=4.66",
        "wandb>=0.17",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
    )
    # Mount the local sapient1 package + data/ + configs/ so the remote train
    # fn can `from sapient1 import ...`, `from data.dataset import ...`, and
    # read configs/<name>.yaml. Without this the image has NO local code →
    # ModuleNotFoundError: No module named 'sapient1'. add_local_* must be LAST.
    .workdir("/root")
    .add_local_dir(".", remote_path="/root")
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")]

app = modal.App(APP_NAME)


# ---- Core training function (importable, runs in either Modal or local) ----

def _move(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(model, loader, optim, scheduler, scaler, cfg, device, *,
                    epoch: int, wandb_run=None):
    import torch
    model.train()
    grad_accum = cfg["train"]["grad_accum_steps"]
    log_every = cfg["train"].get("log_every", 50)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    from sapient1.losses import masked_mse_loss

    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = _move(batch, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(batch["video"], batch["audio"], batch["text"], batch["subject"])
            loss = masked_mse_loss(pred, batch["fmri"], batch["mask"])
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

        if step % log_every == 0:
            loss_val = loss.item() * grad_accum
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"  e{epoch:02d} s{step:05d}  loss={loss_val:.4f}  lr={lr:.2e}  "
                  f"ips={(step+1)/max(elapsed,1e-3):.2f}")
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss_val, "train/lr": lr,
                               "epoch": epoch, "step": step})


def evaluate(model, loader, device, *, epoch: int):
    import torch
    from sapient1.metrics import vertex_pearson
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                p = model(batch["video"], batch["audio"], batch["text"], batch["subject"])
            preds.append(p.float().cpu())
            targets.append(batch["fmri"].cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return vertex_pearson(pred, target)


def main_train(config_path: str, *, data_root: str = "data",
               manifest_name: str = "manifest.json", local: bool = False) -> None:
    """Run end-to-end training. Importable by Modal or by `python train.py`."""
    # All imports here so the Modal image can be built without local torch.
    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import DataLoader

    # Make `sapient1` importable even when run from outside the repo.
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, "/root")  # the add_local_dir mount (container)
    from sapient1 import SapientConfig, SapientModel
    from sapient1.utils import load_config, set_seed
    from data.dataset import SapientDataset

    cfg = load_config(config_path)
    cfg["__source__"] = str(config_path)
    set_seed(cfg["train"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {config_path}  →  variant={cfg.get('variant_name', '?')}")

    # ---- Build datasets ----
    manifest_path = Path(data_root) / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run data/manifest.py first."
        )
    text_dim = cfg["model"].get("d_text", 3072)
    train_ds = SapientDataset(manifest_path, split="train",
                              seed=cfg["train"]["seed"], text_dim=text_dim)
    val_ds   = SapientDataset(manifest_path, split="val", text_dim=text_dim)
    print(f"Train: {len(train_ds)} runs | Val: {len(val_ds)} windows")

    # num_workers=0 + pin_memory=False: the shared-memory collate path crashes
    # ("resize storage that is not resizable") on small sets with mmap-backed
    # tensors. Single-process loading is plenty fast for a first-light run.
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False,
    )

    # ---- Build model ----
    model_cfg = SapientConfig.from_yaml(cfg)
    model = SapientModel(model_cfg).to(device)
    print(f"Trainable params: {model.n_trainable_params():,}")

    # ---- Optimizer + schedule ----
    optim = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=tuple(cfg["train"]["betas"]),
    )
    total_optim_steps = (len(train_loader) * cfg["train"]["epochs"]
                         // cfg["train"]["grad_accum_steps"])
    scheduler = OneCycleLR(
        optim,
        max_lr=float(cfg["train"]["lr"]),
        total_steps=max(total_optim_steps, 1),
        pct_start=float(cfg["train"]["pct_start"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---- W&B ----
    wandb_run = None
    if os.environ.get("WANDB_API_KEY") and not local:
        import wandb
        wandb_run = wandb.init(
            project="sapient",
            name=cfg.get("variant_name", "sapient1"),
            config=cfg,
        )

    # ---- Checkpoint directory ----
    ckpt_root = Path(data_root) / "checkpoints" / cfg.get("variant_name", "sapient1")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    best_metric = -float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{cfg['train']['epochs']} ===")
        train_one_epoch(model, train_loader, optim, scheduler, scaler, cfg,
                        device, epoch=epoch, wandb_run=wandb_run)
        val_metrics = evaluate(model, val_loader, device, epoch=epoch)
        print(f"  val: {json.dumps(val_metrics, indent=None)}")
        if wandb_run is not None:
            wandb_run.log({**{f"val/{k}": v for k, v in val_metrics.items()},
                           "epoch": epoch})

        key = cfg["train"].get("save_best_metric", "vertex_pearson_mean")
        cur = val_metrics.get(key, -float("inf"))
        if cur > best_metric:
            best_metric = cur
            best_path = ckpt_root / f"best_e{epoch:02d}.pt"
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "metrics": val_metrics, "config": cfg,
            }, best_path)
            print(f"  ✓ new best ({key}={cur:.4f}) → {best_path}")

        # Always save the latest, regardless.
        torch.save({
            "epoch": epoch, "state_dict": model.state_dict(),
            "metrics": val_metrics, "config": cfg,
        }, ckpt_root / "latest.pt")

    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nDone. Best {cfg['train'].get('save_best_metric')} = {best_metric:.4f}")


# ---- Modal wrapper ----

@app.function(
    image=image,
    gpu="H100",  # Modal rejects "H100-80GB"
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
    secrets=secrets,
)
def train_remote(config_path: str, manifest_name: str = "manifest.json") -> None:
    main_train(config_path, data_root="/data", manifest_name=manifest_name)


@app.local_entrypoint()
def main(config: str = "configs/sapient1_av.yaml",
         manifest: str = "manifest_sapient1.json") -> None:
    """`modal run train.py --config configs/sapient1_av.yaml`"""
    train_remote.remote(config, manifest)


# ---- Local mode ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-name", default="manifest.json")
    ap.add_argument("--local", action="store_true",
                    help="Run locally instead of launching a Modal job")
    args = ap.parse_args()
    if args.local:
        main_train(args.config, data_root=args.data_root,
                   manifest_name=args.manifest_name, local=True)
    else:
        print("Run via 'modal run train.py --config ...' to launch on H100.\n"
              "Use --local to run on the current machine.", file=sys.stderr)
        sys.exit(2)
