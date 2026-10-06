"""Single-file training entry point for sapient-2. Per spec §4 + §4.6.

Two Modal app definitions, one per training lineage:
  • sapient-2-train-scratch-llama  — random init on ds004996 + ds001740
  • sapient-2-train-ft-llama       — warm-started from sapient-1-llama,
                                     lr=5e-5, subject_embed dropped on load.

Both apps share the same training loop (`main_train`). They differ only in
how the model is initialized — decided by the `init.from_pretrained` field
in the YAML config. The Makefile picks the right entry point per target:
  • make train-scratch → modal run train.py::main_scratch --config configs/sapient2_scratch_llama.yaml
  • make train-ft      → modal run train.py::main_ft       --config configs/sapient2_ft_llama.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import modal

# ---- Two Modal apps, one per lineage ----
APP_NAME_SCRATCH = "sapient-2-train-scratch-llama"
APP_NAME_FT      = "sapient-2-train-ft-llama"

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
    # Mount the local sapient2 package + data/ + configs/ into the container so the
    # remote train fn can `from sapient2 import ...`, `from data.dataset import ...`,
    # and read configs/<name>.yaml. Without this the image had NO local code →
    # ModuleNotFoundError: No module named 'sapient2'. add_local_* must be LAST.
    .workdir("/root")
    .add_local_dir(".", remote_path="/root")
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")]

app_scratch = modal.App(APP_NAME_SCRATCH)
app_ft      = modal.App(APP_NAME_FT)


# ---- Shared training core ----

def _move(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(model, loader, optim, scheduler, scaler, cfg, device, *,
                    epoch: int, wandb_run=None):
    import torch
    model.train()
    grad_accum = cfg["train"]["grad_accum_steps"]
    log_every = cfg["train"].get("log_every", 50)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    from sapient2.losses import masked_mse_loss

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
    from sapient2.metrics import vertex_pearson
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


def _load_from_pretrained(model, ref: str, drop_keys: list[str]) -> None:
    """Load sapient-1 weights into a sapient-2 model. Decision #12: drop
    `subject_embed` because the subject set differs."""
    import torch
    from huggingface_hub import hf_hub_download
    # Accept either a local .pt path or an HF repo id.
    if Path(ref).exists():
        state = torch.load(ref, map_location="cpu", weights_only=False)
    else:
        # Best-effort: try the canonical file names in the HF repo.
        for fname in ("best.pt", "latest.pt", "model.safetensors"):
            try:
                local = hf_hub_download(repo_id=ref, filename=fname)
                state = torch.load(local, map_location="cpu", weights_only=False)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"Could not find a weights file in {ref}")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    for k in list(state.keys()):
        if any(d in k for d in drop_keys):
            print(f"  drop on load: {k}")
            del state[k]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {len(missing)}, unexpected: {len(unexpected)}")


def main_train(config_path: str, *, data_root: str = "data", local: bool = False) -> None:
    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, "/root")  # the add_local_dir mount (container) — robust to __file__ location
    from sapient2 import SapientConfig, SapientModel
    from sapient2.utils import load_config, set_seed
    from data.dataset import SapientDataset

    cfg = load_config(config_path)
    cfg["__source__"] = str(config_path)
    set_seed(cfg["train"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = cfg.get("variant_name", "sapient2")
    print(f"Device: {device}  |  variant: {variant}  |  config: {config_path}")

    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run data/manifest.py first."
        )
    text_dim = cfg["model"].get("d_text", 3072)
    train_ds = SapientDataset(manifest_path, split="train",
                              seed=cfg["train"]["seed"], text_dim=text_dim)
    val_ds   = SapientDataset(manifest_path, split="val", text_dim=text_dim)
    print(f"Train: {len(train_ds)} runs | Val: {len(val_ds)} windows")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False,
    )

    model_cfg = SapientConfig.from_yaml(cfg)
    model = SapientModel(model_cfg).to(device)
    print(f"Trainable params: {model.n_trainable_params():,}")

    # Warm-start (decision #11, #12 — sapient-2-ft only)
    init_ref = (cfg.get("init") or {}).get("from_pretrained")
    if init_ref:
        drop_keys = (cfg.get("init") or {}).get("drop_keys_on_load", ["subject_embed"])
        print(f"Loading from {init_ref}; dropping {drop_keys}")
        _load_from_pretrained(model, init_ref, drop_keys)

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

    wandb_run = None
    if os.environ.get("WANDB_API_KEY") and not local:
        import wandb
        wandb_run = wandb.init(project="sapient", name=variant, config=cfg)

    ckpt_root = Path(data_root) / "checkpoints" / variant
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

        torch.save({
            "epoch": epoch, "state_dict": model.state_dict(),
            "metrics": val_metrics, "config": cfg,
        }, ckpt_root / "latest.pt")

    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nDone. Best {cfg['train'].get('save_best_metric')} = {best_metric:.4f}")


# ---- Modal wrappers — two apps, two remotes, two local_entrypoints ----

@app_scratch.function(image=image, gpu="H100", timeout=24 * 60 * 60,
                      volumes={"/data": volume}, secrets=secrets)
def train_scratch_remote(config_path: str) -> None:
    main_train(config_path, data_root="/data")


@app_scratch.local_entrypoint()
def main_scratch(config: str = "configs/sapient2_scratch_llama.yaml") -> None:
    """`modal run train.py::main_scratch --config configs/sapient2_scratch_llama.yaml`"""
    train_scratch_remote.remote(config)


@app_ft.function(image=image, gpu="H100", timeout=24 * 60 * 60,
                 volumes={"/data": volume}, secrets=secrets)
def train_ft_remote(config_path: str) -> None:
    main_train(config_path, data_root="/data")


@app_ft.local_entrypoint()
def main_ft(config: str = "configs/sapient2_ft_llama.yaml") -> None:
    """`modal run train.py::main_ft --config configs/sapient2_ft_llama.yaml`"""
    train_ft_remote.remote(config)


# ---- Local mode ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    if args.local:
        main_train(args.config, data_root=args.data_root, local=True)
    else:
        print("Run via 'modal run train.py::main_{scratch,ft} --config ...'.\n"
              "Use --local to run on the current machine.", file=sys.stderr)
        sys.exit(2)
