"""Single-file training entry point for Mary (6-stream ORCLE encoder).

Optimizer + schedule (ORCLE-Nano whitepaper §4.5; CONTRACTS §5):
  AdamW, lr=3e-4, weight_decay=0.01, grad_clip=1.0, cosine_with_warmup
  (warmup_steps=1000), max_epochs (config), batch_size=16, bf16 mixed precision,
  early stopping on val/pearson_mean (patience 5).

Loss: composite MSE(1.0) + NegCorr(0.5) + InfoNCE(0.1, τ=0.07)  [mary.losses.composite_loss].
Selection metric: vertex_pearson_mean on the val split.

Runs as Modal H100-80GB job `mary-train`. Local debug: `python train.py --config ... --local`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "mary-train"

_REPO = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "numpy>=1.26,<3",
        "scipy>=1.13",
        "pyyaml>=6",
        "tqdm>=4.66",
        "wandb>=0.17",
        "transformers==4.46.0",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
    )
    # Ship the Mary repo source (mary/, data/, configs/) into the container so
    # the remote function can import them. Modal 1.x does not auto-mount
    # deferred (function-body) imports.
    .add_local_dir(
        str(_REPO), remote_path="/root/maryrepo",
        ignore=["**/__pycache__", "**/.venv", "**/*.pt", "**/*.npy",
                "**/.git", "**/logs/**"],
    )
)
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")]
app = modal.App(APP_NAME)


def _move(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if k == "features":
            out[k] = {s: t.to(device, non_blocking=True) for s, t in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def train_one_epoch(model, loader, optim, scheduler, scaler, cfg, device, *,
                    epoch: int, wandb_run=None, vertex_weight=None):
    import torch
    from mary.losses import composite_loss

    model.train()
    log_every = int(cfg["train"].get("log_every", 20))
    grad_clip = float(cfg["train"].get("gradient_clip_norm", 1.0))
    lw = cfg.get("loss", {})
    lw = dict(mse_weight=float(lw.get("mse_weight", 1.0)),
              negcorr_weight=float(lw.get("negcorr_weight", 0.5)),
              infonce_weight=float(lw.get("infonce_weight", 0.1)),
              infonce_temperature=float(lw.get("infonce_temperature", 0.07)))
    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = _move(batch, device)
        with torch.amp.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = model(batch["features"], batch["subject_idx"])
            loss, terms = composite_loss(pred, batch["fmri"], batch["mask"],
                                         vertex_weight=vertex_weight, **lw)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optim)
        scaler.update()
        scheduler.step()
        optim.zero_grad(set_to_none=True)

        if step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            ips = (step + 1) / max(time.time() - t0, 1e-3)
            print(f"  e{epoch:02d} s{step:05d}  loss={loss.item():.4f} "
                  f"(mse={terms['mse']:.4f} negcorr={terms['negcorr']:.4f} "
                  f"infonce={terms['infonce']:.4f})  lr={lr:.2e}  ips={ips:.2f}")
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(), **{f"train/{k}": v for k, v in terms.items()},
                               "train/lr": lr, "epoch": epoch, "step": step})


def evaluate(model, loader, device, responsive_mask=None):
    import torch
    from mary.metrics import vertex_pearson
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                p = model(batch["features"], batch["subject_idx"])
            preds.append(p.float().cpu())
            targets.append(batch["fmri"].cpu())
    if not preds:
        return {"vertex_pearson_mean": float("nan")}
    # responsive_mask lives on `device`; metrics run on CPU tensors, so pass a CPU copy.
    rmask = responsive_mask.cpu() if responsive_mask is not None else None
    return vertex_pearson(torch.cat(preds), torch.cat(targets), responsive_mask=rmask)


def main_train(config_path: str, *, data_root: str = "/data", local: bool = False,
               seed: int | None = None) -> None:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).parent))
    from mary.model import MaryConfig, MaryModel
    from mary.utils import load_config, set_seed
    from data.dataset import MaryDataset, collate_mary

    cfg = load_config(config_path)
    seeds = cfg.get("seeds", [13])
    active_seed = int(seed) if seed is not None else int(seeds[0])
    set_seed(active_seed)
    # For ensemble runs, give each seed its own checkpoint dir so they don't clobber.
    if seed is not None:
        cfg["variant_name"] = f"{cfg.get('variant_name', 'mary')}_s{active_seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["train"]
    print(f"Device: {device} | variant={cfg.get('variant_name')} | config={config_path}")

    manifest_path = Path(data_root) / cfg.get("manifest_file", "manifest_mary.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found — run data/manifest.py first.")
    seq_len = int(tcfg.get("sequence_length", 200))
    train_ds = MaryDataset(manifest_path, split="train", sequence_length=seq_len,
                           seed=active_seed, oversample=int(tcfg.get("train_oversample", 1)))
    val_ds = MaryDataset(manifest_path, split="val", sequence_length=seq_len)
    print(f"Train entries: {len(train_ds)} | Val windows: {len(val_ds)} | n_subjects={train_ds.n_subjects}")

    bs = int(tcfg.get("batch_size", 16))
    if tcfg.get("balanced_sampling", False):
        from torch.utils.data import WeightedRandomSampler
        w = train_ds.sample_weights()
        sampler = WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=4,
                                  pin_memory=True, drop_last=False, collate_fn=collate_mary)
        print(f"  dataset-balanced sampling ON ({len(w)} weighted samples)")
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=False, collate_fn=collate_mary)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4,
                            pin_memory=True, collate_fn=collate_mary)

    # n_subjects must match the manifest.
    cfg.setdefault("model", {})["n_subjects"] = train_ds.n_subjects
    model = MaryModel(MaryConfig.from_yaml(cfg)).to(device)
    print(f"Trainable params: {model.n_trainable_params():,}")

    # --- Responsive-vertex weighting (per-vertex ISC noise ceiling) ---
    # ~90% of the 20,484 fsaverage5 vertices are noise (ISC≈0) and dilute the
    # whole-brain signal ~10:1. When the ISC map is present we (a) weight the loss
    # toward responsive vertices and (b) report/select on a responsive mask
    # (ISC > 0.05). Every load is guarded — a missing map degrades gracefully to
    # the original whole-brain behavior.
    import numpy as np
    isc_path = Path(data_root) / cfg.get("noise_ceiling_file", "noise_ceiling_isc.npy")
    resp_threshold = float(cfg.get("loss", {}).get("responsive_threshold", 0.05))
    vertex_weight = None
    responsive_mask = None
    # select_on: "responsive" (default when ISC available) selects best.pt on the
    # responsive-vertex metric; "whole" forces the original whole-brain selection.
    select_on = str(cfg.get("select_on", "responsive")).lower()
    vw_flag = str(cfg.get("loss", {}).get("vertex_weight", "isc")).lower()
    if isc_path.exists():
        try:
            isc = np.load(isc_path).astype("float32")            # (20484,)
            if isc.shape[0] != cfg["model"]["n_vertices"]:
                print(f"  WARNING: ISC map has {isc.shape[0]} vertices, expected "
                      f"{cfg['model']['n_vertices']}; ignoring ISC map.")
            else:
                isc_t = torch.from_numpy(isc).to(device)
                responsive_mask = (isc_t > resp_threshold)
                n_resp = int(responsive_mask.sum().item())
                if vw_flag in ("isc", "true", "1", "on"):
                    vertex_weight = isc_t.clamp(min=0.0)
                    print(f"  ✓ ISC vertex_weight ON ({n_resp} responsive vertices "
                          f"@ ISC>{resp_threshold}); loss weighted by ISC ceiling.")
                else:
                    print(f"  ISC map loaded; vertex_weight flag={vw_flag!r} → loss NOT "
                          f"weighted ({n_resp} responsive vertices for metric only).")
        except Exception as e:
            print(f"  WARNING: failed to load ISC map ({e}); whole-brain fallback.")
    else:
        print(f"  WARNING: {isc_path} not found → whole-brain fallback "
              f"(no responsive weighting/selection).")
    if select_on == "responsive" and responsive_mask is None:
        print("  select_on=responsive requested but no ISC map → falling back to "
              "select_on=whole.")
        select_on = "whole"

    optim = AdamW(model.parameters(), lr=float(tcfg.get("learning_rate", 3e-4)),
                  weight_decay=float(tcfg.get("weight_decay", 0.01)))
    epochs = int(tcfg.get("max_epochs", 12))
    total_steps = max(len(train_loader) * epochs, 1)
    warmup = int(tcfg.get("warmup_steps", min(1000, total_steps // 10)))
    warmup = min(warmup, max(total_steps - 1, 1))   # warmup must be < total
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optim, warmup, total_steps)
    except Exception:
        from torch.optim.lr_scheduler import OneCycleLR
        pct = min(max(warmup / max(total_steps, 1), 1e-3), 0.5)
        scheduler = OneCycleLR(optim, max_lr=float(tcfg.get("learning_rate", 3e-4)),
                               total_steps=total_steps, pct_start=pct)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    wandb_run = None
    if os.environ.get("WANDB_API_KEY") and not local:
        try:
            import wandb
            wandb_run = wandb.init(project="sapient", name=cfg.get("variant_name", "mary"), config=cfg)
        except Exception as e:
            print(f"wandb init failed ({e}); continuing without.")

    ckpt_root = Path(data_root) / "checkpoints" / cfg.get("variant_name", "mary")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Resume from a prior checkpoint if present. A dropped laptop connection can
    # cancel the Modal job mid-run; latest.pt is written every epoch, so a relaunch
    # continues from the last completed epoch instead of restarting at 0.
    # (Weights-only warm start: optimizer/scheduler reinit — minor, acceptable.)
    # The metric we early-stop / select best.pt on. responsive → restricted to
    # ISC>thr vertices; whole → original whole-brain mean.
    sel_metric = ("vertex_pearson_mean_responsive" if select_on == "responsive"
                  else "vertex_pearson_mean")
    print(f"  best.pt selection metric: {sel_metric} (select_on={select_on})")

    start_epoch, best, since_best = 0, -float("inf"), 0
    resume_path = ckpt_root / "latest.pt"
    if resume_path.exists():
        try:
            prev = torch.load(resume_path, map_location=device)
            model.load_state_dict(prev["state_dict"])
            start_epoch = int(prev.get("epoch", -1)) + 1
            # Resume best on the SAME selection metric (fall back to whole-brain if
            # the prior run logged only that).
            prev_m = prev.get("metrics", {})
            best = float(prev_m.get(sel_metric,
                                    prev_m.get("vertex_pearson_mean", -float("inf"))))
            print(f"  ↻ RESUMED from {resume_path} at epoch {start_epoch} (prev best={best:.4f})")
        except Exception as e:
            print(f"  resume failed ({e}); starting fresh.")
    patience = int(tcfg.get("early_stopping_patience", 5))
    history = []
    for epoch in range(start_epoch, epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        train_one_epoch(model, train_loader, optim, scheduler, scaler, cfg, device,
                        epoch=epoch, wandb_run=wandb_run, vertex_weight=vertex_weight)
        val_metrics = evaluate(model, val_loader, device, responsive_mask=responsive_mask)
        print(f"  val: {json.dumps(val_metrics)}")
        history.append({"epoch": epoch, **val_metrics})
        if wandb_run is not None:
            wandb_run.log({**{f"val/{k}": v for k, v in val_metrics.items()}, "epoch": epoch})

        # Select on the responsive metric when available (the real signal); the
        # whole-brain mean is still logged in val_metrics for comparison.
        cur = val_metrics.get(sel_metric, -float("inf"))
        if cur != cur:  # NaN guard (e.g. empty responsive mask)
            cur = -float("inf")
        ckpt = {"epoch": epoch, "state_dict": model.state_dict(),
                "metrics": val_metrics, "config": cfg}
        torch.save(ckpt, ckpt_root / "latest.pt")
        if cur > best:
            best, since_best = cur, 0
            torch.save(ckpt, ckpt_root / "best.pt")
            print(f"  ✓ new best {sel_metric}={cur:.4f} → {ckpt_root/'best.pt'} "
                  f"(whole-brain={val_metrics.get('vertex_pearson_mean', float('nan')):.4f})")
        else:
            since_best += 1
            if since_best >= patience:
                print(f"  early stop (no val improvement in {patience} epochs)")
                break
        if "_volume_commit" in cfg:  # noop hook
            pass
        try:
            volume.commit()
        except Exception:
            pass

    (ckpt_root / "history.json").write_text(json.dumps(history, indent=2))
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nDone. Best {sel_metric} = {best:.4f}. Checkpoints in {ckpt_root}")


@app.function(image=image, gpu="H100", timeout=18 * 60 * 60,
              volumes={"/data": volume}, secrets=secrets)
def train_remote(config_path: str, seed: int = -1) -> None:
    # The repo is mounted at /root/maryrepo (image.add_local_dir). Make `mary`
    # and `data` importable and resolve relative config paths from there.
    os.chdir("/root/maryrepo")
    sys.path.insert(0, "/root/maryrepo")
    main_train(config_path, data_root="/data", seed=(None if seed < 0 else seed))


@app.local_entrypoint()
def main(config: str = "configs/mary_nano.yaml", seed: int = -1) -> None:
    """`modal run --detach train.py --config configs/mary_multi.yaml --seed 13`"""
    train_remote.remote(config, seed)


@app.local_entrypoint()
def launch(config: str = "configs/mary_full.yaml", seeds: str = "13,17,23") -> None:
    """Fire-and-forget ensemble launch via spawn — robust to client exit / wifi drops.

    Unlike `.remote()` (which keeps a client attached to a cancellable input, so a
    SIGTERM/disconnect cancels the job), `.spawn()` submits each run as an independent
    function call and returns immediately. Run with --detach so the app outlives this
    entrypoint; the client then exits cleanly in seconds with no job to cancel:
      `modal run --detach train.py::launch --config configs/mary_full.yaml --seeds 13,17,23`
    """
    ids = [int(x) for x in seeds.split(",") if x.strip()]
    for s in ids:
        call = train_remote.spawn(config, s)
        print(f"spawned seed {s} -> call {call.object_id}")
    print(f"LAUNCHED {len(ids)} seeds (detached, fire-and-forget).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    if args.local:
        main_train(args.config, data_root=args.data_root, local=True)
    else:
        print("Use `modal run train.py --config ...` for H100, or --local.", file=sys.stderr)
        sys.exit(2)
