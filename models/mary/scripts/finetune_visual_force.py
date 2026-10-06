"""SHORT hypothesis-test fine-tune: can Mary's VISUAL stream learn to predict the
brain if we FORCE it to rely on visual?

The served model ignores visual content (swap-test delta ~0.3%) — but that could be a
learned shortcut (audio was easier) rather than visual being non-predictive. This run
tests the hypothesis cheaply:

  - Start from the served checkpoint (mary_multi_stable_resp_s13/best.pt).
  - Train ONLY on video-bearing windows (entries with slowfast).
  - Per-stream dropout heavily drops AUDIO+TEXT (p_audio_drop) so the model is
    frequently forced to predict from visual alone — and ALSO trains visual-only
    forward passes directly. This breaks the audio shortcut.
  - Short: a few hundred steps. New checkpoint, never touches the served one.

At the end it measures on the VIDEO val windows:
  - visual-only accuracy (responsive pearson) — can visual alone predict the brain?
  - swap-test delta for slowfast content — does visual content now carry signal?

If visual-only acc stays ~0 and the swap delta stays ~0 even after forcing, visual
genuinely doesn't predict the brain for this content (honest negative result).

Usage:
  modal run --detach scripts/finetune_visual_force.py --steps 400 --p-audio-drop 0.5
"""
from __future__ import annotations
import json, time
from pathlib import Path
import modal

_REPO = Path(__file__).parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "scipy>=1.13", "pyyaml>=6",
                 "tqdm>=4.66", "transformers==4.46.0", "huggingface_hub>=0.25,<2",
                 "safetensors>=0.4")
    .add_local_dir(str(_REPO), remote_path="/root/maryrepo",
                   ignore=["**/__pycache__", "**/.venv", "**/*.pt", "**/*.npy",
                           "**/.git", "**/logs/**"])
)
volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token")]
app = modal.App("mary-finetune-visual")

SRC_CKPT = "/data/checkpoints/mary_multi_stable_resp_s13/best.pt"
OUT_DIR = "/data/checkpoints/mary_visual_force_v1"
AUDIO_TEXT = ["beats", "whisper", "qwen_ctx"]
VISUAL = ["slowfast", "qwen_vl", "got_ocr"]


@app.function(image=image, gpu="H100", timeout=2 * 60 * 60,
              volumes={"/data": volume}, secrets=secrets)
def finetune(steps: int = 400, p_audio_drop: float = 0.5, lr: float = 1e-4,
             visual_only_frac: float = 0.35, batch_size: int = 8) -> dict:
    import os, sys
    os.chdir("/root/maryrepo"); sys.path.insert(0, "/root/maryrepo")
    import numpy as np, torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from mary.model import MaryConfig, MaryModel
    from mary.losses import composite_loss
    from mary.metrics import vertex_pearson
    from data.dataset import MaryDataset, collate_mary

    device = torch.device("cuda")
    ckpt = torch.load(SRC_CKPT, map_location=device)
    cfg = ckpt["config"]
    model = MaryModel(MaryConfig.from_yaml(cfg)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"loaded {SRC_CKPT} epoch={ckpt.get('epoch')}")

    isc = np.load("/data/noise_ceiling_isc.npy").astype("float32")
    isc_t = torch.from_numpy(isc).to(device)
    rmask = (isc_t > 0.05)
    vertex_weight = isc_t.clamp(min=0.0)
    key = "vertex_pearson_mean_responsive"

    manifest = "/data/" + cfg.get("manifest_file", "manifest_mary_multi.json")
    # Restrict BOTH train and val to video-bearing entries.
    train_ds = MaryDataset(manifest, split="train", sequence_length=200,
                           seed=13, oversample=20)
    train_ds.entries = [e for e in train_ds.entries if "slowfast" in e.get("feature_paths", {})]
    val_ds = MaryDataset(manifest, split="val", sequence_length=200)
    val_ds.entries = [e for e in val_ds.entries if "slowfast" in e.get("feature_paths", {})]
    val_ds._eval_windows = val_ds._build_eval_windows()
    print(f"video train entries: {len(train_ds.entries)} | video val windows: {len(val_ds._eval_windows)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_mary, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_mary)

    lw = cfg.get("loss", {})
    lw = dict(mse_weight=float(lw.get("mse_weight", 0.2)),
              negcorr_weight=float(lw.get("negcorr_weight", 1.0)),
              infonce_weight=float(lw.get("infonce_weight", 0.3)),
              infonce_temperature=float(lw.get("infonce_temperature", 0.07)))

    # Disable the model's internal random modality dropout — we drive stream
    # presence explicitly below (forced audio/text drop + visual-only passes).
    model.cfg.modality_dropout = 0.0
    optim = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(13)

    def eval_visual(model):
        model.eval()
        # visual-only accuracy + swap-test on val video windows
        items = [val_ds[i] for i in range(len(val_ds))]
        items = [it for it in items if "slowfast" in it["features"]]
        batch = collate_mary(items)
        feats = {s: t.to(device) for s, t in batch["features"].items()}
        subj = batch["subject_idx"].to(device)
        tgt = batch["fmri"]
        B = subj.shape[0]
        def acc(fd):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p = model(fd, subj).float().cpu()
            return vertex_pearson(p, tgt, responsive_mask=rmask.cpu())[key]
        base = acc(feats)
        vonly = {s: feats[s] for s in VISUAL if s in feats}
        v_only_acc = acc(vonly) if vonly else None
        aonly = {s: feats[s] for s in AUDIO_TEXT if s in feats}
        a_only_acc = acc(aonly) if aonly else None
        perm = torch.tensor([(i + B // 2) % B for i in range(B)])
        fsw = dict(feats)
        for s in VISUAL:
            if s in fsw: fsw[s] = fsw[s][perm].contiguous()
        swap_visual = acc(fsw)
        return {"base_all": base, "visual_only": v_only_acc, "audio_only": a_only_acc,
                "swap_all_visual": swap_visual, "delta_swap_visual": swap_visual - base}

    print("PRE-finetune eval:", json.dumps(eval_visual(model), default=str))

    model.train()
    t0 = time.time()
    step = 0
    log = []
    while step < steps:
        for batch in train_loader:
            if step >= steps: break
            feats = {s: t.to(device) for s, t in batch["features"].items()}
            subj = batch["subject_idx"].to(device)
            fmri = batch["fmri"].to(device)
            mask = batch["mask"].to(device)
            B = subj.shape[0]

            # Build the per-sample stream set:
            #  - with prob visual_only_frac, drop ALL audio/text (visual-only sample)
            #  - else, drop each audio/text stream independently w.p. p_audio_drop
            drop_audio_text = {}
            visual_only_mask = rng.random(B) < visual_only_frac
            used = dict(feats)
            # We can't easily per-sample mask different stream sets in one forward
            # (collate stacks all). Instead zero out the dropped streams per-sample;
            # the fusion present-mask is by name, so to truly mask we must drop names
            # for the whole batch. Compromise: alternate batch-level regimes.
            regime = rng.random()
            f = dict(feats)
            if regime < visual_only_frac:
                # visual-only batch
                for s in AUDIO_TEXT:
                    f.pop(s, None)
                tag = "vis_only"
            else:
                # drop a random subset of audio/text for the whole batch
                for s in AUDIO_TEXT:
                    if rng.random() < p_audio_drop:
                        f.pop(s, None)
                tag = "mixed"
            # safety: ensure at least one stream
            if not f:
                f = {VISUAL[0]: feats[VISUAL[0]]}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(f, subj)
                loss, terms = composite_loss(pred, fmri, mask,
                                             vertex_weight=vertex_weight, **lw)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            if step % 25 == 0:
                ips = (step + 1) / max(time.time() - t0, 1e-3)
                print(f"  s{step:04d} loss={loss.item():.4f} ({tag}) "
                      f"negcorr={terms['negcorr']:.4f} ips={ips:.2f}")
            step += 1
            model.train()

    post = eval_visual(model)
    print("POST-finetune eval:", json.dumps(post, default=str))

    # save NEW checkpoint
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_ckpt = {"epoch": ckpt.get("epoch", 0), "state_dict": model.state_dict(),
                "config": cfg, "finetune": {"steps": steps, "p_audio_drop": p_audio_drop,
                "visual_only_frac": visual_only_frac, "lr": lr},
                "post_eval": post}
    torch.save(out_ckpt, f"{OUT_DIR}/best.pt")
    torch.save(out_ckpt, f"{OUT_DIR}/latest.pt")
    volume.commit()
    print(f"saved {OUT_DIR}/best.pt")

    elapsed_min = (time.time() - t0) / 60
    return {"out_ckpt": f"{OUT_DIR}/best.pt", "steps": steps,
            "train_minutes": elapsed_min, "post_eval": post}


@app.local_entrypoint()
def main(steps: int = 400, p_audio_drop: float = 0.5, lr: float = 1e-4,
         visual_only_frac: float = 0.35, batch_size: int = 8):
    res = finetune.remote(steps=steps, p_audio_drop=p_audio_drop, lr=lr,
                          visual_only_frac=visual_only_frac, batch_size=batch_size)
    print(json.dumps(res, indent=2, default=str))
