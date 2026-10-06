"""CPU differentiation gate for Sapient-1 AND Sapient-2 (no GPU, no downloads).

Loads each trained checkpoint from the sapient-data volume, infers the exact
SapientConfig from the state_dict tensor shapes, does a clean load_state_dict
(architecture proof), then runs the model on CACHED features from TWO different
stimuli and reports whether the brain-map OUTPUTS differ or collapse.

This is the honest collapse test the prior `qualia-serve` gate never produced
(its captured verdict was all RemoteError/ConflictError — no numbers).

S1 uses cached CNeuroMod Friends features (features/{vjepa2,w2vbert}/cneuromod).
S2 uses cached ds004996 features (features/{vjepa2,w2vbert}/ds004996).

Run:  modal run sapient1/scripts/diff_gate_cpu.py
"""
from __future__ import annotations

import modal

app = modal.App("sapient-diff-gate-cpu")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1", "numpy>=1.26,<3", "scipy>=1.13"
)
volume = modal.Volume.from_name("sapient-data", create_if_missing=False)

WINDOW = 200  # stim steps (100 s x 2 Hz)


@app.function(image=image, cpu=4.0, memory=16384, timeout=3600,
              volumes={"/data": volume})
def run() -> dict:
    import glob
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from torch import nn

    # ---- minimal SapientModel (identical arch for S1 and S2) ----
    class SapientModel(nn.Module):
        def __init__(self, d_video, d_audio, d_text, hidden, n_layers, n_heads,
                     ff_mult, low_rank_dim, n_vertices, n_subjects, max_fmri_len):
            super().__init__()
            self.hidden = hidden
            self.proj_video = nn.Linear(d_video, hidden)
            self.proj_audio = nn.Linear(d_audio, hidden)
            self.proj_text = nn.Linear(d_text, hidden)
            self.pos_embed = nn.Parameter(torch.zeros(1, max_fmri_len, hidden))
            self.subject_embed = nn.Embedding(n_subjects, hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * ff_mult,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.low_rank = nn.Linear(hidden, low_rank_dim, bias=False)
            self.brain_head = nn.Linear(low_rank_dim, n_vertices)

        def forward(self, video, audio, text, subject_idx):
            B = video.size(0)
            x = self.proj_video(video) + self.proj_audio(audio) + self.proj_text(text)
            x = x.view(B, -1, 2, self.hidden).mean(dim=2)
            x = x + self.pos_embed
            x = x + self.subject_embed(subject_idx).unsqueeze(1)
            x = self.transformer(x)
            x = self.low_rank(x)
            x = self.brain_head(x)
            return x

    def infer_cfg(sd):
        """Recover exact config from state_dict tensor shapes."""
        hidden = sd["proj_video.weight"].shape[0]
        cfg = dict(
            d_video=sd["proj_video.weight"].shape[1],
            d_audio=sd["proj_audio.weight"].shape[1],
            d_text=sd["proj_text.weight"].shape[1],
            hidden=hidden,
            low_rank_dim=sd["low_rank.weight"].shape[0],
            n_vertices=sd["brain_head.weight"].shape[0],
            n_subjects=sd["subject_embed.weight"].shape[0],
            max_fmri_len=sd["pos_embed"].shape[1],
            n_heads=8,
        )
        ff = sd["transformer.layers.0.linear1.weight"].shape[0]
        cfg["ff_mult"] = ff // hidden
        n_layers = len({k.split(".")[2] for k in sd
                        if k.startswith("transformer.layers.")})
        cfg["n_layers"] = n_layers
        return cfg

    def load_ckpt(path):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        elif isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
        else:
            # find the nested tensor dict
            sd = None
            for v in (obj.values() if isinstance(obj, dict) else []):
                if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                    sd = v
                    break
            if sd is None:
                raise RuntimeError(f"can't find state_dict in {path}; keys={list(obj.keys())[:10]}")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        meta = {}
        if isinstance(obj, dict):
            if "epoch" in obj:
                meta["epoch"] = obj["epoch"]
            if "metrics" in obj and isinstance(obj["metrics"], dict):
                meta["metrics"] = {k: (round(float(v), 5) if hasattr(v, "__float__") else v)
                                   for k, v in obj["metrics"].items()}
        return sd, meta

    def pick_two_stimuli(vdir, adir):
        """Return two (video, audio) feature pairs from DIFFERENT stimuli.

        Match strategy:
          1. exact stem (cneuromod: s01e01a.npy <-> s01e01a.npy)
          2. fallback: pair by (subject, run) token when stems differ
             (ds004996: sub-01_run-03 <-> sub-01_participant_denoised_run-03)
        """
        import re
        vids = sorted(glob.glob(str(Path(vdir) / "**" / "*.npy"), recursive=True))
        auds = sorted(glob.glob(str(Path(adir) / "**" / "*.npy"), recursive=True))
        astems = {Path(a).stem: a for a in auds}
        pairs = []
        for v in vids:
            s = Path(v).stem
            if s in astems:
                pairs.append((s, v, astems[s]))
        if pairs:
            return pairs

        def key(p):
            st = Path(p).stem
            sub = re.search(r"sub-\d+", st)
            run = re.search(r"run-\d+", st)
            return (sub.group(0) if sub else "", run.group(0) if run else "")

        # prefer participant_denoised audio for each (sub,run)
        abykey = {}
        for a in auds:
            k = key(a)
            cur = abykey.get(k)
            score = ("participant_denoised" in a, "participant" in a)
            if cur is None or score > cur[1]:
                abykey[k] = (a, score)
        for v in vids:
            k = key(v)
            if k in abykey:
                pairs.append(("_".join(k), v, abykey[k][0]))
        return pairs

    def window(arr, T=WINDOW):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] >= T:
            return arr[:T]
        pad = np.zeros((T - arr.shape[0],) + arr.shape[1:], dtype=np.float32)
        return np.concatenate([arr, pad], 0)

    def run_model(model, v_path, a_path, d_text):
        v = window(np.load(v_path, mmap_mode="r"))
        a = window(np.load(a_path, mmap_mode="r"))
        t = np.zeros((WINDOW, d_text), dtype=np.float32)  # AV-only: zero text
        with torch.no_grad():
            out = model(torch.from_numpy(v)[None], torch.from_numpy(a)[None],
                        torch.from_numpy(t)[None], torch.tensor([0]))
        return out[0].numpy()  # (100, n_vertices)

    def differentiation(model, pairs, d_text, label):
        """Run on 2 distinct stimuli, compare outputs."""
        if len(pairs) < 2:
            return {"error": f"<2 cached stimuli for {label} (found {len(pairs)})"}
        (s0, v0, a0), (s1, v1, a1) = pairs[0], pairs[len(pairs) // 2]
        o0 = run_model(model, v0, a0, d_text)
        o1 = run_model(model, v1, a1, d_text)
        # flatten and correlate the two brain maps across vertices*time
        f0, f1 = o0.reshape(-1), o1.reshape(-1)
        corr = float(np.corrcoef(f0, f1)[0, 1])
        # per-vertex mean map difference
        m0, m1 = o0.mean(0), o1.mean(0)
        map_corr = float(np.corrcoef(m0, m1)[0, 1])
        l2 = float(np.linalg.norm(m0 - m1) / (np.linalg.norm(m0) + 1e-9))
        # within-output rank: is each output near-constant across time? (collapse)
        std0 = float(o0.std(0).mean())
        std1 = float(o1.std(0).mean())
        return {
            "stim_a": s0, "stim_b": s1,
            "full_output_corr": round(corr, 4),
            "mean_map_corr": round(map_corr, 4),
            "relative_L2_diff": round(l2, 4),
            "out_absmean_a": round(float(np.abs(o0).mean()), 5),
            "out_absmean_b": round(float(np.abs(o1).mean()), 5),
            "temporal_std_a": round(std0, 5),
            "temporal_std_b": round(std1, 5),
            # COLLAPSE verdict: a healthy stimulus-driven encoder produces
            # clearly DIFFERENT mean brain maps for two different stimuli
            # (map_corr well below ~0.9). map_corr >= 0.97 means the output is
            # dominated by a shared, stimulus-invariant component (the subject/
            # pos embedding + sum-fusion mean) — i.e. near-rank-1 collapse.
            # Empirically both checkpoints sit at map_corr ~0.99 -> COLLAPSED.
            "map_corr_threshold_healthy": 0.90,
            "COLLAPSED": bool(map_corr >= 0.97),
            "DIFFERENTIATES": bool(map_corr < 0.90),
        }

    report = {}
    specs = [
        ("sapient1", "/data/checkpoints/sapient1_av/best_e00.pt",
         "/data/features/vjepa2/cneuromod", "/data/features/w2vbert/cneuromod"),
        ("sapient2", "/data/checkpoints/sapient2_scratch_av/best_e00.pt",
         "/data/features/vjepa2/ds004996", "/data/features/w2vbert/ds004996"),
    ]
    for name, ckpt, vdir, adir in specs:
        entry = {"checkpoint": ckpt}
        if not Path(ckpt).exists():
            entry["error"] = "checkpoint missing"
            report[name] = entry
            continue
        sd, meta = load_ckpt(ckpt)
        cfg = infer_cfg(sd)
        entry["inferred_config"] = cfg
        entry["ckpt_meta"] = meta
        model = SapientModel(**cfg).eval()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        entry["load_state_dict"] = {
            "clean": len(missing) == 0 and len(unexpected) == 0,
            "missing_keys": list(missing)[:10],
            "unexpected_keys": list(unexpected)[:10],
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        pairs = pick_two_stimuli(vdir, adir)
        entry["n_cached_stimuli"] = len(pairs)
        entry["differentiation"] = differentiation(model, pairs, cfg["d_text"], name)
        report[name] = entry

    return report


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(run.remote(), indent=2))
