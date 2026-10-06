"""Diagnostic: which streams does a Mary checkpoint actually train + use?

Loads checkpoints on CPU and reports, per stream: whether trained weights exist in
the checkpoint (stream_encoders.<s>.*), whether the stream is in model.stream_order
(the forward loop only processes streams in stream_order), and the weight norm.
Answers: does this checkpoint actually incorporate VIDEO (qwen_vl/slowfast/got_ocr)?

Run:  modal run qualia/demo/diagnose_streams.py
"""
from __future__ import annotations

import modal

MARY = "/Users/robertgutierrez/Desktop/sapient-models/mary/mary"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "einops", "pyyaml")
    .env({"QUALIA_MARY_ROOT": "/root/maryrepo"})
    .add_local_dir(MARY, "/root/maryrepo/mary")
)
app = modal.App("qualia-diagnose-streams")
vol = modal.Volume.from_name("sapient-data")

CKPTS = {
    "mary_multi_v2_s13": "/data/checkpoints/mary_multi_v2_s13/best.pt",
    "mary_full_s13": "/data/checkpoints/mary_full_s13/best.pt",
}
ALL = ["slowfast", "qwen_vl", "got_ocr", "beats", "whisper", "qwen_ctx"]


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def run() -> dict:
    import sys
    sys.path.insert(0, "/root/maryrepo")
    import torch
    from mary.model import MaryConfig, MaryModel

    out = {}
    for name, path in CKPTS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        sd = ck["state_dict"]
        model = MaryModel(MaryConfig.from_yaml(cfg))
        order = list(getattr(model, "stream_order", []))
        per_stream = {}
        for s in ALL:
            keys = [k for k in sd if f"stream_encoders.{s}." in k]
            norm = float(sum(float(sd[k].float().pow(2).sum()) for k in keys) ** 0.5) if keys else 0.0
            per_stream[s] = {"in_stream_order": s in order,
                             "trained_weights_in_ckpt": len(keys) > 0,
                             "weight_norm": round(norm, 2)}
        # surface any stream-ish config fields
        cfg_streams = {k: cfg.get(k) for k in ("streams", "stream_order", "active_streams")
                       if isinstance(cfg, dict) and k in cfg}
        out[name] = {"d_model": int(model.cfg.d_model), "stream_order": order,
                     "cfg_stream_fields": cfg_streams, "per_stream": per_stream}
        print(f"\n=== {name} (d_model={out[name]['d_model']}) ===")
        print(f"stream_order: {order}")
        for s in ALL:
            ps = per_stream[s]
            print(f"  {s:9s} in_order={ps['in_stream_order']!s:5s} "
                  f"trained_weights={ps['trained_weights_in_ckpt']!s:5s} norm={ps['weight_norm']}")
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=2))
