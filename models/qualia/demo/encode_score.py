"""Qualia demo — encode NeuroEngage operator audio through Mary and score the
human-vs-robot per-network response. Runs on Modal (weights + features on /data).

Pipeline per run (sub-XX_operator_run-YY):
  cached {beats, whisper} features  ->  MaryEngine.encode  ->  predicted fMRI (T, 20484)
  ->  per-vertex temporal std (how strongly Mary predicts that vertex is driven by
      this stimulus)  ->  Schaefer-1000 -> Yeo-7 network means.

Mary is loaded via the registry `improving` channel, so pointing that channel at a
better Mary (e.g. the finished Full-tier) upgrades this demo with no code change.

HONEST LABELING: this is Mary's *predicted response magnitude* per network for hearing
a human vs a robot partner — representational, not a claim of feeling, and not yet
controlled for low-level acoustic differences (that's the validated experiment).

Run:  modal run qualia/demo/encode_score.py
Out:  /data/qualia_demo/neuroengage_runs.json  (per-run Yeo-7) + brain PNGs, then
      pulled locally by build_results.py.
"""
from __future__ import annotations

import modal

MARY = "/Users/robertgutierrez/Desktop/sapient-models/mary/mary"
QUALIA = "/Users/robertgutierrez/Desktop/sapient-models/qualia"
SHARED = "/Users/robertgutierrez/Desktop/sapient-research/Mary-Papers/shared"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1", "numpy>=1.26,<3", "scipy>=1.13",
        "nilearn", "nibabel", "einops", "pyyaml", "matplotlib",
    )
    .env({
        "NILEARN_DATA": "/cache/nilearn",
        "QUALIA_MARY_ROOT": "/root/maryrepo",
        # pin the engine to the strongest complete core; flip to Full-tier later.
        "QUALIA_IMPROVING_CKPT": "/data/checkpoints/mary_multi_v2_s13/best.pt",
    })
    .add_local_dir(MARY, "/root/maryrepo/mary")          # the Mary model package
    .add_local_dir(QUALIA, "/root/qualia")               # MaryEngine + registry
    .add_local_file(f"{SHARED}/eval_harness.py", "/root/shared/eval_harness.py")  # schaefer_yeo imports it
    .add_local_file(f"{SHARED}/schaefer_yeo.py", "/root/shared/schaefer_yeo.py")
    .add_local_file(f"{SHARED}/plotting.py", "/root/shared/plotting.py")
)

app = modal.App("qualia-encode-neuroengage")
data_vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("mary-hf-cache", create_if_missing=True)

NEURO_DIR = "/data/features/mary/neuroengage/_stories"
OUT_DIR = "/data/qualia_demo"
SUBJECT_IDX = 0  # fixed head across all runs -> isolates the stimulus (human vs robot)


@app.function(image=image, gpu="A10G", timeout=3600,
              volumes={"/data": data_vol, "/cache": hf_vol})
def run() -> dict:
    import os
    import sys
    import json
    import numpy as np

    sys.path.insert(0, "/root/qualia")
    sys.path.insert(0, "/root/shared")
    sys.path.insert(0, "/root/maryrepo")
    from qualia.core import MaryEngine
    import schaefer_yeo as sy
    import plotting

    os.makedirs(OUT_DIR, exist_ok=True)
    eng = MaryEngine.from_channel("improving", device="cuda")
    print("engine:", eng.provenance())

    # Vertex -> Schaefer-1000 -> Yeo-7 (built once; downloads atlas to /cache).
    M = sy.vertex_to_parcel_matrix()            # (1000, 20484) sparse
    parcel_net = sy.parcel_to_yeo7()            # list[1000] network names

    runs = sorted(d for d in os.listdir(NEURO_DIR)
                  if os.path.isdir(os.path.join(NEURO_DIR, d)))
    print(f"{len(runs)} neuroengage runs: {runs}")

    results: dict[str, dict] = {}
    for key in runs:
        fdir = os.path.join(NEURO_DIR, key)
        # ALL available streams incl. VIDEO (qwen_vl/slowfast) — Mary IS trained on
        # video (cneuromod/HAD/wen2017); assemble_features loads only the present ones.
        feats = eng.assemble_features(
            fdir, streams=("beats", "whisper", "qwen_vl", "slowfast", "got_ocr"), device="cuda")
        present = sorted(feats.keys())
        verts = eng.encode(feats, subject_idx=SUBJECT_IDX)        # (T, 20484)
        emb = eng.embed(feats)                                    # (1024,) subject-independent representation
        act = verts.std(axis=0).astype(np.float32)                # per-vertex engagement
        parcels = np.asarray(M @ act).ravel()                     # (1000,)
        nets: dict[str, list] = {}
        for p, name in enumerate(parcel_net):
            if not str(name).startswith("_"):
                nets.setdefault(name, []).append(float(parcels[p]))
        results[key] = {n: float(np.mean(v)) for n, v in nets.items()}
        results[key]["_embedding"] = [float(x) for x in emb]      # for human-vs-robot separation test
        results[key]["_streams"] = present
        # brain PNG (per-vertex engagement) for the demo's left panel
        try:
            png = os.path.join(OUT_DIR, f"{key}_brain.png")
            plotting.brain_surface_map(act, out_png=png, title=key)
        except Exception as e:
            print(f"  (brain png skipped for {key}: {e})")
        print(f"  {key} [{'+'.join(present)}]: " + ", ".join(f"{n}={results[key][n]:.3f}" for n in
              ("Default", "Limbic", "Frontoparietal", "Visual") if n in results[key]))

    with open(os.path.join(OUT_DIR, "neuroengage_runs.json"), "w") as f:
        json.dump({"engine": eng.provenance(), "subject_idx": SUBJECT_IDX,
                   "metric": "per-vertex temporal std of Mary-predicted response, Yeo-7 mean",
                   "runs": results}, f, indent=2)
    data_vol.commit()
    return {"n_runs": len(runs), "out": f"{OUT_DIR}/neuroengage_runs.json"}


@app.local_entrypoint()
def main() -> None:
    print(run.remote())
