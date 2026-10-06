"""Build a per-vertex noise-ceiling map for Mary via inter-subject correlation (ISC).

Why ISC (not within-subject repeats): for naturalistic stimuli, the field-standard
ceiling is how reliably a vertex tracks the *stimulus* across people. Wen2017's repeats
are stored already-averaged, but 169/193 stories here are shared by >=2 subjects (Huth:
22 subjects/story), so leave-one-out ISC gives a clean reliability map.

For each shared (dataset, story) group:
  X = stack of subjects' (T, V) BOLD (already 1Hz, z-scored per vertex per run).
  For subject s: r_s[v] = corr_t( X[s,:,v], mean_{j!=s} X[j,:,v] )   (leave-one-out)
  group ISC[v] = mean_s r_s[v]
Aggregate across groups: weighted mean per vertex, weight = (n_subj-1)*T (estimator
precision). Negative ISC clamped to 0.

Outputs on the volume:
  /data/noise_ceiling_isc.npy        (20484,) float32 — per-vertex ISC ceiling
  /data/noise_ceiling_isc_meta.json  quantiles + responsive-vertex counts at thresholds

Run: modal run scripts/build_noise_ceiling.py
     modal run scripts/build_noise_ceiling.py --manifest manifest_mary_multi.json
"""
from __future__ import annotations

import modal

app = modal.App("mary-build-noise-ceiling")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy>=1.26,<3")
vol = modal.Volume.from_name("sapient-data", create_if_missing=True)


def _pearson_along_time(x, y):
    """Per-vertex Pearson r along axis 0. x, y: (T, V) -> (V,)."""
    import numpy as np
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = (x * y).sum(axis=0)
    den = np.sqrt((x ** 2).sum(axis=0) * (y ** 2).sum(axis=0))
    den = np.where(den < 1e-8, 1e-8, den)
    r = num / den
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


@app.function(image=image, cpu=4.0, memory=32768, timeout=40 * 60, volumes={"/data": vol})
def build(manifest: str = "manifest_mary_multi.json") -> dict:
    import json
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    m = json.loads((Path("/data") / manifest).read_text())
    entries = m["entries"]
    V = 20484

    def story_key(e):
        return (e.get("dataset"), e.get("story", e.get("run", e.get("run_key"))))

    groups = defaultdict(list)
    for e in entries:
        groups[story_key(e)].append(e)

    acc = np.zeros(V, dtype=np.float64)      # weighted sum of ISC
    wsum = np.zeros(V, dtype=np.float64)     # sum of weights
    n_groups_used = 0
    per_group = []

    for key, es in groups.items():
        # One run per subject for this story (dedupe subject_idx, first run wins).
        by_subj = {}
        for e in es:
            by_subj.setdefault(e["subject_idx"], e)
        if len(by_subj) < 2:
            continue
        arrs = []
        for e in by_subj.values():
            try:
                a = np.asarray(np.load(e["fmri_path"], mmap_mode="r"), dtype=np.float32)
            except Exception:
                continue
            arrs.append(a)
        if len(arrs) < 2:
            continue
        T = min(a.shape[0] for a in arrs)
        if T < 20:
            continue
        X = np.stack([a[:T] for a in arrs], axis=0)        # (S, T, V)
        S = X.shape[0]
        total = X.sum(axis=0)                               # (T, V)
        isc = np.zeros(V, dtype=np.float64)
        for s in range(S):
            loo = (total - X[s]) / (S - 1)                  # (T, V) mean of others
            isc += _pearson_along_time(X[s], loo)
        isc /= S
        w = float((S - 1) * T)
        acc += w * isc
        wsum += w
        n_groups_used += 1
        per_group.append({"group": f"{key[0]}:{key[1]}", "n_subj": S, "T": T,
                          "isc_mean": float(isc.mean()), "isc_p95": float(np.quantile(isc, 0.95))})

    ceiling = (acc / np.where(wsum < 1e-9, 1e-9, wsum)).astype(np.float32)
    ceiling = np.clip(ceiling, 0.0, None)

    np.save("/data/noise_ceiling_isc.npy", ceiling)

    q = {f"p{int(p*100)}": float(np.quantile(ceiling, p)) for p in
         (0.5, 0.75, 0.9, 0.95, 0.99)}
    thresholds = {f">{t}": int((ceiling > t).sum()) for t in (0.0, 0.02, 0.05, 0.1, 0.2)}
    meta = {
        "manifest": manifest,
        "n_vertices": V,
        "n_groups_used": n_groups_used,
        "ceiling_mean": float(ceiling.mean()),
        "ceiling_quantiles": q,
        "n_responsive_at_threshold": thresholds,
        "top_groups_by_isc": sorted(per_group, key=lambda d: -d["isc_p95"])[:10],
    }
    (Path("/data") / "noise_ceiling_isc_meta.json").write_text(json.dumps(meta, indent=2))
    try:
        vol.commit()
    except Exception:
        pass
    return meta


@app.local_entrypoint()
def main(manifest: str = "manifest_mary_multi.json"):
    import json
    print(json.dumps(build.remote(manifest), indent=2))
