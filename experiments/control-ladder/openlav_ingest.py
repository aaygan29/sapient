"""
openlav_ingest.py - Commercial-clean DIMENSIONAL-AFFECT VIDEO Gate-1 bridge (OpenLAV, CC-BY 3.0).

The regime the GoEmotions text negative did NOT test: real video + continuous valence/arousal, the
domain TRIBE was actually trained on. OpenLAV = 188 CC-BY videos with per-video mean valence/arousal
(422 raters). Commercial-clean, no EULA, no affiliation.

Rigorous Gate-1 design (one TRIBE pass per video via the video_features Modal method):
  M0   = TRIBE foundation features (encoder INPUT: aggregated V-JEPA2 + audio + text), H-d
  M0+  = capacity-matched random lift of M0 to 400-d
  M2   = [M0+ , predicted-BOLD->Schaefer-400]   (the brain step's output added on)
  y    = per-video valence and arousal (continuous; z-scored)
Gate 1: skill(M2) > skill(M0+) for predicting valence (and arousal), leave-one-video-out, bootstrap CI.
Because M0 is TRIBE's OWN foundation rep, this is the clean test "does the BRAIN step add over the
foundation model" (not confounded by a weaker external content encoder).

Run:
  python3 openlav_ingest.py --selftest
  python3 openlav_ingest.py --run            # uses openlav/videos/*.webm + openlav/video_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
OL = HERE / "openlav"
VIDEOS = OL / "videos"
TARGETS_CSV = OL / "video_data.csv"
CACHE = OL / "video_feat_cache.npz"
ALPHAS = np.logspace(-2, 4, 13)


# ─────────────────────────── targets ───────────────────────────
def load_targets():
    """video_code -> (valence, arousal) from video_data.csv (comma-sep, has valence/arousal cols)."""
    import csv
    out = {}
    with open(TARGETS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["video_code"].strip()] = (float(row["valence"]), float(row["arousal"]))
            except (KeyError, ValueError):
                continue
    return out


# ─────────────────────────── neural+content features (TRIBE video pass) ───────────────────────────
def extract_features(codes, video_dir: Path, use_cache=True):
    """Map TRIBE.video_features over the videos -> {code: (neural400, contentH)}. Skips failures."""
    import modal

    cache = {}
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = {k: z[k] for k in z.files}

    feats, kept, to_run = {}, [], []
    for c in codes:
        if f"{c}__neural" in cache:
            feats[c] = (cache[f"{c}__neural"], cache[f"{c}__content"]); kept.append(c)
        else:
            vp = video_dir / f"{c}.webm"
            if vp.exists() and vp.stat().st_size > 1000:
                to_run.append((c, vp))
    if to_run:
        print(f"  fanning out {len(to_run)} TRIBE video passes via .starmap ({len(kept)} cached)...", flush=True)
        Tribe = modal.Cls.from_name("tribev2-inference", "Tribe")
        obj = Tribe()
        args = [(c, vp.read_bytes()) for c, vp in to_run]
        skipped = 0
        # stream as-completed (no head-of-line block); each result self-identifies via "code"; save EVERY result
        for res in obj.video_features.starmap(args, return_exceptions=True, order_outputs=False):
            if isinstance(res, Exception):
                skipped += 1; print(f"      skip ({type(res).__name__}: {str(res)[:50]})", flush=True); continue
            c = res["code"]
            neu = np.asarray(res["neural"], dtype=np.float32)
            con = np.asarray(res["content"], dtype=np.float32)
            feats[c] = (neu, con); kept.append(c)
            cache[f"{c}__neural"] = neu; cache[f"{c}__content"] = con
            np.savez(CACHE, **cache)  # persist after every result -> resumable, never lose >1
            if len(kept) % 10 == 0:
                print(f"      ...{len(kept)} kept (cache saved)", flush=True)
        np.savez(CACHE, **cache)
        print(f"  kept {len(kept)}, skipped {skipped}")
    return feats, kept


def cap_match(m0, out_dim, seed=0):
    rng = np.random.default_rng(seed)
    return np.tanh(m0 @ rng.standard_normal((m0.shape[1], out_dim)))


# ─────────────────────────── Gate-1 (regression, LOVO) ───────────────────────────
def r2(yt, yp):
    ss = np.sum((yt - yp) ** 2); tot = np.sum((yt - yt.mean()) ** 2)
    return 1 - ss / tot if tot > 0 else 0.0


def lovo_pred(X, y):
    """Leave-one-video-out out-of-fold predictions (computed once; O(N) fits)."""
    yp = np.empty(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        yp[i] = m.predict(sc.transform(X[i:i+1]))[0]
    return yp


def gate1(M0, M0p, M2, Y, names=("valence", "arousal"), n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    for k, nm in enumerate(names):
        y = (Y[:, k] - Y[:, k].mean()) / (Y[:, k].std() + 1e-8)
        # compute OOF predictions ONCE per arm, then bootstrap R^2 over the held-out predictions
        p0, p0p, p2 = lovo_pred(M0, y), lovo_pred(M0p, y), lovo_pred(M2, y)
        sk = {"M0": r2(y, p0), "M0+": r2(y, p0p), "M2": r2(y, p2)}
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            diffs[b] = r2(y[idx], p2[idx]) - r2(y[idx], p0p[idx])
        lo, hi = np.percentile(diffs, [2.5, 97.5]); d = float(diffs.mean())
        print(f"\n[{nm}]  R2  M0={sk['M0']:+.3f}  M0+={sk['M0+']:+.3f}  M2={sk['M2']:+.3f}")
        print(f"   GATE 1  M2 > M0+ : diff={d:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  -> "
              f"{'PASS (brain step adds)' if lo > 0 else 'no (Gate 1 not met)'}")


# ─────────────────────────── entry points ───────────────────────────
def selftest():
    rng = np.random.default_rng(0); N = 180          # realistic N (OpenLAV ~188)
    content = rng.standard_normal((N, 64))            # foundation features (M0)
    brain_extra = rng.standard_normal((N, 64))        # latent the BRAIN step adds, absent from content
    W = rng.standard_normal((128, 400))
    neural = np.tanh(np.concatenate([content, brain_extra], 1) @ W) + 0.1 * rng.standard_normal((N, 400))
    Y = np.stack([brain_extra @ rng.standard_normal(64),
                  brain_extra @ rng.standard_normal(64)], 1) + 0.1 * rng.standard_normal((N, 2))
    M0 = content; M0p = cap_match(M0, 400, 0); M2 = np.concatenate([M0p, neural], 1)
    print("=" * 66); print("SELFTEST (synthetic). Expect Gate 1 PASS (brain carries signal"); print("       absent from foundation features)."); print("=" * 66)
    gate1(M0, M0p, M2, Y, n_boot=300)
    print("Selftest done.")


def run():
    targets = load_targets()
    codes = [c for c in targets if (VIDEOS / f"{c}.webm").exists()]
    print(f"{len(codes)} videos present with targets. Extracting TRIBE features...")
    feats, kept = extract_features(codes, VIDEOS)
    if len(kept) < 20:
        print(f"only {len(kept)} usable; need >=20"); return
    H = min(feats[k][1].shape[0] for k in kept)                  # common foundation dim
    M0 = np.stack([feats[c][1][:H] for c in kept])
    NEU = np.stack([feats[c][0] for c in kept])
    M0p = cap_match(M0, 400, 0)
    M2 = np.concatenate([M0p, NEU], 1)
    Y = np.array([targets[c] for c in kept], dtype=np.float64)
    print(f"features: M0 {M0.shape}  M0+ {M0p.shape}  M2 {M2.shape}  N={len(kept)}")
    gate1(M0, M0p, M2, Y)


def analyze():
    """Run Gate 1 on whatever is already in the feature cache (read-only; for an early/partial read)."""
    targets = load_targets()
    z = np.load(CACHE, allow_pickle=True)
    codes = sorted({k.split("__")[0] for k in z.files if k.endswith("__neural") and k.split("__")[0] in targets})
    if len(codes) < 20:
        print(f"only {len(codes)} cached; need >=20"); return
    H = min(z[f"{c}__content"].shape[0] for c in codes)
    M0 = np.stack([z[f"{c}__content"][:H] for c in codes])
    NEU = np.stack([z[f"{c}__neural"] for c in codes])
    M0p = cap_match(M0, 400, 0); M2 = np.concatenate([M0p, NEU], 1)
    Y = np.array([targets[c] for c in codes], dtype=np.float64)
    print(f"[analyze] N={len(codes)} cached  M0 {M0.shape}  M2 {M2.shape}")
    gate1(M0, M0p, M2, Y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest()
    elif a.run: run()
    elif a.analyze: analyze()
    else: print(__doc__); sys.exit(1)
