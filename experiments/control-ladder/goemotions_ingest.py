"""
goemotions_ingest.py - Commercial-clean Gate-1 text bridge (GoEmotions, CC-BY 4.0).

Question: does the TRIBE v2 neural prior (predicted BOLD for hearing a comment) add incremental validity
over content-only (M0) and capacity-matched content (M0+) for predicting the human-assigned affect of
the text? Gate 1 only (population value; GoEmotions raters are not TRIBE's 25 enrolled subjects).

Bridge per comment:
  text --> TRIBE v2 text branch (TTS -> whisperx -> features) --> predicted BOLD (T, 20484)
        --> Schaefer-400 ROI means --> mean over time --> 400-d neural vector  [M2 add-on]
  text --> TF-IDF + SVD content embedding                                       [M0]
  M0   --> random tanh lift to 400-d                                            [M0+]
  y    --> binary affect: positive vs negative (from GoEmotions sentiment map)  [target]

Gate 1 (pre-registered): AUC(M2) > AUC(M0+), bootstrap CI over held-out comments excludes 0.
M2 = [M0+ , neural-400]. License: GoEmotions CC-BY 4.0; Schaefer atlas MIT. Commercial-clean, no EULA.

Run:
  python3 goemotions_ingest.py --selftest            # analysis core on synthetic stand-in (no TRIBE)
  python3 goemotions_ingest.py --probe               # 1 real TRIBE text call, verify the path + shape
  python3 goemotions_ingest.py --pilot 40            # real Gate-1 pilot on N balanced comments (GPU $)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ATLAS_DIR = HERE / "atlas"
DATA = HERE / "goemotions" / "data"
CACHE = HERE / "goemotions" / "neural_cache.npz"

# GoEmotions sentiment grouping (Demszky et al. 2020). idx order = emotions.txt (0..27).
POSITIVE = {0,1,4,5,7,8,13,15,17,18,20,21,23}   # admiration..relief etc.
NEGATIVE = {2,3,9,10,11,12,16,19,24,25}          # anger, annoyance, disappointment, ... sadness, remorse
# (ambiguous: confusion, curiosity, realization, surprise; neutral=27 -> dropped from the binary target)


# ─────────────────────────── atlas reduction (shared with ingest_liris) ───────────────────────────
_LABELS = None
def schaefer_labels():
    global _LABELS
    if _LABELS is None:
        import nibabel.freesurfer.io as fs
        lh,_,_ = fs.read_annot(ATLAS_DIR / "lh.Schaefer2018_400Parcels_7Networks_order.annot")
        rh,_,_ = fs.read_annot(ATLAS_DIR / "rh.Schaefer2018_400Parcels_7Networks_order.annot")
        rh = rh.copy(); rh[rh>0] += int(lh.max())
        _LABELS = np.concatenate([lh, rh]).astype(np.int64)
    return _LABELS

def bold_to_vec(bold: np.ndarray) -> np.ndarray:
    """(T, 20484) predicted BOLD -> 400-d per-comment vector (ROI means, then mean over time)."""
    lab = schaefer_labels()
    parcels = [p for p in np.unique(lab) if p != 0]
    roi = np.stack([bold[:, lab==p].mean(1) for p in parcels], axis=1)  # (T, 400)
    return roi.mean(0)  # (400,)


# ─────────────────────────── data ───────────────────────────
def load_split(name: str):
    rows = []
    for line in (DATA / f"{name}.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3: continue
        text, labels, cid = parts[0], parts[1], parts[2]
        ids = {int(x) for x in labels.split(",")}
        rows.append((cid, text, ids))
    return rows

def binary_target(ids: set) -> int | None:
    pos = len(ids & POSITIVE) > 0; neg = len(ids & NEGATIVE) > 0
    if pos and not neg: return 1
    if neg and not pos: return 0
    return None  # mixed / ambiguous / neutral -> excluded

def balanced_subset(rows, n: int, seed=0):
    rng = np.random.default_rng(seed)
    pos = [(c,t) for c,t,i in rows if binary_target(i)==1]
    neg = [(c,t) for c,t,i in rows if binary_target(i)==0]
    k = n//2
    rng.shuffle(pos); rng.shuffle(neg)
    sel = [(c,t,1) for c,t in pos[:k]] + [(c,t,0) for c,t in neg[:k]]
    rng.shuffle(sel)
    return sel  # list of (cid, text, y)


# ─────────────────────────── neural features (TRIBE text branch) ───────────────────────────
def tribe_text_vec(text: str) -> np.ndarray:
    import modal
    Tribe = modal.Cls.from_name("tribev2-inference", "Tribe")
    out = Tribe().predict_bold.remote(text=text)
    return bold_to_vec(np.asarray(out["bold"], dtype=np.float32))

def extract_neural(pool, target, use_cache=True, buffer=1.2):
    """Extract 400-d neural vectors for `target` comments, fanning out across Modal containers via
    .map() (parallel). Skips TRIBE text-pipeline failures (return_exceptions=True). Streams results to
    the on-disk cache so GPU spend is never lost mid-batch. Returns (feats dict, kept list)."""
    import modal

    cache = {}
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True); cache = {k: z[k] for k in z.files}

    feats, kept, to_run = {}, [], []
    for cid, text, y in pool:
        if cid in cache and len(kept) < target:
            feats[cid] = cache[cid]; kept.append((cid, text, y))
        elif cid not in cache:
            to_run.append((cid, text, y))
    if len(kept) >= target:
        return feats, kept[:target]

    need = target - len(kept)
    batch = to_run[: int(need * buffer) + 5]            # submit a buffered batch (≈0% skip observed)
    print(f"  fanning out {len(batch)} TRIBE calls via .map (need {need} more, {len(kept)} cached)...")
    Tribe = modal.Cls.from_name("tribev2-inference", "Tribe")
    obj = Tribe()
    texts = [t for _, t, _ in batch]
    skipped = 0
    for i, ((cid, text, y), res) in enumerate(zip(batch, obj.neural_vec.map(texts, return_exceptions=True))):
        if isinstance(res, Exception):
            skipped += 1
            print(f"      skip {cid} ({type(res).__name__}: {str(res)[:50]})")
            continue
        v = np.asarray(res, dtype=np.float32)
        feats[cid] = v; cache[cid] = v; kept.append((cid, text, y))
        if (i + 1) % 25 == 0:
            np.savez(CACHE, **cache); print(f"      ...{len(kept)} kept (cache saved)")
    np.savez(CACHE, **cache)
    print(f"  kept {len(kept)}, skipped {skipped}")
    return feats, kept[:target]


# ─────────────────────────── content features (M0) ───────────────────────────
def content_embed(texts, dim=100):
    tf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1)
    X = tf.fit_transform(texts)
    d = min(dim, X.shape[1]-1, len(texts)-1)
    return TruncatedSVD(n_components=max(2,d), random_state=0).fit_transform(X)  # (N, d)

def cap_match(m0, out_dim=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.tanh(m0 @ rng.standard_normal((m0.shape[1], out_dim)))


# ─────────────────────────── Gate 1 ───────────────────────────
def auc_cv(X, y, n_splits=5, seed=0):
    """Stratified k-fold pooled AUC."""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y); yt, yp = [], []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        yt.append(y[te]); yp.append(m.predict_proba(sc.transform(X[te]))[:,1])
    return roc_auc_score(np.concatenate(yt), np.concatenate(yp))

def gate1(M0, M0plus, M2, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y)
    sk = {"M0": auc_cv(M0,y), "M0+": auc_cv(M0plus,y), "M2": auc_cv(M2,y)}
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2: continue
        diffs.append(auc_cv(M2[idx], y[idx]) - auc_cv(M0plus[idx], y[idx]))
    lo,hi = np.percentile(diffs,[2.5,97.5]); d=float(np.mean(diffs))
    print(f"\n  AUC  M0={sk['M0']:.3f}  M0+={sk['M0+']:.3f}  M2={sk['M2']:.3f}")
    print(f"  GATE 1  M2 > M0+ : diff={d:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  -> "
          f"{'PASS (neural prior adds)' if lo>0 else 'no (Gate 1 not met)'}")


# ─────────────────────────── entry points ───────────────────────────
def selftest():
    rng = np.random.default_rng(0); N=200
    m0 = rng.standard_normal((N,16))                                     # content
    z  = rng.standard_normal((N,8))                                      # latent ABSENT from content
    Wc = rng.standard_normal((16,200)); Wz = rng.standard_normal((8,200))
    neural = np.concatenate([np.tanh(m0@Wc), np.tanh(z@Wz)], 1)          # neural carries content + z
    neural += 0.2*rng.standard_normal((N,400))
    wz = rng.standard_normal(8)
    y = ((z @ wz + 0.3*(m0[:,0]) + 0.5*rng.standard_normal(N)) > 0).astype(int)  # affect driven by z (unseen in content)
    M0=m0; M0plus=cap_match(m0,400,0); M2=np.concatenate([M0plus,neural],1)
    print("="*68); print("SELFTEST (synthetic; analysis core only). Expect Gate 1 PASS"); print("       (neural carries a latent z that drives affect but is absent from content).") ; print("="*68)
    gate1(M0,M0plus,M2,y,n_boot=400)
    print("Selftest done.")

def probe():
    sel = balanced_subset(load_split("test"), 2)
    cid,text,y = sel[0]
    print(f"Probing TRIBE text path on: {text[:60]!r}")
    v = tribe_text_vec(text)
    print(f"neural vector shape: {v.shape}  finite: {np.isfinite(v).all()}")
    assert v.shape == (400,), v.shape
    print("PROBE PASS: TRIBE text branch -> 400-d neural vector.")

def pilot(n: int):
    pool = balanced_subset(load_split("test"), n * 3)   # over-request so skips don't starve the target
    print(f"Pilot target N={n} (balanced pos/neg), pool={len(pool)}. Extracting neural features (cached)...")
    neural, kept = extract_neural(pool, target=n)
    cids = [c for c,_,_ in kept]; texts=[t for _,t,_ in kept]; y=[yy for _,_,yy in kept]
    NEU = np.stack([neural[c] for c in cids])
    M0 = content_embed(texts); M0plus = cap_match(M0,400,0); M2 = np.concatenate([M0plus,NEU],1)
    print(f"features: M0 {M0.shape}  M0+ {M0plus.shape}  M2 {M2.shape}  y(+ rate)={np.mean(y):.2f}")
    gate1(M0,M0plus,M2,y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--pilot", type=int, default=0)
    a = ap.parse_args()
    if a.selftest: selftest()
    elif a.probe: probe()
    elif a.pilot: pilot(a.pilot)
    else: print(__doc__); sys.exit(1)
