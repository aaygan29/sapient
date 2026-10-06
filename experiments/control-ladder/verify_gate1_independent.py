"""Independent recomputation of the GoEmotions Gate-1 result, from the cached neural
features, plus a positive control that proves the apparatus can detect a real effect.

WHY THIS EXISTS
---------------
goemotions/RESULTS.md reports Gate 1 NOT met (M0=0.725, M0+=0.703, M2=0.731,
diff=+0.010, CI [-0.016,+0.039]). That number is about to be written into
sapient_gate_evals as the first stored gate row, so it should be RECOMPUTED here
rather than transcribed. A negative result deserves the same audit as a positive one:
a broken pipeline also returns "no effect".

This file deliberately does NOT import goemotions_ingest. It re-implements the
protocol from the docstring/spec so a bug in the original does not reproduce itself.

THE CONTROL THAT MATTERS
------------------------
A null is only meaningful if the apparatus could have found a signal. So we also run
M2_control = M0+ concatenated with a NOISY COPY OF THE TARGET instead of the neural
features. If the gate does not fire there, the gate is broken and the real null is
uninterpretable. This is the check that separates "neural adds nothing" from
"our test can't detect anything".

Run:  python3 verify_gate1_independent.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).parent
CACHE = HERE / "goemotions/neural_cache.npz"
DATA = HERE / "goemotions/data"

# GoEmotions sentiment grouping (Demszky et al. 2020), idx order = emotions.txt 0..27.
POSITIVE = {0, 1, 4, 5, 7, 8, 13, 15, 17, 18, 20, 21, 23}
NEGATIVE = {2, 3, 9, 10, 11, 12, 16, 19, 24, 25}
SEED = 0
N_TARGET = 500


def load_rows():
    rows = []
    for split in ("train", "dev", "test"):
        p = DATA / f"{split}.tsv"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            text, labels, cid = parts[0], parts[1], parts[2]
            try:
                ids = {int(x) for x in labels.split(",")}
            except ValueError:
                continue
            rows.append((text, ids, cid))
    return rows


def binary_target(ids: set):
    pos, neg = bool(ids & POSITIVE), bool(ids & NEGATIVE)
    if pos == neg:          # both or neither -> ambiguous, drop
        return None
    return 1 if pos else 0


def auc_cv(X, y, seed=SEED, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tr, te in skf.split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X[tr] - mu) / sd, y[tr])
        yp.append(clf.predict_proba((X[te] - mu) / sd)[:, 1])
        yt.append(y[te])
    return float(roc_auc_score(np.concatenate(yt), np.concatenate(yp))), \
        np.concatenate(yt), np.concatenate(yp)


def boot_diff(y_a, p_a, y_b, p_b, n_boot=2000, seed=SEED):
    """Bootstrap CI on AUC(a) - AUC(b) over comments (paired by construction)."""
    rng = np.random.default_rng(seed)
    n = y_a.size
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y_a[i])) < 2:
            continue
        ds.append(roc_auc_score(y_a[i], p_a[i]) - roc_auc_score(y_b[i], p_b[i]))
    ds = np.asarray(ds)
    return float(ds.mean()), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


def main() -> int:
    cache = np.load(CACHE)
    have = set(cache.files)
    print(f"neural cache: {len(have)} comments x {cache[cache.files[0]].shape[0]}d")

    rows = [(t, i, c) for (t, i, c) in load_rows() if c in have]
    scored = [(t, binary_target(i), c) for (t, i, c) in rows]
    scored = [(t, y, c) for (t, y, c) in scored if y is not None]
    print(f"cached comments with an unambiguous pos/neg target: {len(scored)}")

    # balanced subset, deterministic
    rng = np.random.default_rng(SEED)
    pos = [r for r in scored if r[1] == 1]
    neg = [r for r in scored if r[1] == 0]
    k = min(len(pos), len(neg), N_TARGET // 2)
    rng.shuffle(pos); rng.shuffle(neg)
    sel = pos[:k] + neg[:k]
    rng.shuffle(sel)
    texts = [r[0] for r in sel]
    y = np.array([r[1] for r in sel])
    N = y.size
    print(f"balanced subset: N={N} ({k} pos / {k} neg)")

    # ---- M0: content ---------------------------------------------------------
    tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    X = tf.fit_transform(texts)
    M0 = TruncatedSVD(n_components=100, random_state=SEED).fit_transform(X)

    # ---- M0+: capacity-matched random lift to 400d ---------------------------
    r2 = np.random.default_rng(SEED)
    W = r2.normal(0, 1.0 / np.sqrt(M0.shape[1]), (M0.shape[1], 400))
    M0p = M0 @ W

    # ---- M2: M0+ concat real TRIBE neural (400d) -----------------------------
    NEU = np.stack([cache[r[2]] for r in sel]).astype(np.float64)
    M2 = np.hstack([M0p, NEU])

    # ---- POSITIVE CONTROL: M0+ concat a noisy copy of the target -------------
    r3 = np.random.default_rng(SEED + 1)
    leak = (y[:, None] + r3.normal(0, 1.0, (N, 400))).astype(np.float64)
    M2_ctrl = np.hstack([M0p, leak])

    auc0, _, _ = auc_cv(M0, y)
    auc0p, y0p, p0p = auc_cv(M0p, y)
    auc2, y2, p2 = auc_cv(M2, y)
    aucC, yC, pC = auc_cv(M2_ctrl, y)

    d_real, ci_real = boot_diff(y2, p2, y0p, p0p)
    d_ctrl, ci_ctrl = boot_diff(yC, pC, y0p, p0p)

    print(f"\nAUC  M0  (content, SVD-100)          = {auc0:.3f}")
    print(f"AUC  M0+ (capacity-matched, 400d)    = {auc0p:.3f}")
    print(f"AUC  M2  (M0+ + TRIBE neural, 800d)  = {auc2:.3f}")
    print(f"GATE 1  diff = {d_real:+.3f}  CI [{ci_real[0]:+.3f}, {ci_real[1]:+.3f}]  "
          f"-> {'MET' if ci_real[0] > 0 else 'NOT met'}")
    print(f"\nPOSITIVE CONTROL (M0+ + noisy target copy)")
    print(f"AUC  = {aucC:.3f}   diff = {d_ctrl:+.3f}  CI [{ci_ctrl[0]:+.3f}, {ci_ctrl[1]:+.3f}]  "
          f"-> {'FIRES (apparatus works)' if ci_ctrl[0] > 0 else 'DID NOT FIRE — GATE IS BROKEN'}")

    out = {
        "_what": "Independent recomputation of GoEmotions Gate 1 from the cached TRIBE "
                 "features, with a positive control. Does NOT import goemotions_ingest.",
        "reported_in_RESULTS_md": {"M0": 0.725, "M0+": 0.703, "M2": 0.731,
                                   "diff": 0.010, "ci": [-0.016, 0.039], "gate1": "NOT met"},
        "recomputed": {
            "N": int(N), "seed": SEED,
            "auc_M0_content": auc0, "auc_M0plus_capacity_matched": auc0p,
            "auc_M2_plus_neural": auc2,
            "gate1_diff": d_real, "gate1_ci95": ci_real,
            "gate1_met": bool(ci_real[0] > 0),
        },
        "positive_control": {
            "_what": "M0+ concatenated with a noisy copy of y instead of neural features. "
                     "If this does not fire, the null is uninterpretable.",
            "auc": aucC, "diff": d_ctrl, "ci95": ci_ctrl,
            "fired": bool(ci_ctrl[0] > 0),
        },
        "scope": [
            "text modality only; lexical content is a very strong sentiment baseline",
            "categorical pos/neg, NOT dimensional valence/arousal",
            "UNSEEN/population regime — GoEmotions raters are not TRIBE's 25 enrolled subjects, "
            "so only Gate 1 is testable here; Gate 2 individuation is not",
            "TTS-synthesized short Reddit comments are out-of-distribution for TRIBE "
            "(trained on movies/podcasts/narratives)",
        ],
    }
    (HERE / "goemotions/gate1_independent_verification.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote goemotions/gate1_independent_verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
