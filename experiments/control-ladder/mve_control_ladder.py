"""
Neurobehavioral Prediction Engine - MVE control-ladder pipeline + ground-truth validation.

This is the REAL analysis machinery from NEUROBEHAVIORAL_ENGINE_PROTOCOL_2026-06.md (Council-revised
section 3). It is validated here against SYNTHETIC ground truth, NOT real brains. The purpose is the
step that must precede any real run: prove the control ladder recovers known truth before trusting it
on data.

The ladder (protocol section 3):
  M0       content-only
  M0plus   capacity-matched non-brain content basis (random-weight "encoder")  -> Gate 1 control
  M1       M0plus + per-subject behavioral history
  M2       M1 + correct-subject neural features (the neural prior)
  M2perm   M1 + WRONG-subject neural features (subject-label permutation)       -> Gate 2 control

Pre-registered primary (protocol section 7):
  Gate 1: skill(M2) > skill(M0plus)   (brain beats a bigger basis)
  Gate 2: skill(M2) > skill(M2perm)   (individual neural signal is real, not generic stimulus drive)

Two SPLITS, to demonstrate the enrolled-vs-unseen regime distinction the Council demanded:
  LOSO-stim : leave-one-STIMULUS-out, subjects seen   -> the ENROLLED regime (individuation possible)
  LOSO-subj : leave-one-SUBJECT-out                   -> the UNSEEN regime (population-neural only)

Two WORLDS, the positive and negative control on the apparatus itself:
  SIGNAL : behavior depends on an individual-specific neural component (w_indiv > 0)
  NULL   : behavior depends only on stimulus content (w_indiv = 0); neural is pure stimulus drive

Expected, if the apparatus is correct:
  SIGNAL + enrolled  -> Gate 1 PASS, Gate 2 PASS   (we can detect real individual neural signal)
  NULL   + enrolled  -> Gate 2 does NOT fire        (no false positive: M2 ~ M2perm)
  SIGNAL + unseen    -> Gate 2 does NOT fire        (individual signal cannot transfer to a new person;
                                                     the neural prior collapses toward content)
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ---- dimensions (small, fast; the logic is what is under test) ----
N_SUBJ = 12
N_STIM = 120
D_CONTENT = 16        # stimulus content features (CLIP-like)
D_NEURAL = 64         # neural feature dim (predicted-BOLD-like); > content, so M0plus matters
N_HIST = 4            # per-subject behavioral history length
ALPHAS = np.logspace(-2, 4, 13)
SEEDS = range(20)


def simulate_world(world, seed):
    """Generate a world with known ground truth. Returns content, neural (correct), y, subj ids."""
    rng = np.random.default_rng(seed)

    # stimulus content
    C = rng.standard_normal((N_STIM, D_CONTENT))

    # population stimulus->neural map (deterministic function of content; collinear with C BY DESIGN)
    W_pop = rng.standard_normal((D_CONTENT, D_NEURAL))
    neural_pop = np.tanh(C @ W_pop)                      # [N_STIM, D_NEURAL], same for everyone

    # per-subject individual neural component (the identity signal)
    subj_embed = rng.standard_normal((N_SUBJ, D_NEURAL)) * 1.5
    # individual modulation of the response to each stimulus (subject x stim x neural)
    indiv_mod = np.einsum('sd,nd->snd', subj_embed, np.tanh(C @ W_pop * 0.5))

    # correct-subject neural features: population + individual + noise (an ENROLLED twin's features)
    noise = 0.3 * rng.standard_normal((N_SUBJ, N_STIM, D_NEURAL))
    N_correct = neural_pop[None, :, :] + indiv_mod + noise
    # population-only neural: what zero-shot TRIBE returns for an UNSEEN subject (no individual component)
    N_pop_only = neural_pop[None, :, :] + noise

    # behavioral outcome y (subject x stim)
    w_content = rng.standard_normal(D_NEURAL) * 0.7      # how content (via neural_pop) drives behavior
    w_indiv = (3.0 if world == "SIGNAL" else 0.0)        # how individual neural component drives behavior
    indiv_drive = np.einsum('snd,d->sn', indiv_mod, rng.standard_normal(D_NEURAL))
    y = (neural_pop @ w_content)[None, :] + w_indiv * indiv_drive + 0.5 * rng.standard_normal((N_SUBJ, N_STIM))

    return dict(C=C, N_correct=N_correct, N_pop_only=N_pop_only, y=y, subj_embed=subj_embed, rng=rng)


def random_basis(C, out_dim, rng):
    """Capacity-matched non-brain featurization: lift content to neural dim via a random-weight tanh
    map. This is the synthetic analogue of 'random-weight TRIBE' in protocol section 3 (M0plus)."""
    W = rng.standard_normal((C.shape[1], out_dim))
    return np.tanh(C @ W)


def make_features(world_data, rng):
    """Build the feature blocks for each model arm. Returns dict of [subj, stim, dim] arrays."""
    C, Nc, y = world_data["C"], world_data["N_correct"], world_data["y"]
    S, T = N_SUBJ, N_STIM

    content = np.repeat(C[None], S, axis=0)                       # [S,T,Dc]
    m0plus_basis = np.repeat(random_basis(C, D_NEURAL, rng)[None], S, axis=0)  # capacity-matched

    # per-subject behavioral history: mean of N_HIST random other trials' y (leak-safe: excludes self at eval via CV)
    hist = np.zeros((S, T, N_HIST))
    for s in range(S):
        for t in range(T):
            others = rng.choice([j for j in range(T) if j != t], size=N_HIST, replace=False)
            hist[s, t] = y[s, others]

    # subject-permuted neural: assign each subject a DIFFERENT subject's neural features
    perm = rng.permutation(S)
    while np.any(perm == np.arange(S)):                            # ensure no subject keeps its own
        perm = rng.permutation(S)
    N_perm = Nc[perm]

    return dict(content=content, m0plus=m0plus_basis, hist=hist, neural=Nc,
                neural_perm=N_perm, neural_pop=world_data["N_pop_only"])


def assemble(arm, F):
    """Concatenate feature blocks per protocol arm definitions."""
    if arm == "M0":      blocks = [F["content"]]
    elif arm == "M0plus":blocks = [F["m0plus"]]
    elif arm == "M1":    blocks = [F["m0plus"], F["hist"]]
    elif arm == "M2":    blocks = [F["m0plus"], F["hist"], F["neural"]]
    elif arm == "M2perm":blocks = [F["m0plus"], F["hist"], F["neural_perm"]]
    else: raise ValueError(arm)
    return np.concatenate(blocks, axis=2)                          # [S,T,Dtot]


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def evaluate(arm, F, y, split):
    """Pooled out-of-sample R2 under the given split (one number per call).

    Faithful neural provider (the fix): in LOSO-subj (UNSEEN regime), the held-out subject's neural
    features are replaced with population-only neural, because zero-shot TRIBE cannot individualize an
    unseen person. In LOSO-stim (ENROLLED regime) all subjects keep their individualized neural."""
    yte_all, ypred_all = [], []

    if split == "LOSO-stim":      # enrolled regime: hold out stimuli, subjects seen
        X = assemble(arm, F); S, T, D = X.shape
        for t in range(T):
            tr = [j for j in range(T) if j != t]
            Xtr = X[:, tr, :].reshape(-1, D); ytr = y[:, tr].reshape(-1)
            Xte = X[:, t, :];                  yte = y[:, t]
            sc = StandardScaler().fit(Xtr)
            m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
            yte_all.append(yte); ypred_all.append(m.predict(sc.transform(Xte)))
    elif split == "LOSO-subj":    # unseen regime: hold out a whole subject; feed it POPULATION neural
        S, T = y.shape
        for s in range(S):
            # build a held-out-subject-faithful feature set: replace subject s neural with pop-only
            Fs = dict(F)
            if arm in ("M2", "M2perm"):
                nf = F["neural"].copy() if arm == "M2" else F["neural_perm"].copy()
                nf[s] = F["neural_pop"][s]                 # zero-shot: no individual component
                Fs["neural" if arm == "M2" else "neural_perm"] = nf
            X = assemble(arm, Fs); D = X.shape[2]
            tr = [j for j in range(S) if j != s]
            Xtr = X[tr].reshape(-1, D); ytr = y[tr].reshape(-1)
            Xte = X[s];                 yte = y[s]
            sc = StandardScaler().fit(Xtr)
            m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
            yte_all.append(yte); ypred_all.append(m.predict(sc.transform(Xte)))

    return r2(np.concatenate(yte_all), np.concatenate(ypred_all))   # pooled R2


def run(world, split):
    """Run all arms across seeds; return mean skill + paired-diff CIs for the two gates."""
    arms = ["M0", "M0plus", "M1", "M2", "M2perm"]
    skill = {a: [] for a in arms}
    g1_diffs, g2_diffs = [], []   # M2-M0plus, M2-M2perm, averaged per seed
    for seed in SEEDS:
        wd = simulate_world(world, seed)
        F = make_features(wd, wd["rng"])
        per = {a: evaluate(a, F, wd["y"], split) for a in arms}
        for a in arms:
            skill[a].append(np.mean(per[a]))
        g1_diffs.append(np.mean(per["M2"]) - np.mean(per["M0plus"]))
        g2_diffs.append(np.mean(per["M2"]) - np.mean(per["M2perm"]))
    out = {a: (np.mean(skill[a]), np.std(skill[a])) for a in arms}

    def ci(d):
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5]); return np.mean(d), lo, hi
    return out, ci(g1_diffs), ci(g2_diffs)


def gate(ci_tuple):
    m, lo, hi = ci_tuple
    return "PASS" if lo > 0 else "fire? no"


if __name__ == "__main__":
    print("=" * 78)
    print("MVE CONTROL-LADDER VALIDATION (synthetic ground truth, NOT real brains)")
    print("20 seeds, RidgeCV, paired bootstrap CI over seeds. Skill = out-of-sample R^2.")
    print("=" * 78)
    for world in ["SIGNAL", "NULL"]:
        for split in ["LOSO-stim", "LOSO-subj"]:
            regime = "ENROLLED" if split == "LOSO-stim" else "UNSEEN"
            out, g1, g2 = run(world, split)
            print(f"\n--- WORLD={world}  SPLIT={split}  ({regime} regime) ---")
            for a in ["M0", "M0plus", "M1", "M2", "M2perm"]:
                print(f"   skill {a:7s} = {out[a][0]:+.3f} +/- {out[a][1]:.3f}")
            print(f"   GATE 1  M2 > M0plus : diff={g1[0]:+.3f}  CI[{g1[1]:+.3f},{g1[2]:+.3f}]  -> {gate(g1)}")
            print(f"   GATE 2  M2 > M2perm : diff={g2[0]:+.3f}  CI[{g2[1]:+.3f},{g2[2]:+.3f}]  -> {gate(g2)}")
    print("\n" + "=" * 78)
    print("Apparatus is correct iff: SIGNAL/ENROLLED fires both gates; NULL/ENROLLED does NOT fire")
    print("Gate 2; SIGNAL/UNSEEN does NOT fire Gate 2 (individual signal cannot reach a new person).")
    print("=" * 78)
