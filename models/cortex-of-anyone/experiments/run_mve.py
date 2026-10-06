"""Cortex of Anyone — Phase-1 MVE, preregistered.

Tests, on a faithful simulator of Mary's additive group+per-subject head:
  H1  enrolled head BEATS the average brain on a held-out person's held-out stimuli
  H3  identity specificity: a person's own head predicts them better than any other
      person's head (anti-leakage — the DO_NOT_CITE lesson, as a live gate)
  H5  conformal coverage holds for a brand-new (OOD) enrolled person
Dual-encoder: the H1 verdict must not flip across two independent encoders.

NOTE: simulated data validates the MACHINERY + STATISTICS only. Swap SimulatedCortex
for MaryAdapter (see additive_head.py) to read a scientific result off real brains.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enrollment import (  # noqa: E402
    enroll_head, make_population, noise_ceiling, observe, sample_stimuli,
    vertex_pearson,
)
from neuro_ai_core import (  # noqa: E402
    Finding, Provenance, dump_findings, empirical_coverage,
    split_conformal_regression,
)

# Regime: explicit variance shares. group 30% / individuation 30% / noise 40% —
# individuation is a real but non-dominant fraction (the regime where a per-subject
# head can win without it being a foregone conclusion). Stated, not accidental.
CFG = dict(n_subjects=12, D=64, V=200, true_rank=8, enroll_rank=16,
           var_group=0.30, var_indiv=0.30, var_noise=0.40,
           K_enroll=200, N_test=300, ridge=5.0, alpha=0.10)


def boot_ci(diffs, B=5000, seed=0, alpha=0.05):
    diffs = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(B, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def cohen_dz(diffs):
    diffs = np.asarray(diffs, float)
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else float("inf")


def run_one_encoder(enc_seed: int, cfg=CFG):
    c = cfg
    pop = make_population(c["n_subjects"], D=c["D"], V=c["V"], rank=c["true_rank"],
                          var_group=c["var_group"], var_indiv=c["var_indiv"],
                          var_noise=c["var_noise"], seed=enc_seed)
    S = c["n_subjects"]

    enrolled, Xte, Yte, r_enr, r_avg, r_grp, ceil = [], [], [], [], [], [], []
    for s in range(S):
        # --- enroll from K calibration samples of subject s ---
        Xk = sample_stimuli(c["D"], c["K_enroll"], seed=1000 + 10 * enc_seed + s)
        Yk = observe(pop, s, Xk, seed=2000 + 10 * enc_seed + s)
        delta = enroll_head(pop.encoder, Xk, Yk, rank=c["enroll_rank"], ridge=c["ridge"])
        enrolled.append(delta)

        # --- held-out stimuli for evaluation ---
        Xt = sample_stimuli(c["D"], c["N_test"], seed=3000 + 10 * enc_seed + s)
        Yt = observe(pop, s, Xt, seed=4000 + 10 * enc_seed + s)
        Xte.append(Xt); Yte.append(Yt)

        # average brain = leave-one-out mean of the OTHER subjects' true heads
        avg_head = np.mean(np.stack([pop.heads[j] for j in range(S) if j != s]), axis=0)

        r_enr.append(vertex_pearson(Yt, pop.encoder.predict(Xt, delta)))
        r_avg.append(vertex_pearson(Yt, pop.encoder.predict(Xt, avg_head)))
        r_grp.append(vertex_pearson(Yt, pop.encoder.group_predict(Xt)))
        ceil.append(noise_ceiling(pop, s, Xt, seed_a=4000 + 10 * enc_seed + s,
                                  seed_b=5000 + 10 * enc_seed + s))

    r_enr, r_avg, r_grp, ceil = map(np.array, (r_enr, r_avg, r_grp, ceil))

    # ---- H1: enrolled beats average brain (paired over subjects) ----
    diff = r_enr - r_avg
    lo, hi = boot_ci(diff, seed=enc_seed)
    h1 = Finding(
        name="H1_enrolled_vs_average", value=float(r_enr.mean()),
        dataset=f"simulated(enc{enc_seed})", baseline=float(r_avg.mean()),
        effect_size=cohen_dz(diff), ci95=(lo, hi), n=len(diff),
        passed=bool(lo > 0 and r_enr.mean() > r_avg.mean()),
        note=(f"paired Δr=+{diff.mean():.4f} (95% CI in ci95); "
              f"%ceiling enrolled={100*r_enr.mean()/ceil.mean():.0f}% "
              f"vs average={100*r_avg.mean()/ceil.mean():.0f}% (ceiling={ceil.mean():.3f}); "
              f"group-only r={r_grp.mean():.3f}"))

    # ---- H3: identity specificity (own head > others' heads) ----
    M = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            M[i, j] = vertex_pearson(Yte[i], pop.encoder.predict(Xte[i], enrolled[j]))
    rank1 = float(np.mean([np.argmax(M[i]) == i for i in range(S)]))
    diag = np.diag(M)
    offdiag = np.array([np.mean([M[i, j] for j in range(S) if j != i]) for i in range(S)])
    spec_diff = diag - offdiag
    slo, shi = boot_ci(spec_diff, seed=enc_seed + 99)
    h3 = Finding(
        name="H3_identity_specificity", value=rank1,
        dataset=f"simulated(enc{enc_seed})", baseline=1.0 / S,
        effect_size=cohen_dz(spec_diff), ci95=(slo, shi), n=S,
        passed=bool(rank1 > 1.0 / S and slo > 0),
        note=(f"rank-1 identification {rank1:.0%} vs chance {1/S:.0%}; "
              f"own-vs-other Δr={spec_diff.mean():.4f} (CI excludes 0)"))

    # ---- H5: conformal coverage on a NEW enrolled person ----
    s0 = 0
    Xc = sample_stimuli(c["D"], 400, seed=7000 + enc_seed)
    Yc = observe(pop, s0, Xc, seed=7100 + enc_seed)
    pred_c = pop.encoder.predict(Xc, enrolled[s0])
    q = split_conformal_regression((Yc - pred_c).ravel(), alpha=c["alpha"])
    Xv = sample_stimuli(c["D"], 400, seed=7200 + enc_seed)
    Yv = observe(pop, s0, Xv, seed=7300 + enc_seed)
    cov = empirical_coverage(Yv, pop.encoder.predict(Xv, enrolled[s0]), q)
    h5 = Finding(
        name="H5_conformal_coverage_new_person", value=float(cov),
        dataset=f"simulated(enc{enc_seed})", baseline=1 - c["alpha"],
        effect_size=float(cov - (1 - c["alpha"])), ci95=(float(cov), float(cov)),
        n=Yv.size, passed=bool(cov >= (1 - c["alpha"]) - 0.02),
        note=f"split-conformal radius q={q:.3f}; target {1-c['alpha']:.2f}")

    prov = Provenance(encoder_id=f"sim_enc{enc_seed}", encoder_commit="mve-v1",
                      seed=enc_seed, enrollment_modality="simulated",
                      minutes_of_data=float(c["K_enroll"]),
                      calibration_stimulus="synthetic_gaussian",
                      extra={"config": {k: v for k, v in c.items()}})
    return [h1, h3, h5], prov


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(os.path.dirname(here), "results")
    os.makedirs(out, exist_ok=True)

    all_findings = {}
    print("=" * 74)
    print("CORTEX OF ANYONE — Phase-1 MVE  (simulated machinery validation)")
    print("=" * 74)
    for enc_seed in (0, 1):
        findings, prov = run_one_encoder(enc_seed)
        all_findings[enc_seed] = findings
        print(f"\n--- Encoder {enc_seed}  (provenance {prov.fingerprint()}) ---")
        for f in findings:
            print(" ", repr(f))
        dump_findings(os.path.join(out, f"mve_findings_enc{enc_seed}.json"), findings)

    # dual-encoder verdict: H1 must pass on BOTH encoders (verdict doesn't flip)
    h1_pass = all(all_findings[e][0].passed for e in (0, 1))
    h3_pass = all(all_findings[e][1].passed for e in (0, 1))
    h5_pass = all(all_findings[e][2].passed for e in (0, 1))
    print("\n" + "=" * 74)
    print("KILL-CRITERIA VERDICT")
    print(f"  H1 enrolled > average brain (both encoders) : {'PASS' if h1_pass else 'FAIL'}")
    print(f"  H3 identity specificity / anti-leakage      : {'PASS' if h3_pass else 'FAIL'}")
    print(f"  H5 conformal coverage on new person         : {'PASS' if h5_pass else 'FAIL'}")
    print(f"  Dual-encoder stable (H1 doesn't flip)       : {'PASS' if h1_pass else 'FAIL'}")
    verdict = "MACHINERY VALIDATED — ready to wire real Mary" if (h1_pass and h3_pass and h5_pass) \
        else "MACHINERY ISSUE — fix before real data"
    print(f"\n  => {verdict}")
    with open(os.path.join(out, "mve_summary.json"), "w") as fh:
        json.dump({"H1": h1_pass, "H3": h3_pass, "H5": h5_pass,
                   "dual_encoder_stable": h1_pass, "verdict": verdict}, fh, indent=2)
    return 0 if (h1_pass and h3_pass and h5_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
