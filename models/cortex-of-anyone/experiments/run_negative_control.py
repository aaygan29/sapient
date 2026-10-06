"""Negative control — proof the gates are not rigged to always pass.

Run the identical pipeline with individuation switched OFF (var_indiv = 0): the
population is group + noise only, with no person-specific signal to recover. The
preregistered kill criteria MUST now fail:
  H1 (enrolled > average)  -> FAIL  (nothing to enroll; enrolled ≈ average)
  H3 (identity specificity)-> FAIL  (no identity to recover; ≈ chance)
If they passed here, the test would be measuring an artifact, not individuation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_mve import CFG, run_one_encoder  # noqa: E402

if __name__ == "__main__":
    cfg = dict(CFG)
    cfg.update(var_group=0.30, var_indiv=0.00, var_noise=0.70)  # NO individuation
    print("=" * 74)
    print("NEGATIVE CONTROL — individuation OFF (var_indiv=0). Gates must FAIL.")
    print("=" * 74)
    ok = True
    for enc_seed in (0, 1):
        findings, prov = run_one_encoder(enc_seed, cfg=cfg)
        h1, h3, h5 = findings
        print(f"\n--- Encoder {enc_seed} ---")
        for f in findings:
            print(" ", repr(f))
        # we WANT H1 and H3 to be UNVALIDATED here
        if h1.passed or h3.passed:
            ok = False
    print("\n" + "=" * 74)
    print("CONTROL VERDICT:",
          "PASS — gates correctly fail with no individuation (not rigged)"
          if ok else "PROBLEM — a gate passed on null individuation (artifact!)")
    sys.exit(0 if ok else 1)
