"""Dependency-light tests for the enrollment engine (run: python3 tests/test_enrollment.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enrollment import (  # noqa: E402
    BrainFile, enroll_head, make_population, observe, sample_stimuli, vertex_pearson,
)

_failures = []


def check(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    if not cond:
        _failures.append(name)


def test_enroll_beats_group_and_wrong_subject():
    pop = make_population(8, D=64, V=200, rank=8,
                          var_group=0.30, var_indiv=0.30, var_noise=0.40, seed=3)
    s = 0
    Xk = sample_stimuli(64, 200, seed=11)
    Yk = observe(pop, s, Xk, seed=12)
    head = enroll_head(pop.encoder, Xk, Yk, rank=16, ridge=5.0)

    Xt = sample_stimuli(64, 300, seed=13)
    Yt = observe(pop, s, Xt, seed=14)
    r_enrolled = vertex_pearson(Yt, pop.encoder.predict(Xt, head))
    r_group = vertex_pearson(Yt, pop.encoder.group_predict(Xt))

    # enroll a DIFFERENT subject's head and apply it to subject s -> should be worse
    Xk2 = sample_stimuli(64, 200, seed=21)
    Yk2 = observe(pop, 1, Xk2, seed=22)
    wrong = enroll_head(pop.encoder, Xk2, Yk2, rank=16, ridge=5.0)
    r_wrong = vertex_pearson(Yt, pop.encoder.predict(Xt, wrong))

    check(f"enrolled ({r_enrolled:.3f}) > group-only ({r_group:.3f})", r_enrolled > r_group)
    check(f"own head ({r_enrolled:.3f}) > wrong-subject head ({r_wrong:.3f})", r_enrolled > r_wrong)
    check("enrolled head shape is (V, D)", head.shape == (200, 64))


def test_brainfile_roundtrip(tmp="/tmp/coa_brainfile_test"):
    pop = make_population(4, seed=5)
    Xk = sample_stimuli(pop.encoder.D, 150, seed=31)
    Yk = observe(pop, 0, Xk, seed=32)
    head = enroll_head(pop.encoder, Xk, Yk)
    bf = BrainFile(head, provenance={"modality": "simulated", "minutes": 150},
                   encoder_id="sim", fingerprint="deadbeef")
    bf.save(tmp)
    bf2 = BrainFile.load(tmp)
    check("brain_file head round-trips", np.allclose(bf.head, bf2.head))
    check("brain_file provenance round-trips", bf2.provenance["modality"] == "simulated")
    for ext in (".npz", ".json"):
        if os.path.exists(tmp + ext):
            os.remove(tmp + ext)


if __name__ == "__main__":
    for t in (test_enroll_beats_group_and_wrong_subject, test_brainfile_roundtrip):
        print(f"\n[{t.__name__}]")
        t()
    print("\n" + ("ALL ENROLLMENT TESTS PASSED" if not _failures else f"FAILURES: {_failures}"))
    sys.exit(1 if _failures else 0)
