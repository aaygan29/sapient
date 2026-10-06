"""neuro-ai-core — the shared honesty/validation layer for the Neuro-AI program.

Four primitives, default-on across every instrument:
  - Finding            : the validated-number contract (every score carries its evidence)
  - conformal          : distribution-free prediction intervals + honest abstention
  - specificity_gate   : withhold a construct score when the signal is confounded
  - Provenance         : encoder/data/seed manifest with a short fingerprint
"""
from .conformal import (
    conformal_intervals,
    empirical_coverage,
    prediction_sets,
    split_conformal_classification,
    split_conformal_regression,
)
from .finding import Finding, UnvalidatedClaimError, dump_findings, gate
from .provenance import Provenance, sha256_array, sha256_bytes
from .specificity import concordance, specificity_gate

__all__ = [
    "Finding", "gate", "dump_findings", "UnvalidatedClaimError",
    "split_conformal_regression", "empirical_coverage", "conformal_intervals",
    "split_conformal_classification", "prediction_sets",
    "specificity_gate", "concordance",
    "Provenance", "sha256_array", "sha256_bytes",
]
