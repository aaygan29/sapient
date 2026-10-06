"""The validated-number contract. Every reported number is a Finding, never a bare
scalar. A Finding that has not cleared its preregistered test is passed=False and
renders as UNVALIDATED. Generalized from spikeprint/validate.py."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Tuple


@dataclass
class Finding:
    name: str
    value: float
    dataset: str
    baseline: float
    effect_size: float          # Cohen's d, Δ, or ratio — unit named in `note`
    ci95: Tuple[float, float]
    n: int
    passed: bool
    note: str = ""

    def __repr__(self) -> str:
        tag = "PASS" if self.passed else "UNVALIDATED"
        lo, hi = self.ci95
        out = (f"Finding[{tag}] {self.name}={self.value:.4f} "
               f"(baseline={self.baseline:.4f}, eff={self.effect_size:.3f}, "
               f"95%CI=[{lo:.4f},{hi:.4f}], n={self.n})")
        return out + (f" :: {self.note}" if self.note else "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci95"] = [float(self.ci95[0]), float(self.ci95[1])]
        return d


class UnvalidatedClaimError(RuntimeError):
    pass


def gate(*findings: "Finding") -> None:
    """Raise if any Finding is unvalidated. Call before exporting/serving any claim."""
    bad = [f for f in findings if not f.passed]
    if bad:
        raise UnvalidatedClaimError(
            "Refusing to emit unvalidated claims:\n"
            + "\n".join(f"  - {f!r}" for f in bad)
        )


def dump_findings(path: str, findings) -> None:
    with open(path, "w") as fh:
        json.dump([f.to_dict() for f in findings], fh, indent=2)
