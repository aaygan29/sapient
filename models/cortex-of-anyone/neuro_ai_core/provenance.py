"""Provenance manifest — one stamp per brain_file / per result. Retires the
program-wide G5 gap: encoder commit + data checksums + seeds + enrollment context,
hashed into a short fingerprint so a drifting upstream silently invalidates nothing.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_array(arr) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(arr)).tobytes())


@dataclass
class Provenance:
    encoder_id: str
    encoder_commit: str
    seed: int
    enrollment_modality: str          # 'fmri' | 'eeg' | 'eeg+fmri' | 'simulated'
    minutes_of_data: float
    calibration_stimulus: str
    data_checksums: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    extra: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return sha256_bytes(payload)[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d
