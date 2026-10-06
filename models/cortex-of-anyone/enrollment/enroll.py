"""Few-shot enrollment — the Cortex of Anyone keystone.

Freeze the backbones + group head; fit ONLY a new rank-r per-subject head on K
samples of a new person's data. This is the literal "make a copy of a digital
brain from a person's signals" step, expressed against Mary's additive mechanism.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .additive_head import AdditiveEncoder


def enroll_head(encoder: AdditiveEncoder, X_k: np.ndarray, Y_k: np.ndarray,
                rank: int = 16, ridge: float = 1.0) -> np.ndarray:
    """Estimate a new subject's per-subject delta (V×D) from K (stimulus, response) pairs.

    1. residual after the frozen group head: R = Y_k - group_predict(X_k)
    2. ridge-solve the feature→residual map M (D×V)
    3. truncate to rank r (Mary uses rank 16) for few-shot stability.
    """
    X_k = np.asarray(X_k, dtype=float)
    Y_k = np.asarray(Y_k, dtype=float)
    R = Y_k - encoder.group_predict(X_k)                       # (K, V)
    D = encoder.D
    XtX = X_k.T @ X_k + ridge * np.eye(D)
    M = np.linalg.solve(XtX, X_k.T @ R)                        # (D, V)
    delta = M.T                                                # (V, D)
    if rank and rank < min(delta.shape):
        U, s, Vt = np.linalg.svd(delta, full_matrices=False)
        delta = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    return delta


@dataclass
class BrainFile:
    """The portable digital-brain copy: the enrolled head + its provenance."""
    head: np.ndarray
    provenance: dict
    encoder_id: str = "simulated"
    fingerprint: str = field(default="")

    def save(self, path_stem: str) -> None:
        np.savez_compressed(path_stem + ".npz", head=self.head)
        with open(path_stem + ".json", "w") as fh:
            json.dump({"provenance": self.provenance,
                       "encoder_id": self.encoder_id,
                       "fingerprint": self.fingerprint,
                       "head_shape": list(self.head.shape)}, fh, indent=2)

    @staticmethod
    def load(path_stem: str) -> "BrainFile":
        head = np.load(path_stem + ".npz")["head"]
        with open(path_stem + ".json") as fh:
            meta = json.load(fh)
        return BrainFile(head, meta["provenance"], meta["encoder_id"],
                         meta.get("fingerprint", ""))
