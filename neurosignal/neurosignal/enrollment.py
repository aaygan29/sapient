"""Enrollment — make a personal digital brain from a new person's signals.

Two granularities:

  * VERTEX level (for the Mary repo): `AdditiveEncoder` + `enroll_head` express Mary's
    exact contract  out = group_head(x) + per_subject_head[idx](x)  (rank-16 per subject).
    Fit ONLY a new rank-r head on K samples of a new person; freeze the backbones + group
    head. `MaryAdapter` documents the one-class swap to wire the real frozen Mary model.

  * NETWORK level (runnable here in neurosignal): `EnrolledEncoder` wraps any base Encoder
    and applies a per-subject Yeo-7 network adjustment learned by `enroll_network_head`, so
    a real scan run through `analyze(..., brain_file=...)` produces PERSONALIZED results
    instead of the average-brain read-out.

A `BrainFile` is the portable copy of the digital brain (head + provenance + fingerprint).

Validated end-to-end on a faithful simulator in
`~/Desktop/Research/Neuro-AI/cortex-of-anyone/` (H1 enrolled > average, H3 identity
specificity, H5 conformal coverage; negative control fails correctly).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .validation import Provenance, sha256_array

# ============================================================== VERTEX level (Mary)


class AdditiveEncoder:
    """out = X @ (group_W + head).T ; head is the per-subject delta (V×D), or None.
    Mirrors Mary's additive group + per_subject vertex head."""

    def __init__(self, group_W: np.ndarray):
        self.group_W = np.asarray(group_W, dtype=float)

    @property
    def V(self) -> int:
        return self.group_W.shape[0]

    @property
    def D(self) -> int:
        return self.group_W.shape[1]

    def group_predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.group_W.T

    def predict(self, X, head=None) -> np.ndarray:
        W = self.group_W if head is None else self.group_W + head
        return np.asarray(X, dtype=float) @ W.T


def enroll_head(encoder: AdditiveEncoder, X_k, Y_k, rank: int = 16,
                ridge: float = 5.0) -> np.ndarray:
    """Fit a new subject's rank-r per-subject delta (V×D) from K (stimulus, BOLD) pairs.
    Freeze the group head; ridge-solve the feature→residual map; truncate to rank r."""
    X_k = np.asarray(X_k, dtype=float)
    Y_k = np.asarray(Y_k, dtype=float)
    R = Y_k - encoder.group_predict(X_k)
    XtX = X_k.T @ X_k + ridge * np.eye(encoder.D)
    M = np.linalg.solve(XtX, X_k.T @ R)
    delta = M.T
    if rank and rank < min(delta.shape):
        U, s, Vt = np.linalg.svd(delta, full_matrices=False)
        delta = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    return delta


class MaryAdapter(AdditiveEncoder):  # pragma: no cover - requires the Mary checkpoint
    """Wire the real frozen Mary model here.

    Implement `group_predict(features)` to call Mary's frozen group head on cached
    backbone features (returns predicted fsaverage5 vertices), then `enroll_head` and
    everything downstream is UNCHANGED. The simulator in cortex-of-anyone validates the
    statistics; only this substrate swap is needed for a real Cortex-of-Anyone result.
    """

    def __init__(self, mary_model=None):
        self._model = mary_model

    def group_predict(self, features):
        raise NotImplementedError(
            "Wire Mary's frozen group head: features -> predicted 20,484 fsaverage5 vertices."
        )


# ============================================================ NETWORK level (live)


@dataclass
class BrainFile:
    """Portable digital-brain copy. `network_head` personalizes neurosignal read-outs;
    `head` (optional) is the vertex-level Mary per-subject delta."""
    subject: str = "anon"
    network_head: dict | None = None            # {Yeo-7 network: additive delta in activation units}
    head: np.ndarray | None = None              # (V, D) vertex delta, for the Mary repo
    provenance: dict = field(default_factory=dict)
    encoder_id: str = "neurosignal"
    fingerprint: str = ""

    def save(self, stem: str) -> None:
        arrays = {}
        if self.head is not None:
            arrays["head"] = self.head
        np.savez_compressed(stem + ".npz", **arrays)
        with open(stem + ".json", "w") as fh:
            json.dump({"subject": self.subject, "network_head": self.network_head,
                       "provenance": self.provenance, "encoder_id": self.encoder_id,
                       "fingerprint": self.fingerprint,
                       "has_vertex_head": self.head is not None}, fh, indent=2)

    @staticmethod
    def load(stem: str) -> "BrainFile":
        with open(stem + ".json") as fh:
            meta = json.load(fh)
        head = None
        if meta.get("has_vertex_head"):
            head = np.load(stem + ".npz")["head"]
        return BrainFile(subject=meta["subject"], network_head=meta.get("network_head"),
                         head=head, provenance=meta.get("provenance", {}),
                         encoder_id=meta.get("encoder_id", "neurosignal"),
                         fingerprint=meta.get("fingerprint", ""))


class EnrolledEncoder:
    """Wrap a base Encoder; apply a per-subject Yeo-7 adjustment from a BrainFile.

    encode(features) = clip(base.encode(features) + network_head, 0, 1), per network.
    Networks the base did not speak to are left untouched (coverage stays honest)."""

    def __init__(self, base_encoder, brain_file: BrainFile):
        self._base = base_encoder
        self._bf = brain_file
        self.name = f"enrolled[{brain_file.subject}] <- {getattr(base_encoder, 'name', 'encoder')}"

    def encode(self, features) -> dict[str, float]:
        base = self._base.encode(features)
        head = self._bf.network_head or {}
        return {net: float(np.clip(val + head.get(net, 0.0), 0.0, 1.0))
                for net, val in base.items()}


def enroll_network_head(base_encoder, calibration, subject: str = "anon",
                        provenance: Provenance | None = None) -> BrainFile:
    """Learn a person's Yeo-7 adjustment from calibration pairs (network-level few-shot).

    calibration: iterable of (features, measured_networks) for the NEW subject, where
    measured_networks is {Yeo-7 network: activation in [0,1]} from their actual scan.
    delta[net] = mean over samples of (measured[net] - base_pred[net]).  This is the
    network-granularity analogue of Mary's per-subject vertex head.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    n_samples = 0
    for features, measured in calibration:
        n_samples += 1
        pred = base_encoder.encode(features)
        for net, mv in measured.items():
            if net in pred:
                sums[net] = sums.get(net, 0.0) + (float(mv) - float(pred[net]))
                counts[net] = counts.get(net, 0) + 1
    head = {net: sums[net] / counts[net] for net in sums if counts[net] > 0}
    prov = (provenance or Provenance(encoder_id=getattr(base_encoder, "name", "encoder"),
                                     enrollment_modality="network-calibration",
                                     minutes_of_data=float(n_samples))).to_dict()
    bf = BrainFile(subject=subject, network_head=head, provenance=prov,
                   encoder_id=getattr(base_encoder, "name", "encoder"))
    if head:
        payload = ";".join(f"{k}:{head[k]:.6f}" for k in sorted(head))
        bf.fingerprint = sha256_array(np.frombuffer(payload.encode(), dtype=np.uint8))[:16]
    return bf
