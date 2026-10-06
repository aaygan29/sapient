"""Inference-side adapters that bridge to the downstream sapienteval/ stack.

Per spec §7: sapienteval/ expects parcellated input (1000-dim per timepoint),
not vertex-level (20,484-dim). This module exposes the deterministic
post-processing step that converts model output to the parcel interface.

Importantly, no changes to sapienteval/ are required — the (T, 1000) shape
is exactly what its existing parcellate.py expects.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz


def vertices_to_parcels(
    vertex_pred: torch.Tensor | np.ndarray,   # (T, 20484) or (B, T, 20484)
    parcel_matrix: csr_matrix | np.ndarray,    # (1000, 20484) averaging matrix
) -> np.ndarray:
    """Project vertex predictions to Schaefer-1000 parcels.

    Decision #10: the parcel projection is a *post-processing* step. It runs
    once at inference, never at training time.
    """
    if isinstance(vertex_pred, torch.Tensor):
        vertex_pred = vertex_pred.detach().cpu().float().numpy()
    if isinstance(parcel_matrix, csr_matrix):
        # vertex_pred @ parcel_matrix.T preserves shape (..., 1000).
        # scipy's sparse @ dense returns dense; do it in chunks for safety.
        flat = vertex_pred.reshape(-1, vertex_pred.shape[-1])
        out = flat @ parcel_matrix.T.toarray()        # (N, 1000)
        return out.reshape(*vertex_pred.shape[:-1], parcel_matrix.shape[0])
    return vertex_pred @ parcel_matrix.T              # (..., 1000)


def load_parcellation(path: str = "data/parcellate.npz") -> csr_matrix:
    """Load the cached vertex→parcel averaging matrix built by data/prepare_fmri.py."""
    return load_npz(path)
