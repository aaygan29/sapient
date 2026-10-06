"""vertices_to_parcels: (T, 20484) → (T, 1000) and deterministic."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix

from sapient2 import vertices_to_parcels


def _toy_parcel_matrix(n_parcels: int = 1000, n_vertices: int = 20484) -> csr_matrix:
    rng = np.random.default_rng(0)
    rows, cols, data = [], [], []
    assignment = rng.integers(0, n_parcels, size=n_vertices)
    for p in range(n_parcels):
        idx = np.where(assignment == p)[0]
        if len(idx) == 0:
            continue
        w = 1.0 / len(idx)
        rows.extend([p] * len(idx))
        cols.extend(idx.tolist())
        data.extend([w] * len(idx))
    return csr_matrix((data, (rows, cols)), shape=(n_parcels, n_vertices),
                      dtype=np.float32)


def test_shape() -> None:
    T = 50
    M = _toy_parcel_matrix()
    v = torch.randn(T, 20484)
    p = vertices_to_parcels(v, M)
    assert p.shape == (T, 1000), p.shape


def test_deterministic() -> None:
    T = 10
    M = _toy_parcel_matrix()
    v = torch.randn(T, 20484)
    p1 = vertices_to_parcels(v, M)
    p2 = vertices_to_parcels(v, M)
    np.testing.assert_array_equal(p1, p2)


def test_batched_shape() -> None:
    B, T = 2, 20
    M = _toy_parcel_matrix()
    v = torch.randn(B, T, 20484)
    p = vertices_to_parcels(v, M)
    assert p.shape == (B, T, 1000), p.shape
