"""Small reusable nn.Modules.

In v1 the bulk of the model lives in `model.py`. This file exists so that
CODEBASE_MAP §4's "InputProjection / TransformerBlock / LowRankHead" line is
not a lie — but we deliberately rely on PyTorch's built-in
`nn.Linear` and `nn.TransformerEncoderLayer` rather than re-implementing
those. If we ever need a custom attention pattern, it goes here.
"""

from __future__ import annotations

import torch
from torch import nn


class LowRankHead(nn.Module):
    """Two-stage brain head: Linear(hidden → low_rank, no bias) → Linear(low_rank → V).

    Identical to the inline implementation in `SapientModel`. Provided as a
    standalone module for re-use by ablations / probes that want to swap the
    head without rebuilding the whole model.
    """

    def __init__(self, hidden: int, low_rank_dim: int, n_vertices: int):
        super().__init__()
        self.low_rank = nn.Linear(hidden, low_rank_dim, bias=False)
        self.brain_head = nn.Linear(low_rank_dim, n_vertices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.brain_head(self.low_rank(x))
