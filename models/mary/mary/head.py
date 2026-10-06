"""Mary vertex head (step 8) — group linear + per-subject (low-rank) linear, additive.

From ORCLE-Nano §3 step 8 and the key design principle (Paper B): the *per-subject*
component is what makes the model work — an average-subject-only model is anti-correlated
with real visual cortex. So:

    output = group_head(x) + per_subject_head[subject_idx](x)

where `group_head: Linear(d_model → n_vertices)` is shared across subjects and the
per-subject term is selected by `subject_idx` and added.

**Per-subject term is LOW-RANK** (default rank=16). The guide §3 step 8 explicitly flags
this: "Consider a low-rank per-subject term to keep params down." A full dense per-subject
linear is `n_subjects × n_vertices × d_model` (≈141M at 9 subjects / 20484 / 768), which
blows past the paper's ~3.9M vertex-head budget (§3 param table). The low-rank factorization
    per_subject[s](x) = (x @ A_s) @ B_s + bias_s,   A_s: (d_model, r), B_s: (r, n_vertices)
costs `n_subjects × r × (d_model + n_vertices)` ≈ 3.0M at r=16 — matching the paper budget
while preserving the faithful additive group + per-subject structure. Set `per_subject_rank=0`
(or None) for a full-dense per-subject head if ever desired.

This is the largest single adapter component and dominates early-training gradient flow —
mitigated by the group+per-subject split.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class VertexHead(nn.Module):
    """Group (shared) linear + per-subject low-rank linear, selected by subject_idx, additive."""

    def __init__(
        self,
        d_model: int,
        n_vertices: int,
        n_subjects: int,
        per_subject_rank: Optional[int] = 16,
        group_rank: Optional[int] = 128,
    ):
        super().__init__()
        self.n_subjects = n_subjects
        self.per_subject_rank = per_subject_rank
        self.group_rank = group_rank

        # Shared group head. Low-rank two-stage by default (matches sapient1's
        # low_rank → brain_head pattern and the paper's ~3.9M head budget §3): a full
        # dense Linear(768, 20484) alone is 15.7M. group_rank=0/None → full dense.
        if group_rank and group_rank > 0:
            self.group_head = nn.Sequential(
                nn.Linear(d_model, group_rank, bias=False),
                nn.Linear(group_rank, n_vertices),
            )
        else:
            self.group_head = nn.Linear(d_model, n_vertices)
        self.subject_bias = nn.Parameter(torch.zeros(n_subjects, n_vertices))

        if per_subject_rank and per_subject_rank > 0:
            # Low-rank factor per subject: A (d_model→r), B (r→n_vertices).
            self.subject_A = nn.Parameter(torch.empty(n_subjects, d_model, per_subject_rank))
            self.subject_B = nn.Parameter(torch.zeros(n_subjects, per_subject_rank, n_vertices))
            nn.init.normal_(self.subject_A, std=0.02)
            self.subject_weight = None
        else:
            # Full-dense fallback: (n_subjects, n_vertices, d_model).
            self.subject_weight = nn.Parameter(torch.empty(n_subjects, n_vertices, d_model))
            nn.init.kaiming_uniform_(self.subject_weight, a=5 ** 0.5)
            self.subject_A = None
            self.subject_B = None

    def forward(self, x: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        # x: (B, T_TR, d_model); subject_idx: (B,)
        group = self.group_head(x)                       # (B, T_TR, n_vertices)
        b = self.subject_bias[subject_idx]               # (B, n_vertices)
        if self.subject_A is not None:
            A = self.subject_A[subject_idx]              # (B, d_model, r)
            B = self.subject_B[subject_idx]              # (B, r, n_vertices)
            subj = torch.bmm(torch.bmm(x, A), B)         # (B, T, n_vertices)
        else:
            w = self.subject_weight[subject_idx]         # (B, n_vertices, d_model)
            subj = torch.bmm(x, w.transpose(1, 2))       # (B, T, n_vertices)
        subj = subj + b.unsqueeze(1)
        return group + subj
