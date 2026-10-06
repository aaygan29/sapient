"""
loaders.py — real, openly-licensed datasets for exercising the validation harness.

These are NOT neural validation sets (that needs Mary vertex exports + measured ad
outcomes). They are real behavioral-outcome data used to prove the forecasting and
validation machinery runs honestly on real numbers, and to give the harness something
to regress against before your own data arrives.

Datasets
--------
advertising : ISLR "Advertising" — 200 markets, ad spend by channel (TV/radio/newspaper)
              -> product sales. Openly mirrored teaching dataset. Real behavioral outcome.
              Use it as a smoke test: does Neuroforecaster's LOCO + permutation machinery
              recover the (real) channel->sales relationship and report an honest r?

Pointers to real NEURAL validation sources (fetch when you have access/compute):
  - OpenNeuro neuromarketing / value studies (e.g., ds-series with reward/choice tasks)
  - Kühn et al. 2016 point-of-sale paradigm (NeuroImage) — chocolate sales forecast
  - Falk et al. 2012/2016 health-campaign paradigms (MPFC self-localizer)
See ../RESEARCH_BASIS.md for the full citation table.
"""
from __future__ import annotations
import os
import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "raw")


def load_advertising() -> dict:
    """
    Load the ISLR Advertising dataset from data/raw/Advertising.csv.

    Returns dict with:
      spend:   (n, 3) TV / radio / newspaper spend
      sales:   (n,)   product sales (the behavioral outcome)
      channels: column names
    """
    path = os.path.join(RAW, "Advertising.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is openly mirrored; fetch with:\n"
            "  curl -sL -o data/raw/Advertising.csv "
            "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
        )
    # parse without pandas to keep the engine dependency-light
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            rows.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    arr = np.asarray(rows, float)
    return {
        "spend": arr[:, :3],
        "sales": arr[:, 3],
        "channels": header[1:4],
        "n": len(arr),
        "source": "ISLR Advertising (openly mirrored)",
    }
