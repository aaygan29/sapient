"""
mary_readouts.py — drop-in corrective read-out layer for the Mary brain-encoder.

WHY THIS EXISTS
---------------
An audit of the legacy Cognitive Sovereignty Index (CSI) found that, on the
released predicted-BOLD maps, the "manipulation" read-out:
  (1) did not reconstruct from the released maps (composite r = -0.42): traced to
      a NORMALIZATION/PROVENANCE mismatch (clean-ROI dlPFC reconstructs at r=0.88
      but slope 1.31, not 1.0). An out-of-range Destrieux label (temporal_pole=77;
      atlas is 0-75) is fixed below as hygiene but is NOT the root cause;
  (2) was 78% explained by stimulus text length;
  (3) was driven by a whole-cortex global-activation offset (d = -1.46 between
      corpora);
  (4) concorded ~3-5x more with AUDITORY/VISUAL meta-analytic maps than with the
      executive/social circuitry it named (NeuroQuery 2.87x, Neurosynth 4.94x).

This module replaces hand-picked ROIs with META-ANALYTICALLY-GROUNDED spatial
weight maps for the constructs advertisers actually care about and that ARE
mapped in human fMRI, and bakes in the three corrections the audit demands:
  - correct, validated ROI/atlas handling,
  - global-signal removal + length/acoustic confound residualization,
  - a SPECIFICITY QC GATE that refuses to report a read-out unless its spatial
    basis concords with its target construct more than with low-level confounds.

GROUNDING (advertiser-relevant constructs with real fMRI + population-outcome evidence)
  value/reward (NAcc, vmPFC/MPFC)  -> choice & market-level purchase/funding
                                      (Knutson/Genevsky J Neurosci 2015/2017;
                                       Falk Psychol Sci 2012)
  engagement   (inter-subject reliability) -> audience preference (Dmochowski Nat Comm 2014)
  memory       (MTL/parahippocampal)       -> ad recall / commercial success (Boksem & Smidts JMR 2015)
  emotion/arousal (amygdala, insula)        -> affective response

The module is model-agnostic: it takes a predicted whole-cortex vector
(20,484 fsaverage5 vertices) — Mary's native output — and returns calibrated,
confound-controlled, QC-gated scores.
"""
from __future__ import annotations
import os, json, numpy as np
from dataclasses import dataclass, field

N_VERT = 20484

# Corrected Destrieux fsaverage5 label sets (0-75 valid). temporal_pole fixed:
# 'Pole_temporal' is index 44 in nilearn's Destrieux labels; 77 was invalid.
DESTRIEUX_ROIS_FIXED = {
    "dlPFC": [15, 16, 54, 55],
    "vmPFC": [1, 5, 24, 31, 63, 64, 65, 71],
    "ACC":   [6, 7],
    "insula":[18, 19, 49],          # verify circular-insula indices against atlas
    "temporal_pole": [44],          # FIX: was [38, 77]; 77 is out of range
    "OFC":   [2, 3, 22, 23],
}

# Constructs that are advertiser-relevant AND meta-analytically grounded.
CONSTRUCTS = ["value", "reward", "emotion", "arousal", "attention", "memory"]
CONFOUNDS  = ["auditory", "visual", "language"]   # low-level; a valid read-out must beat these


@dataclass
class GroundedReadouts:
    """Confound-controlled, meta-analytically-grounded read-out layer for Mary."""
    maps_dir: str                                  # dir of <construct>.npy weight maps (fsaverage5)
    construct_maps: dict = field(default_factory=dict)
    confound_maps: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in CONSTRUCTS:
            p = os.path.join(self.maps_dir, f"{name}.npy")
            if os.path.exists(p): self.construct_maps[name] = np.load(p)
        for name in CONFOUNDS:
            p = os.path.join(self.maps_dir, f"{name}.npy")
            if os.path.exists(p): self.confound_maps[name] = np.load(p)
        if not self.construct_maps:
            raise FileNotFoundError(f"No construct weight maps in {self.maps_dir}")

    # ---- correction 1: remove the global-activation offset that contaminated CSI ----
    @staticmethod
    def remove_global_signal(vmap: np.ndarray) -> np.ndarray:
        return vmap - float(np.mean(vmap))

    # ---- meta-analytic weighted read-out (replaces hand-picked ROI mean) ----
    @staticmethod
    def _weighted(vmap: np.ndarray, wmap: np.ndarray) -> float:
        w = np.clip(wmap, 0, None)            # positive meta-analytic evidence only
        s = w.sum()
        return float(np.dot(vmap, w) / s) if s > 0 else float("nan")

    def score(self, vmap: np.ndarray, gsr: bool = True) -> dict:
        """Raw grounded read-outs for one predicted vertex map."""
        assert vmap.shape == (N_VERT,)
        x = self.remove_global_signal(vmap) if gsr else vmap
        out = {k: self._weighted(x, m) for k, m in self.construct_maps.items()}
        out.update({f"_confound_{k}": self._weighted(x, m) for k, m in self.confound_maps.items()})
        return out

    # ---- correction 2: residualize stimulus confounds (length, acoustic energy) ----
    @staticmethod
    def residualize(scores: np.ndarray, confounds: np.ndarray) -> np.ndarray:
        """Regress nuisance confounds out of a per-stimulus score vector (OLS residuals)."""
        X = np.column_stack([np.ones(len(scores)), confounds])
        beta, *_ = np.linalg.lstsq(X, scores, rcond=None)
        return scores - X @ beta

    # ---- correction 3: the SPECIFICITY QC GATE (the test CSI failed) ----
    def specificity_gate(self, diff_map: np.ndarray, construct: str,
                         min_ratio: float = 1.0) -> dict:
        """
        Given a stimulus-contrast map (e.g. condition_A - condition_B), decide whether
        the contrast is attributable to `construct` rather than to low-level confounds.
        Returns the concordance of the contrast with the target construct map vs the
        max confound concordance, and PASS/FAIL.
        A read-out should NOT be reported as measuring `construct` if it FAILS.
        """
        def corr(a, b):
            a = a - a.mean(); b = b - b.mean()
            d = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / d) if d > 0 else 0.0
        tgt = abs(corr(diff_map, self.construct_maps[construct]))
        cnf = max(abs(corr(diff_map, m)) for m in self.confound_maps.values()) if self.confound_maps else 0.0
        ratio = tgt / cnf if cnf > 0 else float("inf")
        return {"construct": construct, "target_concordance": round(tgt, 3),
                "max_confound_concordance": round(cnf, 3), "ratio": round(ratio, 2),
                "pass": ratio >= min_ratio}

    def report(self, vmaps: np.ndarray, stim_confounds: np.ndarray | None = None,
               contrast: tuple | None = None) -> dict:
        """
        Full calibrated report for a set of stimuli.
        vmaps: (n_stim, 20484). stim_confounds: (n_stim, k) e.g. [word_count, audio_energy].
        contrast: (idx_A, idx_B) groups to contrast and run the specificity gate on.
        """
        raw = np.array([[self.score(v)[c] for c in self.construct_maps] for v in vmaps])
        cols = list(self.construct_maps)
        rep = {"constructs": cols}
        if stim_confounds is not None:
            corr_raw = {c: float(np.corrcoef(raw[:, i], stim_confounds[:, 0])[0, 1])
                        for i, c in enumerate(cols)}
            resid = np.column_stack([self.residualize(raw[:, i], stim_confounds) for i in range(len(cols))])
            corr_res = {c: float(np.corrcoef(resid[:, i], stim_confounds[:, 0])[0, 1])
                        for i, c in enumerate(cols)}
            rep["confound_corr_before"] = corr_raw
            rep["confound_corr_after"]  = corr_res
        if contrast is not None:
            A, B = contrast
            diff = vmaps[A].mean(0) - vmaps[B].mean(0)
            diff = self.remove_global_signal(diff)
            rep["specificity_gate"] = {c: self.specificity_gate(diff, c) for c in self.construct_maps}
        return rep
