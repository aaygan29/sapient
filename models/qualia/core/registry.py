"""qualia.core.registry — Mary model channels.

Two channels decouple "the standard" from "the latest brain":

  - instrument : a FROZEN, version-pinned Mary checkpoint. `align()` (the Brain
                 Alignment / Social Brain Score) ALWAYS runs against this so
                 scores stay comparable over time and across customers. Bump it
                 only deliberately (an announced "Brain Alignment v2").
  - improving  : the latest/best Mary (e.g. retrained by the flywheel). The demo,
                 `understand()`, and `state()` ride this — so a Mary UPGRADE flows
                 through with zero code change.

To upgrade: train a better Mary -> drop the checkpoint on the volume -> point the
channel here (or via env QUALIA_INSTRUMENT_CKPT / QUALIA_IMPROVING_CKPT). Nothing
else in Qualia changes.
"""
from __future__ import annotations

import os

# Checkpoints live on the Modal volume `sapient-data` at /data/checkpoints/.
# Verified 2026-05-30: mary_multi_v2_s13 is the strongest COMPLETE core
# (27 epochs, val vertex_pearson_mean 0.0127 / top10pct 0.0434). The Full-tier
# (mary_full_s13) is only ~10 epochs in — still training; when it finishes and
# surpasses v2, point `improving` at it and the demo/understand/state upgrade
# with NO code change (the engine->layer compounding). Seeds s13/s17/s23/s29/s37
# exist for a future ensemble bump. Override per env for dev / to roll a version.
# UPGRADE 2026-06-03: the prior served checkpoint (mary_multi_v2_s23) was COLLAPSED —
# whole-brain val r ≈ 0.0005, ~noise even on the most reliable vertices (an unstable
# scaled run; LR 3e-4 too hot for the 1024-d model → it never learned / decayed). The
# "recovery" that pointed here had repointed to a non-predictive run. Diagnosed and
# retrained: mary_multi_stable (same 1024-d capacity, LR lowered to 1e-4 + long warmup)
# trains cleanly and its best.pt is genuinely predictive AND video-capable —
# whole-brain r 0.0284 (~48% of the ISC noise ceiling), responsive-vertex r 0.034,
# top10pct 0.037 on the full 5-dataset val (verified via scripts/eval_against_ceiling.py).
# That's ~30x the collapsed s23. Repointed BOTH channels to it. NOTE: we use best.pt —
# selection is now made on a denoised signal, and this run peaked early (epoch 7) then
# drifted, so best.pt (the peak) is correct here, not latest.pt.
# UPGRADE 2026-06-07: cut over to mary_multi_stable_resp_s13/best.pt — the
# responsive-trained core (masked loss on ISC>0.05 vertices) with responsive-vertex
# r 0.1157 (~3.4x the prior stable_s13's 0.034). Paired with serve.py's responsive
# un-flatten (each Yeo-7 reduction restricted to ISC>0.05 vertices, ISC-weighted),
# this both predicts the reliable signal better AND differentiates content instead of
# collapsing to the old ~0.075-flat all-vertex mean. BOTH channels repointed.
# Calibration: KPI_BASELINE in modal/serve.py + src/mary/MaryPillars.tsx were recomputed
# against THIS checkpoint's output THROUGH the responsive un-flatten (build_kpi_baseline.py
# applies the same ISC>0.05 ISC-weighted reduction serve uses), so the 0-100 scores stay
# calibrated. Raw KPI means are baseline-independent given a fixed reduction.
CHANNELS: dict[str, dict] = {
    # Pinned standard for align() — bump deliberately/announced only.
    "instrument": {"version": "stable-resp-s13", "ckpt": "/data/checkpoints/mary_multi_stable_resp_s13/best.pt"},
    # Rides upgrades — point at the Full-tier / ensemble when it surpasses this.
    "improving":  {"version": "stable-resp-s13", "ckpt": "/data/checkpoints/mary_multi_stable_resp_s13/best.pt"},
}


def resolve(channel: str = "improving") -> dict:
    """Return {channel, version, ckpt} for a channel, honoring an env override.

    The demo and understand/state use 'improving' (rides Mary upgrades). align()
    uses 'instrument' (pinned). env QUALIA_<CHANNEL>_CKPT overrides the path.
    """
    if channel not in CHANNELS:
        raise KeyError(f"unknown channel {channel!r}; expected one of {list(CHANNELS)}")
    d = dict(CHANNELS[channel])
    d["channel"] = channel
    override = os.environ.get(f"QUALIA_{channel.upper()}_CKPT")
    if override:
        d["ckpt"] = override
        d["version"] = d["version"] + "+local"
    return d
