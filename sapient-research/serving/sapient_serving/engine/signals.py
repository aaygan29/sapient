"""Consumer-neuro signal layer: map predicted cortical activation -> interpretable
constructs and a buy/sell (purchase-intent) timeline.

NEUROFORECASTING BASIS
----------------------
The composite follows the established "approach minus avoidance" neuroforecasting
form (Knutson et al., 2007, *Neuron* — NAcc/mPFC reward predicts purchase; anterior
insula "pain of paying" predicts NOT buying), corroborated and extended by the
papers supplied for this build:

  * Kapoor, Sahay, Singh, Pammi & Banerjee (2023), *J. Business Research* 154:113230.
    fMRI of strong/weak brand choice. Preferred/strong choice -> LEFT VENTRAL STRIATUM
    (reward) + L-DLPFC. Hesitant/weak choice -> ROSTRAL + DORSAL ACC (conflict, error
    commission, regret) + R-DLPFC (attentional conflict, top-down control). ACC
    conflict scales with longer response time. => conflict/regret is a BUY-suppressing
    ("Sell"/avoid) signal; ventral-striatum reward is the BUY signal.
  * Çakir, Çakar, Girisken & Yurdakul (2018), *Eur. J. Marketing* 52(1/2):224.
    fNIRS. POSITIVE purchase decisions raise activity in FRONTO-POLAR / OFC / vmPFC
    (subjective value). Buy/pass decoded at 85% once budget sensitivity is added.
    => reward/value lives in vmPFC/OFC/fronto-polar; price/budget sensitivity matters.
  * Shang, Deng & Liu (2018), *NeuroQuantology* 16(5):246. ERP staging of online
    purchase: N2 (risk) -> N400 (conflict) -> LPP (value/valence). Involvement
    moderates price vs reputation. => risk/conflict precede value evaluation.
  * Vlăsceanu (2014), *Procedia SBS* 127:758. Review: vmPFC + amygdala reward & somatic
    markers; dual-process (System-1 emotion vs System-2 deliberation, gated by load);
    strong brand -> prefrontal engagement. => emotion drives approach; deliberation
    (DLPFC/cognitive load) is friction.

NETWORK MAPPING (fine regions -> Yeo-7 networks the encoder predicts)
  vmPFC / OFC / mPFC (reward, value)     -> Limbic + Default
  amygdala / affect (emotional arousal)  -> Limbic
  anterior insula + ACC (conflict, risk) -> VentralAttention (salience)
  DLPFC / dmPFC (deliberation, control)  -> Frontoparietal
  orienting attention                    -> DorsalAttention
  retinotopic / sensory                  -> Visual
  brand memory associations              -> Default

HONESTY: this is an interpretable, literature-grounded heuristic over PREDICTED
activation (predicted activity, not correlation with measured fMRI). In mock mode
the activations are illustrative. /v1/info states the provenance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import ConstructScore, PeakMoment, Signals, SignalTimeline


@dataclass(frozen=True)
class ConstructDef:
    key: str
    label: str
    networks: tuple[str, ...]
    description: str
    weight: float  # >0 = approach/buy driver, <0 = avoidance/friction (sell) driver


# Weights reflect the neuroforecasting literature above (approach drivers positive,
# friction drivers negative). Reward/value dominates approach (Knutson; Çakir; Kapoor).
CONSTRUCTS: list[ConstructDef] = [
    ConstructDef("reward", "Reward & Value", ("Limbic", "Default"),
                 "vmPFC/OFC/ventral-striatum value signal — the core 'buy' impulse "
                 "(Knutson 2007; Çakir 2018; Kapoor 2023).", 0.30),
    ConstructDef("emotion", "Emotional Resonance", ("Limbic",),
                 "Amygdala/limbic affective arousal and somatic markers (Vlăsceanu 2014).", 0.22),
    ConstructDef("attention", "Attention Capture", ("DorsalAttention",),
                 "Orienting of attention toward the stimulus.", 0.18),
    ConstructDef("visual", "Visual & Sensory Salience", ("Visual",),
                 "Retinotopic / sensory salience of the content.", 0.10),
    ConstructDef("memory", "Memory Encoding", ("Default",),
                 "Default-mode engagement; brand-memory association and recall.", 0.10),
    ConstructDef("conflict", "Decision Conflict & Risk", ("VentralAttention",),
                 "Salience-network (anterior insula + ACC) conflict, risk, regret and "
                 "'pain of paying' — suppresses buying / drives SELL (Kapoor 2023; Shang 2018).",
                 -0.22),
    ConstructDef("load", "Cognitive Load (Deliberation)", ("Frontoparietal",),
                 "DLPFC/dmPFC top-down deliberation; high System-2 load is friction "
                 "(Vlăsceanu 2014; Kapoor 2023).", -0.15),
]

_APPROACH_W = sum(c.weight for c in CONSTRUCTS if c.weight > 0)
_AVOID_W = sum(-c.weight for c in CONSTRUCTS if c.weight < 0)


def _component(construct_ts: dict[str, np.ndarray], positive: bool, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=float)
    total = 0.0
    for c in CONSTRUCTS:
        if (c.weight > 0) == positive:
            w = abs(c.weight)
            out += w * np.clip(np.asarray(construct_ts.get(c.key, np.zeros(n)), dtype=float), 0.0, 1.0)
            total += w
    return out / total if total else out


def network_ts_from_construct_ts(construct_ts: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Invert the construct->network table: per-second Yeo-7 network activation (0..1).

    Each network's activation is the mean of the constructs that read from it (the same
    table the readout uses). Networks no construct reads (e.g. Somatomotor) default to 0.5.
    Lets engines that produce per-construct timeseries (mock) drive the live digital brain.
    """
    from .postprocess import YEO7_NETWORKS

    acc = {net: [] for net in YEO7_NETWORKS}
    for c in CONSTRUCTS:
        ts = np.clip(np.asarray(construct_ts.get(c.key, np.full(n, 0.5)), dtype=float), 0.0, 1.0)
        for net in c.networks:
            if net in acc:
                acc[net].append(ts)
    return {net: (np.mean(series, axis=0) if series else np.full(n, 0.5))
            for net, series in acc.items()}


def network_ts_from_parcels(parcel_ts: np.ndarray, network_ids: np.ndarray) -> dict[str, np.ndarray]:
    """REAL path: per-second Yeo-7 network activation (0..1) from predicted parcels."""
    from .postprocess import YEO7_NETWORKS

    net = _minmax_global(_network_timeseries(np.asarray(parcel_ts, dtype=float),
                                             np.asarray(network_ids, dtype=int), len(YEO7_NETWORKS)))
    return {name: net[:, i] for i, name in enumerate(YEO7_NETWORKS)}


def build_signals(construct_ts: dict[str, np.ndarray], seconds: np.ndarray,
                  network_ts: dict[str, np.ndarray] | None = None) -> Signals:
    """construct_ts: per-construct activation timeseries in [0,1]. seconds: time axis.

    network_ts (optional): per-second Yeo-7 network activation in [0,1]. If omitted it is
    derived from construct_ts so the per-second network_timeline is always available for
    the live digital brain.
    """
    n = len(seconds)
    if network_ts is None:
        network_ts = network_ts_from_construct_ts(construct_ts, n)
    approach = _component(construct_ts, positive=True, n=n)   # 0..1
    avoid = _component(construct_ts, positive=False, n=n)     # 0..1

    # Buy intent = baseline 50 + approach deviation − half the avoidance deviation.
    # Neutral inputs (all 0.5) -> 50; high reward/low conflict -> ~Buy; high conflict -> ~Sell.
    pi = np.clip(100.0 * (0.5 + (approach - 0.5) - 0.5 * (avoid - 0.5)), 0.0, 100.0)

    headline = float(pi.mean())
    recommendation = "Buy" if headline >= 60 else ("Sell" if headline <= 40 else "Hold")

    conflict = np.clip(np.asarray(construct_ts.get("conflict", np.full(n, 0.5)), dtype=float), 0.0, 1.0)
    confidence = float(np.clip(0.92 - 0.45 * conflict.mean() - 0.3 * (pi.std() / 50.0), 0.40, 0.96))

    k = min(3, n)
    peak_idx = sorted(np.argsort(pi)[::-1][:k].tolist())
    peaks = [
        PeakMoment(t_seconds=float(seconds[i]), purchase_intent=round(float(pi[i]), 1),
                   label="Peak buy moment")
        for i in peak_idx
    ]

    constructs = []
    for c in CONSTRUCTS:
        ts = np.clip(np.asarray(construct_ts.get(c.key, np.zeros(n)), dtype=float), 0.0, 1.0)
        constructs.append(
            ConstructScore(key=c.key, label=c.label, score=round(float(ts.mean()) * 100.0, 1),
                           networks=list(c.networks), description=c.description)
        )

    def pct(key):
        return [round(float(x) * 100.0, 1) for x in np.clip(np.asarray(construct_ts.get(key, np.zeros(n))), 0, 1)]

    network_timeline = {
        net: [round(float(x) * 100.0, 1) for x in np.clip(np.asarray(ts, dtype=float), 0.0, 1.0)]
        for net, ts in network_ts.items()
    }
    timeline = SignalTimeline(
        seconds=[float(s) for s in seconds],
        purchase_intent=[round(float(x), 1) for x in pi],
        attention=pct("attention"),
        emotion=pct("emotion"),
        network_timeline=network_timeline,
    )

    return Signals(
        purchase_intent=round(headline, 1),
        recommendation=recommendation,
        confidence=round(confidence, 2),
        constructs=constructs,
        timeline=timeline,
        peak_moments=peaks,
    )


def _network_timeseries(parcel_ts: np.ndarray, network_ids: np.ndarray, n_networks: int) -> np.ndarray:
    """(T, n_parcels) + (n_parcels,) ids -> (T, n_networks) mean activation per network."""
    T = parcel_ts.shape[0]
    out = np.full((T, n_networks), 0.5, dtype=float)
    for k in range(n_networks):
        sel = network_ids == k
        if np.any(sel):
            out[:, k] = parcel_ts[:, sel].mean(axis=1)
    return out


def _minmax_global(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def construct_ts_from_parcels(parcel_ts: np.ndarray, network_ids: np.ndarray) -> dict[str, np.ndarray]:
    """REAL readout: predicted parcel activation over time -> per-construct timeseries in [0,1].

    Aggregates Schaefer-1000 parcels into Yeo-7 networks, normalizes their relative
    activation, then maps networks -> constructs per CONSTRUCTS (the paper-grounded table).
    This is the region-weighted buy/sell readout RealEngine uses; the mock approximates it.
    """
    from .postprocess import YEO7_NETWORKS

    parcel_ts = np.asarray(parcel_ts, dtype=float)
    network_ids = np.asarray(network_ids, dtype=int)
    net = _minmax_global(_network_timeseries(parcel_ts, network_ids, len(YEO7_NETWORKS)))
    name_to_idx = {name: i for i, name in enumerate(YEO7_NETWORKS)}

    construct_ts: dict[str, np.ndarray] = {}
    for c in CONSTRUCTS:
        idxs = [name_to_idx[n] for n in c.networks if n in name_to_idx]
        construct_ts[c.key] = net[:, idxs].mean(axis=1) if idxs else np.full(parcel_ts.shape[0], 0.5)
    return construct_ts
