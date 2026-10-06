"""
demo.py — end-to-end run of the ag_push prediction engine on synthetic data.

Pipeline:
  Mary vertexwise prediction
    -> mary_readouts.GroundedReadouts   (construct-valid, confound-controlled, gated)
    -> behavioral_bridge.BehavioralBridge  (affect-weighted market forecast)
    -> neuroforecasting.Neuroforecaster    (validate against real outcomes, LOCO)
    -> brain_trajectory.BrainTrajectory     (temporal engagement arc)
    -> persona_map.PersonaMap                (per-subject phenotype + few-shot)

No external data needed; synthetic maps stand in for Mary output so the wiring runs.
Replace `simulate_*` with real Mary exports + real ad outcomes to validate.
"""
import numpy as np

from mary_readouts import GroundedReadouts, CONSTRUCTS
from behavioral_bridge import BehavioralBridge, forecast_campaign_set
from neuroforecasting import Neuroforecaster
from brain_trajectory import BrainTrajectory
from persona_map import PersonaMap

N_VERT = 20484
rng = np.random.default_rng(7)


def simulate_construct_maps():
    # each construct weights a distinct vertex block; `value` lives on vertices :200
    # so the value read-out will track the latent appeal baked there by simulate_ads.
    maps = {c: np.abs(rng.standard_normal(N_VERT)) * 0.05 for c in CONSTRUCTS}
    blocks = {c: slice(i * 200, (i + 1) * 200) for i, c in enumerate(CONSTRUCTS)}
    for c, sl in blocks.items():
        maps[c][sl] += 1.0
    return maps


def simulate_ads(n=40):
    vmaps = rng.standard_normal((n, N_VERT)) * 0.3
    # latent "appeal" drives the value block (vertices :200) and the market outcome
    appeal = rng.standard_normal(n)
    vmaps[:, :200] += appeal[:, None] * 1.2
    outcomes = np.clip(55 + 12 * appeal + rng.standard_normal(n) * 4, 0, 100)
    self_report = np.clip(55 + 6 * appeal + rng.standard_normal(n) * 8, 0, 100)
    return vmaps, outcomes, self_report


def main():
    print("=" * 68)
    print("ag_push prediction engine — end-to-end demo (synthetic)")
    print("=" * 68)

    cmaps = simulate_construct_maps()
    vmaps, outcomes, self_report = simulate_ads(40)

    # construct read-outs use in-memory maps (skip disk load for the demo)
    ro = GroundedReadouts.__new__(GroundedReadouts)
    ro.construct_maps = cmaps
    ro.confound_maps = {}
    readouts = [ro.score(v) for v in vmaps]

    # 1) brain -> behavior forecast (literature-weighted)
    bridge = BehavioralBridge()
    value_raw = np.array([sum(bridge.cw.get(k, 0) * s for k, s in r.items()
                              if not k.startswith("_")) for r in readouts])
    cal = bridge.calibrate(value_raw, outcomes)
    forecasts = forecast_campaign_set(bridge, readouts, self_reports=list(self_report))

    print(f"\n[BehavioralBridge] calibrated on 40 ads: "
          f"in-sample r={cal['in_sample_r']:.2f}, scale={cal['scale']:.2f}")
    order = np.argsort([-f.index for f in forecasts])
    print("  top 3 forecast ads:")
    for rank, i in enumerate(order[:3], 1):
        print(f"   {rank}. ad{i:02d}  {forecasts[i]}")

    # 2) validate forecast against outcomes (neuroforecasting LOCO)
    nf = Neuroforecaster(construct="value", outcome_metric="ad_outcome", min_effect=0.30)
    res = nf.predict_outcomes(value_raw, outcomes, self_report=self_report, n_perms=2000)
    print(f"\n[Neuroforecaster] brain r={res.r_brain:.2f} (p={res.p_value:.3f}) "
          f"vs self-report r={res.r_self_report:.2f} -> "
          f"{'PASS' if res.passes_specificity else 'FAIL'}")

    # 3) temporal engagement arc for one ad
    n_t = 60
    t = np.linspace(0, 30, n_t)
    vt = rng.standard_normal((n_t, N_VERT)) * 0.4
    vt += 0.6 * np.exp(-((t - 15) ** 2) / 40)[:, None]     # peak at :15
    traj = BrainTrajectory(vt, t, construct="arousal")
    hm = traj.timeline_heatmap(cmaps["arousal"])
    print(f"\n[BrainTrajectory] {hm['summary']}")
    for rec in traj.creative_recommendations(cmaps["arousal"])[:2]:
        print(f"   {rec}")

    # 4) per-subject phenotype + few-shot
    persona = PersonaMap.from_subject_fmri(vmaps[:10], cmaps,
                                           population_stats={"means": np.zeros(len(CONSTRUCTS)),
                                                             "stds": np.ones(len(CONSTRUCTS))})
    persona.subject_id = "subj_demo"
    card = persona.phenotype_card()
    print(f"\n[PersonaMap] {card}")
    print(f"   top constructs: {', '.join(card.responsive_constructs)}")

    print("\n" + "=" * 68)
    print("Engine wired end-to-end. Swap synthetic -> real Mary exports + ad outcomes.")
    print("=" * 68)


if __name__ == "__main__":
    main()
