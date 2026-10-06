"""Build the demo's frontend-ready results.json from Mary's VALIDATED Paper A results.

Uses the integrity-checked numbers in Mary-Papers/paper-A-mary-orcle/experiments/results.json
(per-subject ensemble, OOD held-out stimuli) — real encoding r vs real fMRI per Yeo-7 network,
with the inter-subject noise ceiling and the per-subject-vs-average (personalization) gap.

Output: qualia/demo/results.json + qualia/demo/assets/*.png (copied figures).
This is the honest "Mary perceives the world through a human brain — measured" demo, and the
data is upgrade-coupled in spirit: regenerate from a stronger Mary's results to improve it.

Usage:  python build_demo_data.py
"""
from __future__ import annotations

import json
import os
import shutil

SAP = "/Users/robertgutierrez/Desktop/sapient-research"
PAPER_A = os.path.join(SAP, "Mary-Papers", "paper-A-mary-orcle")
RESULTS_SRC = os.path.join(PAPER_A, "experiments", "results.json")
FIG = os.path.join(PAPER_A, "figures")

HERE = os.path.dirname(__file__)
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "results.json")

SOCIAL = {"VentralAttention", "Limbic", "Frontoparietal", "Default"}
NET_ORDER = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
             "Limbic", "Frontoparietal", "Default"]
# Human-readable stimulus metadata (OOD held-out — Mary never trained on these).
STIM_META = {
    "forgot": {"label": "Spoken narrative ('forgot' story)", "dataset": "Huth Narratives",
               "modality": "audio / language", "n_subjects": 22},
    "bourne": {"label": "Movie (CNeuroMod 'Bourne')", "dataset": "CNeuroMod",
               "modality": "audiovisual movie", "n_subjects": 4},
}


def build_stimulus(name: str, ood: dict) -> dict:
    ens = ood["ensemble"]
    yeo, ceil = ens["yeo7_per_subject"], ens["yeo7_noise_ceiling"]
    per_network = [{
        "network": n,
        "mary_r": round(yeo[n]["r"], 4),
        "noise_ceiling": round(ceil[n]["r"], 4),
        "pct_of_ceiling": round(yeo[n]["pct_of_ceiling"], 1),
        "n_parcels": yeo[n]["n_parcels"],
        "is_social": n in SOCIAL,
    } for n in NET_ORDER if n in yeo]
    ps = ens["conditions"]["per_subject"]
    av = ens["conditions"]["average"]
    return {
        "name": name,
        **STIM_META.get(name, {}),
        "noise_ceiling_overall": round(ens["noise_ceiling"], 4),
        "per_network": per_network,
        "personalization": {
            "per_subject_mean_r": round(ps["mean_r"], 4),
            "average_mean_r": round(av["mean_r"], 4),
            "ratio": round(ens["per_subject_over_average_ratio"], 2),
            "beats_average": ens["per_subject_beats_average"],
        },
        "is_broadcast_artifact": ens["structure_check"]["is_broadcast"],
        "figures": {
            "network_bars": f"assets/{name}_yeo7.png",
            "brain_surface": f"assets/{name}_brain_surface.png",
        },
    }


def main():
    os.makedirs(ASSETS, exist_ok=True)
    src = json.load(open(RESULTS_SRC))
    stimuli = {k: build_stimulus(k, v) for k, v in src["ood_sets"].items()}

    for name in stimuli:
        for fig in (f"{name}_yeo7.png", f"{name}_brain_surface.png"):
            s = os.path.join(FIG, fig)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(ASSETS, fig))

    out = {
        "title": "Mary — perceiving the world through a human brain",
        "claim": ("Mary predicts the human brain's response to held-out stimuli it never trained "
                  "on, across all 7 cortical networks, and per-person models beat the average "
                  "brain ~3.7x."),
        "model": {"name": "Mary", "detail": src.get("model", ""),
                  "space": "fsaverage5", "vertices": 20484, "channel": "improving"},
        "metric": "Encoding Pearson r (Mary's predicted fMRI vs real fMRI), per Yeo-7 network, "
                  "with inter-subject noise ceiling.",
        "network_order": NET_ORDER,
        "social_networks": sorted(SOCIAL),
        "stimuli": stimuli,
        "headline_personalization": stimuli.get("forgot", {}).get("personalization"),
        "honesty": ("Representational alignment (real encoding r vs real fMRI), NOT phenomenal "
                    "feeling. Per-subject > average is the personalization signal that powers the "
                    "data moat. NeuroEngage (human-vs-robot) is the roadmap: needs Mary's video "
                    "streams + fMRI validation."),
        "provenance": "Mary-Papers/paper-A-mary-orcle (integrity-checked; every number traces to results.json).",
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")
    for name, s in stimuli.items():
        p = s["personalization"]
        print(f"  [{name}] per-subject r={p['per_subject_mean_r']} vs avg {p['average_mean_r']} "
              f"({p['ratio']}x), networks={len(s['per_network'])}, "
              f"figs={'ok' if os.path.exists(os.path.join(ASSETS, name+'_yeo7.png')) else 'missing'}")


if __name__ == "__main__":
    main()
