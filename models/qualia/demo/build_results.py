"""Pull Mary's NeuroEngage encode output from the volume -> the demo's results.json.

Reads /data/qualia_demo/neuroengage_runs.json (per-run Yeo-7 from encode_score.py),
maps each run to its human/robot group via participants.tsv, averages per network,
and writes results.json + copies the representative brain PNGs for app_neuroengage.py.

Usage:  python build_results.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from statistics import mean

HERE = os.path.dirname(__file__)
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "results.json")
PARTICIPANTS = os.path.join(ASSETS, "participants.tsv")
SOCIAL = ["VentralAttention", "Limbic", "Frontoparietal", "Default"]


def sh(*args):
    subprocess.run(args, check=True)


def pull():
    os.makedirs(ASSETS, exist_ok=True)
    # Mary's per-run output + brain PNGs (the whole qualia_demo dir).
    sh("modal", "volume", "get", "--force", "sapient-data", "/qualia_demo", ASSETS)
    if not os.path.exists(PARTICIPANTS):
        # public OpenNeuro mirror (no creds)
        sh("bash", "-lc",
           f"curl -s --max-time 30 "
           f"'https://s3.amazonaws.com/openneuro.org/ds004996/participants.tsv' "
           f"-o '{PARTICIPANTS}'")


def sub_of(run_key: str) -> str:
    return run_key.split("_", 1)[0]            # "sub-01_operator_run-01" -> "sub-01"


def main():
    pull()
    runs = json.load(open(os.path.join(ASSETS, "qualia_demo", "neuroengage_runs.json")))["runs"]
    group = {r["participant_id"]: r["group"]
             for r in csv.DictReader(open(PARTICIPANTS), delimiter="\t")}

    agg = {"human": defaultdict(list), "robot": defaultdict(list)}
    networks = None
    rep_png = {"human": None, "robot": None}
    for key, nets in runs.items():
        g = group.get(sub_of(key))
        if g not in ("human", "robot"):
            continue
        networks = list(nets.keys())
        for n, v in nets.items():
            agg[g][n].append(v)
        png_src = os.path.join(ASSETS, "qualia_demo", f"{key}_brain.png")
        if rep_png[g] is None and os.path.exists(png_src):
            dst = os.path.join(ASSETS, f"{g}_brain.png")
            shutil.copy(png_src, dst)
            rep_png[g] = os.path.relpath(dst, HERE)

    cond = {g: {n: mean(vals) for n, vals in agg[g].items()} for g in ("human", "robot")}
    gaps = {n: cond["human"].get(n, 0) - cond["robot"].get(n, 0) for n in (networks or [])}
    top_social = max(SOCIAL, key=lambda n: gaps.get(n, float("-inf"))) if networks else None

    out = {
        "title": "Mary — predicted social-brain response: human vs robot partner",
        "metric": "Mary-predicted response magnitude per Yeo-7 network (representational, not 'feeling')",
        "networks": networks,
        "social_networks": SOCIAL,
        "conditions": cond,
        "brain_png": rep_png,
        "gap_per_network": gaps,
        "headline_social_network": top_social,
        "n_runs": {g: len(next(iter(agg[g].values()), [])) for g in ("human", "robot")},
        "caveat": "Audio-only (operator speech); single fixed subject head; not yet "
                  "controlled for low-level acoustic differences. Video + validation are next.",
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")
    if networks:
        print(f"headline social gap: {top_social} = "
              f"{gaps.get(top_social, 0):+.3f} (human - robot)")
        for n in SOCIAL:
            print(f"  {n:16s} human={cond['human'].get(n,0):.3f}  robot={cond['robot'].get(n,0):.3f}")


if __name__ == "__main__":
    main()
