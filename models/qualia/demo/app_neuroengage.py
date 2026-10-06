"""Qualia demo UI — "Mary perceives the difference between a human and a robot."

Shows Mary's predicted brain response to hearing a HUMAN partner vs a ROBOT partner
(NeuroEngage operator audio), per Yeo-7 network, with the social networks highlighted.
Reads results.json produced by build_results.py. Upgrade-coupled: the numbers come
from Mary via the registry `improving` channel, so a better Mary improves this demo.

Run:  streamlit run app_neuroengage.py
Deps: streamlit, matplotlib
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import streamlit as st

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results.json")


def main():
    st.set_page_config(page_title="Qualia — human vs robot", layout="wide")
    if not os.path.exists(RESULTS):
        st.warning("No results.json yet. Run: `modal run qualia/demo/encode_score.py` "
                   "then `python qualia/demo/build_results.py`.")
        return
    d = json.load(open(RESULTS))
    nets = d["networks"]
    social = set(d.get("social_networks", []))
    cond = d["conditions"]

    st.title("🧠 Qualia — Mary perceives a human vs a robot")
    st.caption(d.get("metric", ""))

    # Brain maps: human vs robot
    c1, c2 = st.columns(2)
    for col, g, label in ((c1, "human", "Hearing a HUMAN partner"),
                          (c2, "robot", "Hearing a ROBOT partner")):
        with col:
            st.subheader(label)
            png = d.get("brain_png", {}).get(g)
            p = os.path.join(HERE, png) if png else None
            if p and os.path.exists(p):
                st.image(p, use_container_width=True)
            else:
                st.info("(brain map pending)")

    # Per-network bars: human vs robot
    st.subheader("Predicted response per brain network — human vs robot")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(nets))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [cond["human"].get(n, 0) for n in nets], w,
           label="human partner", color="#7b2ff7")
    ax.bar([i + w / 2 for i in x], [cond["robot"].get(n, 0) for n in nets], w,
           label="robot partner", color="#bbbbbb")
    ax.set_xticks(list(x))
    ax.set_xticklabels(nets, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mary-predicted response")
    ax.legend(fontsize=9)
    for n in social:
        if n in nets:
            ax.get_xticklabels()[nets.index(n)].set_color("#7b2ff7")
    st.pyplot(fig)

    bs = d.get("headline_social_network")
    if bs:
        gap = d.get("gap_per_network", {}).get(bs, 0)
        st.success(f"### 🟣 Mary's **{bs}** (social) network responds "
                   f"**{gap:+.3f}** more to a human than a robot partner.\n\n"
                   f"This is the human-likeness gap — measurable, in the social brain.")
    st.caption("⚠️ " + d.get("caveat", ""))
    st.caption(f"Engine: {json.dumps(d.get('n_runs', {}))} runs/condition. "
               "Representational/affective alignment — not a claim of machine feeling.")


if __name__ == "__main__":
    main()
