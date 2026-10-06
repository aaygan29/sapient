"""Qualia demo UI (reference) — "Mary perceives the world through a human brain."

Reads results.json (built by build_demo_data.py from Mary's validated Paper-A results):
real encoding r vs real fMRI per Yeo-7 network + the per-person-beats-average result.
This is a reference renderer; the frontend agent can mirror it from results.json + assets/.

Run:  streamlit run app.py     (deps: streamlit, matplotlib)
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import streamlit as st

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results.json")


def main():
    st.set_page_config(page_title="Qualia — perceive through a human brain", layout="wide")
    if not os.path.exists(RESULTS):
        st.error("results.json missing — run: python build_demo_data.py")
        return
    d = json.load(open(RESULTS))

    st.title("🧠 " + d["title"])
    st.subheader(d["claim"])
    hp = d.get("headline_personalization") or {}
    if hp:
        st.success(f"### Per-person models beat the average brain **{hp['ratio']}×** "
                   f"(r={hp['per_subject_mean_r']} vs {hp['average_mean_r']}). "
                   f"That per-person signal is the data moat.")

    names = list(d["stimuli"])
    pick = st.sidebar.radio("Held-out stimulus (Mary never trained on it)", names,
                            format_func=lambda n: d["stimuli"][n].get("label", n))
    s = d["stimuli"][pick]
    st.caption(f"{s.get('label','')} — {s.get('dataset','')}, {s.get('n_subjects','?')} subjects, "
               f"{s.get('modality','')}. Genuine vertex signal (broadcast artifact: "
               f"{s.get('is_broadcast_artifact')}).")

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("**What a human brain does (Mary's prediction)**")
        bs = os.path.join(HERE, s["figures"]["brain_surface"])
        if os.path.exists(bs):
            st.image(bs, use_container_width=True)
    with c2:
        st.markdown("**Encoding accuracy per network — Mary vs noise ceiling**")
        nets = [n["network"] for n in s["per_network"]]
        mary = [n["mary_r"] for n in s["per_network"]]
        ceil = [n["noise_ceiling"] for n in s["per_network"]]
        social = {n["network"] for n in s["per_network"] if n["is_social"]}
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(nets))
        w = 0.4
        ax.bar([i - w / 2 for i in x], mary, w, label="Mary (real r)", color="#7b2ff7")
        ax.bar([i + w / 2 for i in x], ceil, w, label="noise ceiling", color="#2ec4b6", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(nets, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("encoding r")
        ax.legend(fontsize=9)
        for n in social:
            ax.get_xticklabels()[nets.index(n)].set_color("#7b2ff7")
        st.pyplot(fig)
        st.caption("Purple labels = social/affect networks. Bars at/above ceiling = Mary matches "
                   "or exceeds inter-subject reliability.")

    st.caption("⚠️ " + d["honesty"])
    st.caption("Provenance: " + d["provenance"])


if __name__ == "__main__":
    main()
