"""
cultural_calibration.py — cross-cultural calibration HYPOTHESIS layer for
buy_moment_detector.py's ROI-group scores.

THIS IS THE LEAST-EVIDENCED MODULE IN THIS DIRECTORY. Read the whole
docstring before using it for anything.

Motivation
----------
`ACCEPTANCE_CRITERIA.md`'s "Generalizable vs idiosyncratic channels" section
(itself built on Genevsky & Knutson 2018/2025) is about within-country
demographic representativeness: their own quartile analysis shows forecast
accuracy degrades when the training sample doesn't match the target
population's demographics WITHIN one country/culture. This module extends
that concern to a DIFFERENT and BIGGER claim — cross-cultural/cross-market
generalization — which needs its own literature, not just an analogy to the
demographic-quartile result. That literature is thinner, coarser, and
covers a different set of regions than the neuroeconomics citations in
`buy_moment_detector.ROI_GROUPS`. Nothing below should be read as having the
same evidentiary weight as the Genevsky & Knutson ICC numbers.

What was verified this session (WebSearch/WebFetch, real citations only)
--------------------------------------------------------------------------
1. Han, S. & Northoff, G. (2008), "Culture-sensitive neural substrates of
   human cognition: a transcultural neuroimaging approach," Nature Reviews
   Neuroscience 9:646-654. Foundational review: self-referential processing
   engages MPFC differently in individualistic (Western) vs. collectivistic
   (East Asian) samples; East Asian samples often co-recruit regions
   associated with processing significant others (more medial/dorsal MPFC
   and social-cognition territory) where Western samples show more isolated
   ventral MPFC engagement. Confirmed via WebSearch (title, journal,
   volume/pages, and the self-referential-processing claim cross-checked
   across the Nature Reviews Neuroscience listing and independent secondary
   summaries) — this is a review article, not itself a meta-analysis with
   effect sizes, so no quantitative number is taken from it here.

2. Han, S. & Ma, Y. (2014), "Cultural differences in human brain activity:
   A quantitative meta-analysis," NeuroImage 99:293-300. A real
   coordinate-based meta-analysis of 35 fMRI studies (published before Dec
   2013) directly comparing East Asian vs. Western samples. Confirmed via
   WebSearch (title, journal, volume/pages, and the region-level findings
   cross-checked across the PubMed listing, ScienceDirect abstract listing,
   and an independent secondary summary — the ScienceDirect/Ovid full text
   itself sat behind a paywall this session, so this is confirmed at the
   verified-abstract/region-list level, not by reading the full
   coordinate tables). Reported contrasts:
     - East Asian > Western: dorsomedial PFC (dMPFC), left inferior frontal
       cortex, right inferior parietal cortex, right temporoparietal
       junction (TPJ).
     - Western > East Asian: anterior cingulate cortex (ACC), ventromedial
       PFC (vMPFC), bilateral insula, right superior frontal cortex, left
       precentral gyrus, right claustrum.
   No specific p-values, Cohen's-d-style effect sizes, or per-region
   activation-likelihood-estimate (ALE) statistic values were extracted this
   session (blocked by paywall) — only the qualitative direction and region
   list, which is what the calibration below is built on. This is a REAL,
   verified, quantitative (coordinate-based ALE) meta-analysis; it is not a
   single self-report survey, but the specific effect magnitudes are not in
   hand here.

3. Reward-circuitry-specific search (NAcc / OFC / ventral striatum,
   cross-cultural, meta-analytic): WebSearch turned up individual studies on
   reward learning and dopaminergic circuitry showing culture-modulated
   sensitivity (e.g., Eastern-sample ventral-striatal/VMPFC/hippocampal
   connectivity differences in learning-reward tasks), but **no
   coordinate-based meta-analysis specifically isolating NAcc/OFC/ventral-
   striatal anticipatory-affect activity across cultures** was found or
   verified this session — unlike the self-referential/social-cognition
   literature above, which does have a real meta-analysis (Han & Ma 2014).
   This is a genuine, disclosed literature gap, not an oversight: the
   NAcc/OFC "generalizable" channel in `buy_moment_detector.ROI_GROUPS`
   (Genevsky & Knutson's finding) and the cross-cultural self-referential
   literature (Han & Northoff / Han & Ma) come from two different research
   traditions that have not, as far as this session's search found, been
   directly combined into one cross-cultural NAcc/OFC study.

4. US-vs-China (or broader East-Asian-vs-Western) consumer neuroscience /
   neuromarketing fMRI comparison: WebSearch found (a) a 2011 US-China
   *industry* neuromarketing partnership (Sands Research + Brain
   Intelligence Neuro-Consultancy) with no published comparative fMRI
   findings surfaced, (b) an fNIRS study on cross-culture marketing
   strategies for transnational brands (different modality, different
   research question), and (c) explicit statements in a 2024 neuromarketing
   bibliometric review that cross-cultural consumer-neuroscience research is
   an **under-researched gap**. No verified peer-reviewed US-vs-China (or
   equivalent) consumer-neuroscience fMRI comparison study was found. This
   is disclosed as a real, named gap below and in `ACCEPTANCE_CRITERIA.md`,
   not papered over with an assumed number.

5. UPDATE (2026-08-25 session): re-opened this module specifically to look
   for an extractable, quantitative (not just directional) effect size to
   replace the arbitrary +/-15% below, per direct instruction. Re-read the
   Han & Ma (2014) full text this session (WebFetch of the author-hosted PDF
   mirror at mylab.bnu.edu.cn, cross-checked against the PubMed/ScienceDirect
   listing) — the paywall block noted in point 2 above is now resolved. The
   full text confirms point 2's direction summary, but Table 2/3/4 report
   ONLY Activation Likelihood Estimation (ALE) cluster statistics — weighted-
   center MNI coordinates and cluster volume in mm^3 — not Cohen's d,
   Hedges' g, t-values, or any other standardized effect size. This is a
   structural property of coordinate-based ALE meta-analysis (it answers
   "where do foci converge," not "how much bigger is the effect"), not an
   omission specific to this paper. CONCLUSION: Han & Ma (2014) itself still
   cannot license a quantitative multiplier. No 2023-2026 meta-analysis with
   an extractable effect size for this specific vMPFC/dMPFC-TPJ contrast was
   found either (WebSearch turned up adjacent 2024-2025 work — inhibitory
   control ALE meta-analyses, self-referential-encoding meta-analyses in
   education/development contexts — none isolating this contrast with a
   reportable effect size).

   Per the task's fallback instruction, we then searched for a single
   well-powered PRIMARY study reporting raw group-comparison statistics
   instead of a meta-analysis. We found one: **Ma, Y., Bang, D., Wang, C.,
   Allen, M., Frith, C., Roepstorff, A., & Han, S. (2014), "Sociocultural
   patterning of neural activity during self-reflection," Social Cognitive
   and Affective Neuroscience 9(1):73-80** (PMC3871729; confirmed via
   WebFetch of the PMC full text this session) — 30 Chinese and 30 Danish
   participants (matched-ish N, real between-subjects design) judged
   physical/mental/social attributes of self vs. a public figure. It reports
   ANOVA F-statistics, not Cohen's d directly, but F(1, df) with 1 numerator
   df converts to d by the standard, textbook formula for two independent
   groups (t = sqrt(F); d = t * sqrt(1/n1 + 1/n2); Rosenthal 1994; Cohen
   1988):
     - mPFC, self-reflection, Danish > Chinese: F(1,58) = 17.011, P < 0.001
       -> t = sqrt(17.011) = 4.12 -> d = 4.12 * sqrt(1/30 + 1/30) = 4.12 *
       0.2582 ~= **1.07** (large, direction matches Han & Ma's West > East
       Asian vMPFC/MPFC finding).
     - Left TPJ, social attributes, Chinese > Danish: F(1,58) = 9.308, P =
       0.003 -> t = 3.05 -> d ~= **0.79**.
     - Right TPJ, social attributes, Chinese > Danish: F(1,58) = 8.979, P =
       0.004 -> t = 2.996 -> d ~= **0.77**.
   These are real, computable, direction-confirming, LARGE effect sizes from
   an actual 60-subject primary study — stronger evidence than the ALE
   coordinate tables, and independent confirmation the qualitative Han & Ma
   direction is not a fluke of pooling 35 heterogeneous studies.

   HONEST LIMIT on what this buys us: these d-values are standardized mean
   differences on the study's own SPM GLM parameter-estimate scale (an
   arbitrary-origin, arbitrary-unit beta weight from that specific task and
   scanner), not a percent-signal-change or any other unit this pipeline's
   Schaefer-parcel activation values share. Converting a Cohen's d computed
   on one lab's arbitrary-unit contrast estimates into a specific "+/-X%"
   multiplier on THIS pipeline's group-mean parcel-activation series would
   require an additional, unstated assumption about how those two scales
   relate — there is no shared unit to convert through, and Cohen's d
   deliberately discards scale information to make it comparable ACROSS
   studies that don't share units, which is exactly the property that
   prevents reusing it as a same-unit multiplier here. Inventing that bridge
   assumption is the kind of unlicensed conversion this module's discipline
   explicitly prohibits (see `ACCEPTANCE_CRITERIA.md`), so it is not done.

   RESULT: `_MULTIPLIER_STRENGTH` remains the disclosed, arbitrary +/-15%
   placeholder — this session did not find a legitimate path to replace it
   with a literature-fitted number. What DID change: the direction (vMPFC
   stronger Western, dMPFC/TPJ stronger East Asian) now rests on both a
   35-study ALE meta-analysis AND an independent, real, 60-subject primary
   study with large (d ~= 0.77-1.07) effect sizes, which is materially more
   confidence in the DIRECTION than existed before this session — but zero
   additional evidence for the specific 15% MAGNITUDE, which is still
   unfit and still flagged as such everywhere it is surfaced.

What markets/axis this module actually supports
--------------------------------------------------
Given (1)-(4), the ONLY axis with real (if qualitative, not effect-sized)
neuroimaging support is the broad **individualism/collectivism axis** used
in the Han & Northoff / Han & Ma cultural-neuroscience tradition itself —
typically operationalized as "Western" (US/European samples) vs. "East
Asian" (Chinese/Japanese/Korean samples pooled) in the underlying 35
studies. It is NOT a validated "US" vs. "China" distinction specifically —
the underlying literature pools East Asian nationalities, and no study
found here isolates China from Japan/Korea, nor the US from Western Europe.
`target_market` below is therefore named and scoped to what the literature
actually distinguishes, not to the two specific countries a business
audience might expect.

What this module does
-----------------------
`apply_cultural_calibration(detect_result, target_market)` takes the dict
returned by `buy_moment_detector.detect_buy_moments` and applies a SMALL,
DIRECTIONAL, HYPOTHESIS-ONLY multiplier to exactly the two ROI-group means
that have real (if qualitative) cross-cultural support:
  - `Default_PFC_vmPFC` (vmPFC self-referential/value-integration proxy) —
    Han & Ma found Western > East Asian, so `target_market="east_asian"`
    down-weights this group's contribution and `target_market="western"`
    leaves it at baseline (or the reverse framing — see CALIBRATION_WEIGHTS
    for the exact convention used).
  - `Default_Temp_SocCog` (dMPFC/TPJ/precuneus social-cognition proxy) —
    Han & Ma found East Asian > Western, so `target_market="east_asian"`
    up-weights this group and `target_market="western"` leaves it at
    baseline.
All other ROI groups (`Limbic_OFC`, `Limbic_TempPole`, `Cont_PFCl_DLPFC`,
`Cont_Par_CogLoad`, `SalVentAttn_FrOperIns`, `SalVentAttn_ParOper`) are
passed through UNCHANGED, because no verified cross-cultural evidence
covers them specifically (see point 3 above for the OFC/NAcc case in
particular — this is the group most people would expect a "reward circuit"
calibration to touch, and it is deliberately left alone here).

The multiplier magnitudes (`_MULTIPLIER_STRENGTH` below) are a HYPOTHESIS,
not a fitted or validated parameter. Han & Ma 2014 gives us a direction
(which region is relatively stronger in which cultural grouping) but not an
effect size usable to derive a specific multiplier — no study measures "how
much should a buy-moment score change per culture." The magnitude chosen
here (+/-15%) is an arbitrary, clearly-flagged placeholder representing
"a modest, directionally-motivated nudge," not a number derived from any
paper. Treat `target_market != "unspecified"` output as a labeled
hypothesis to test against real data, never as a validated per-market
multiplier.

A 2026-08-25 session re-opened this module specifically to look for a real,
extractable effect size to replace the +/-15% placeholder (see docstring
point 5 below). It found strong, real, quantitative DIRECTIONAL evidence
(a large-d, 60-subject primary study, independent of Han & Ma) but no
legitimate way to convert that into a same-unit MAGNITUDE for this
pipeline's multiplier — so the number itself is still exactly as arbitrary
as before. Read point 5 before assuming "we found an effect size" means
"the 15% is now evidence-based" — it does not.

No behavioral validation of this calibration exists for ANY market. The
robustness check below (`_run_robustness_check`) only confirms the code
does not crash, produce NaNs, or flip signs pathologically when calibration
is toggled on real-shaped synthetic data — it validates the ARITHMETIC, not
the cultural claim. See `ACCEPTANCE_CRITERIA.md`'s new informational section
for the ship-relevant framing.

Run as a script (robustness self-test only — not a cultural validation)
--------------------------------------------------------------------------
    python cultural_calibration.py
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import numpy as np

from buy_moment_detector import ROI_GROUPS, _compose_score, _zscore, detect_buy_moments

# ---------------------------------------------------------------------------
# Allowed target markets — named after what the literature actually
# distinguishes (a broad individualism/collectivism cultural axis), NOT
# "US" / "China" specifically. See module docstring point 4.
# ---------------------------------------------------------------------------
TargetMarket = Literal["unspecified", "western", "east_asian"]

_MULTIPLIER_STRENGTH = 0.15  # +/-15%, an arbitrary hypothesis magnitude —
                             # NOT derived from any effect size in the cited
                             # literature. See docstring.

# Which ROI_GROUPS get touched, and in which direction, per target market.
# Only groups with real (if qualitative) cross-cultural literature support
# are listed. Everything else in ROI_GROUPS is left at multiplier 1.0.
#
# direction convention:
#   "west_higher"       -> Han & Ma found Western > East Asian for this
#                           region's homolog (vMPFC). Multiplier < 1.0 when
#                           target_market == "east_asian", 1.0 when "western"
#                           or "unspecified".
#   "east_asian_higher"  -> Han & Ma found East Asian > Western for this
#                           region's homolog (dMPFC/TPJ). Multiplier > 1.0
#                           when target_market == "east_asian", 1.0 when
#                           "western" or "unspecified".
CALIBRATION_TARGETS: dict[str, dict[str, Any]] = {
    "Default_PFC_vmPFC": {
        "direction": "west_higher",
        "source": "Han & Ma 2014 (NeuroImage 99:293-300): ventromedial PFC "
                   "activity stronger in Western vs. East Asian samples "
                   "across 35 meta-analyzed fMRI studies (ALE coordinates "
                   "only, no effect size). Independently corroborated by "
                   "Ma et al. 2014 (Soc Cogn Affect Neurosci 9(1):73-80, "
                   "PMC3871729), a 30 Chinese vs. 30 Danish primary study: "
                   "mPFC self-reflection activity greater in Danes, "
                   "F(1,58)=17.011, p<0.001, converting to Cohen's d ~= 1.07 "
                   "(t=sqrt(F), d=t*sqrt(1/n1+1/n2)). d is on that study's "
                   "own arbitrary-unit GLM parameter-estimate scale and is "
                   "NOT converted into this module's multiplier — see "
                   "module docstring point 5 for why that conversion is not "
                   "licensed by the data.",
        "confidence": "direction: strong (meta-analysis + large-d primary "
                       "study agree). magnitude: still no fitted number.",
    },
    "Default_Temp_SocCog": {
        "direction": "east_asian_higher",
        "source": "Han & Ma 2014 (NeuroImage 99:293-300): dorsomedial PFC "
                   "and right TPJ activity stronger in East Asian vs. "
                   "Western samples across 35 meta-analyzed fMRI studies "
                   "(ALE coordinates only, no effect size). This group also "
                   "includes posterior cingulate/precuneus parcels not "
                   "directly covered by the dMPFC/TPJ finding — the "
                   "calibration is applied to the whole group as a coarse "
                   "approximation, not parcel-by-parcel. Independently "
                   "corroborated by Ma et al. 2014 (Soc Cogn Affect "
                   "Neurosci 9(1):73-80, PMC3871729): TPJ activity during "
                   "social-attribute self-reflection greater in Chinese "
                   "than Danish participants, left TPJ F(1,58)=9.308 "
                   "p=0.003 (d ~= 0.79), right TPJ F(1,58)=8.979 p=0.004 "
                   "(d ~= 0.77) — same n=30/30 study as above; same caveat "
                   "that d is not converted into this module's multiplier.",
        "confidence": "direction: strong (meta-analysis + large-d primary "
                       "study agree). magnitude: still no fitted number.",
    },
}

# Explicitly, deliberately NOT calibrated — no verified cross-cultural
# literature found for these groups this session (see docstring point 3 for
# the OFC/NAcc case specifically).
UNCALIBRATED_GROUPS = [
    name for name in ROI_GROUPS if name not in CALIBRATION_TARGETS
]


def _group_multiplier(group_name: str, target_market: TargetMarket) -> float:
    if target_market == "unspecified":
        return 1.0
    spec = CALIBRATION_TARGETS.get(group_name)
    if spec is None:
        return 1.0
    direction = spec["direction"]
    if target_market == "east_asian":
        if direction == "west_higher":
            return 1.0 - _MULTIPLIER_STRENGTH
        if direction == "east_asian_higher":
            return 1.0 + _MULTIPLIER_STRENGTH
    if target_market == "western":
        # Western is the implicit baseline both underlying findings were
        # normed against in most of the pooled studies' framing; leave at
        # 1.0 rather than inventing a symmetric up-weight with no basis.
        return 1.0
    return 1.0


def apply_cultural_calibration(
    parcel_act: np.ndarray,
    target_market: TargetMarket = "unspecified",
    *,
    base_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply the cross-cultural calibration hypothesis to a buy-moment result.

    Parameters
    ----------
    parcel_act : (T, 400) Schaefer-400 parcel activations, same input
        `detect_buy_moments` takes.
    target_market : "unspecified" | "western" | "east_asian"
        See module docstring for why these are the only two named markets
        (not "US"/"China") and why "unspecified" is a no-op passthrough.
    base_result : optional pre-computed `detect_buy_moments(parcel_act)`
        result, to avoid recomputing it twice if the caller already has one.
        Must have been computed from the same `parcel_act`.

    Returns
    -------
    dict with:
        target_market       : echoed back
        calibrated_generalizable_score : (T,) z-scored, calibration applied
        calibrated_idiosyncratic_score : (T,) z-scored, calibration applied
        multipliers_applied : {group_name: multiplier} for every ROI group,
                               so the exact adjustment is auditable
        calibration_targets  : the literature-backed subset with sources
        uncalibrated_groups  : groups deliberately left untouched
        disclosure            : one-paragraph reminder this is a hypothesis,
                                 not a validated per-market adjustment
        baseline              : the unmodified `detect_buy_moments` result,
                                 for side-by-side comparison
    """
    baseline = base_result if base_result is not None else detect_buy_moments(parcel_act)

    from buy_moment_detector import _group_means as _gm  # local import to
    group_means = _gm(parcel_act)                        # avoid a circular
                                                            # top-of-file import

    multipliers = {name: _group_multiplier(name, target_market) for name in ROI_GROUPS}

    calibrated_means = {
        name: series * multipliers[name] for name, series in group_means.items()
    }

    generalizable_names = [n for n, s in ROI_GROUPS.items() if s["channel"] == "generalizable"]
    idiosyncratic_names = [n for n, s in ROI_GROUPS.items() if s["channel"] == "idiosyncratic"]

    cal_generalizable_raw = _compose_score(calibrated_means, generalizable_names)
    cal_idiosyncratic_raw = _compose_score(calibrated_means, idiosyncratic_names)

    return {
        "target_market": target_market,
        "calibrated_generalizable_score": _zscore(cal_generalizable_raw),
        "calibrated_idiosyncratic_score": _zscore(cal_idiosyncratic_raw),
        "multipliers_applied": multipliers,
        "calibration_targets": CALIBRATION_TARGETS,
        "uncalibrated_groups": UNCALIBRATED_GROUPS,
        "disclosure": (
            "This is a calibration HYPOTHESIS derived from directional "
            "region-level findings in Han & Ma (2014), the Han & Northoff "
            "(2008) review, and (as of a 2026-08-25 re-check) an "
            "independent 60-subject primary study (Ma et al. 2014, Soc "
            "Cogn Affect Neurosci 9(1):73-80) whose F-statistics convert to "
            "large Cohen's d (~0.77-1.07) confirming the SAME direction — "
            "still not a validated per-market multiplier, because that "
            "study's d is on its own arbitrary-unit GLM scale and cannot be "
            "converted into a same-unit multiplier for this pipeline "
            "without an unlicensed assumption (see module docstring point "
            "5). No study has measured how a buy-moment-style composite "
            "score should change across cultures; the "
            f"+/-{int(_MULTIPLIER_STRENGTH*100)}% magnitude is still an "
            "arbitrary placeholder representing a modest, directionally-"
            "motivated nudge, not a fitted or literature-derived effect "
            "size — this session strengthened confidence in the DIRECTION, "
            "not the MAGNITUDE. Only Default_PFC_vmPFC and "
            "Default_Temp_SocCog are touched; all other ROI groups "
            "(including the NAcc/OFC 'generalizable' reward-anticipation "
            "group most people would expect this to touch) are left "
            "unchanged because no verified cross-cultural meta-analysis of "
            "reward-circuitry activity was found this session. "
            "'western'/'east_asian' reflect the broad individualism/"
            "collectivism axis the underlying literature actually "
            "distinguishes — NOT a validated US-vs-China distinction. See "
            "ACCEPTANCE_CRITERIA.md's cultural-calibration section and "
            "cultural_calibration.py's module docstring for full sourcing "
            "and caveats."
        ),
        "baseline": baseline,
    }


# ---------------------------------------------------------------------------
# Robustness self-test: confirms the calibration arithmetic does not crash,
# NaN, or sign-flip pathologically. This validates the CODE, not the
# cultural claim — see module docstring.
# ---------------------------------------------------------------------------

def _synthetic_parcel_data(T: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(T, 400)).astype(np.float32)


def _run_robustness_check() -> bool:
    ok = True
    parcel_act = _synthetic_parcel_data()
    baseline = detect_buy_moments(parcel_act)

    for market in ("unspecified", "western", "east_asian"):
        result = apply_cultural_calibration(parcel_act, market, base_result=baseline)
        gen = result["calibrated_generalizable_score"]
        idio = result["calibrated_idiosyncratic_score"]

        finite = np.all(np.isfinite(gen)) and np.all(np.isfinite(idio))
        no_nan = not (np.isnan(gen).any() or np.isnan(idio).any())
        shape_ok = gen.shape == (parcel_act.shape[0],) and idio.shape == (parcel_act.shape[0],)

        # "unspecified" must be an exact no-op vs. detect_buy_moments' own
        # generalizable_score/idiosyncratic_score (multipliers all 1.0).
        noop_ok = True
        if market == "unspecified":
            noop_ok = (
                np.allclose(gen, baseline["generalizable_score"], atol=1e-5)
                and np.allclose(idio, baseline["idiosyncratic_score"], atol=1e-5)
            )

        passed = finite and no_nan and shape_ok and noop_ok
        ok = ok and passed
        print(f"target_market={market:12s} finite={finite} no_nan={no_nan} "
              f"shape_ok={shape_ok} noop_ok={noop_ok if market == 'unspecified' else 'n/a'}"
              f"  -> {'PASS' if passed else 'FAIL'}")

    print()
    print("Reminder: this only checks the calibration arithmetic doesn't")
    print("crash or corrupt the signal. It is NOT a validation of the")
    print("cultural-calibration hypothesis itself — no behavioral data for")
    print("any market backs this module. See module docstring.")
    return ok


if __name__ == "__main__":
    import sys
    passed = _run_robustness_check()
    sys.exit(0 if passed else 1)
