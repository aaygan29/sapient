# Adversarial review — six lenses

Each expert tries to break the work and the product. ✅ = fixed in code this pass, ◑ = mitigated
(honest caveat / reframed), ⚠ = structural/open (needs real data or strategy, cannot be coded away).
The point of listing ⚠ items is that hiding them is the actual risk.

## 🧮 Mathematician
1. **✅ Bootstrap CI mislabeled.** The interval only perturbed the reference weights → it was a
   mapping-sensitivity band sold as forecast uncertainty. Now perturbs **both** the affinity
   weights *and* the input activations (`input_sigma`), and the docstring states plainly it is a
   sensitivity+input band, **not** a subject-level CI (n=1).
2. **✅ Significance ≠ magnitude.** A band could exclude 50 by a hair and read as "PASS". Added a
   **minimum-effect floor** to `as_finding` (default 0.1); tiny-but-significant now abstains.
   (In-silico pass rate dropped 8/8 → 7/8 — the floor biting, as intended.)
3. **✅ Power overstated.** Fisher-z on Pearson understated the n for the **Spearman** statistic
   the harness actually uses. Added the Spearman SE inflation `√(1+ρ²/2)`; aggregate n 30 → 33,
   MDES solved by bisection. Also labeled the normal-approx assumption.
4. **✅ Tie handling.** Percentile ranking used `argsort∘argsort` (ties broken arbitrarily). Now
   average-rank (`scipy.stats.rankdata`).
5. **◑ Clipping bias.** `clip(0.5 + approach − avoid)` saturates and biases extremes; `gated_approach`
   and `generalizability` are still partly ad-hoc functional forms. Documented; the honest read-out
   is rank/relative, not the absolute composite.

## 💵 Economist
1. **⚠ Survivorship / range restriction.** The retrospective ad set is all famous winners — no
   flops. A correlation on winners-only is uninterpretable. **Fix requires a matched sample of
   failed campaigns**; until then the retrospective pilot is a machinery demo, now labeled as such.
2. **◑ Incommensurable outcomes.** Vote swing, brand lift, and sales lift were normalized onto one
   0–1 axis — not comparable. The pooled ρ is therefore not a real effect size; the pilot no longer
   leans on it and the real test (`validation/`) uses a single outcome metric per study.
3. **⚠ Funding ≠ ROI.** Genevsky-Knutson prove neural forecasts *aggregate choice/funding*. A
   marketer buys *incremental revenue*. That last causal link (score → ROI, net of spend, brand,
   distribution) is unestablished and endogenous. Named as the key open economic claim.
4. **⚠ Edge competes away.** If neural pre-testing works and diffuses, the alpha erodes; the durable
   value is the calibrated instrument + data flywheel, not the score itself.

## 🧠 Behavioral scientist
1. **✅ Rigged demo.** The validation demo gave the neural indicator less noise than self-report —
   assuming the conclusion. Now **equal signal and noise**; the verdict distinguishes "adds
   incremental validity" from "outpredicts," and on equal inputs correctly reports *comparable, not
   better*.
2. **⚠ Self-authored descriptors.** The retrospective pilot's "neural" inputs are hand-authored by
   the analyst → circular. **Fix: independent blinded raters or the actual encoder**, not one
   person's structural guesses. Flagged as the pilot's core weakness.
3. **◑ Construct/discriminant validity.** "Approach" = reward+emotion+attention+memory may just track
   arousal / low-level salience. The specificity gate + confound residualization exist upstream; a
   dedicated discriminant test (approach vs a pure luminance/motion proxy) is the next `validation/`
   addition.
4. **⚠ Lab ≠ field.** Single-exposure attitude shift ≠ repeated real-world exposure with wear-out.

## 🔬 Neuroscientist
1. **⚠ Predicted, not measured, BOLD.** The whole chain rests on the Mary/TRIBE encoder's predicted
   activation, never validated end-to-end (the repo's own Link-4 gap). Nothing here changes that.
2. **✅→◑ Subcortical blindness.** The core reward (NAcc) and aversion (anterior insula) nuclei are
   subcortical and **absent from the Schaefer/Yeo cortical atlas**; the composite reads cortical
   shadows. Now emitted as a first-class **CORTICAL PROXY** note on every forecast. Real fix is a
   subcortical-inclusive atlas/contrast (documented in the data plan).
3. **◑ Reverse inference.** Network activation → "reward"/"emotion" is reverse inference. Mitigated
   by the specificity gate; still a caveat.
4. **◑ Crude projection.** Empirical reference = positive mean T of a task contrast within resting
   parcels — a coarse mixing of task and rest geometry. Labeled; a proper subject-level GLM is the
   upgrade.

## 💰 Investor
1. **⚠ Moat.** Knutson/Genevsky science, the Schaefer atlas, and TRIBE-class encoders are public.
   Defensibility must come from the **proprietary validation dataset + calibration + workflow**, not
   the equations. Stated explicitly rather than hand-waved.
2. **◑ No real outcome evidence yet.** n=7 descriptors + simulated demo is not traction. The honest
   panel says so; the powered `validation/` study (n≈33 at the aggregate effect) is the cheapest path
   to the first real datapoint.
3. **⚠ Regulatory/brand risk.** "Manipulation"/neuro-marketing scoring invites privacy and ethics
   scrutiny. The engine scores *content*, not people, and collects no neural data from viewers (see
   `docs/PRIVACY_POLICY.md`) — that framing is the mitigation and must be held to.
4. **⚠ Feature vs company.** TAM/GTM unaddressed here — out of scope for the codebase, named so it
   isn't mistaken for solved.

## 📣 Marketing specialist
1. **⚠ Actionability.** A creative director needs timestamped, specific guidance, not a single
   buy/sell number. `neurosignal/timeline.py` already yields per-second arcs and peak moments — the
   product gap is surfacing "cut/keep at t=…", not new science. Named as the top utility fix.
2. **⚠ Benchmark database.** "73rd percentile for Gen Z" needs a category corpus that does not exist
   yet. The percentile here is within an 8-ad toy set — honest but not yet a product benchmark.
3. **◑ Why neural over cheaper tools.** Must beat copy-testing / attention vendors (Realeyes,
   TVision). The defensible claim is **incremental validity over self-report** — which is exactly
   what `validation/` tests, not asserts.
4. **⚠ Segment scores.** Buyers want per-segment ("digital twin") scores; the science supports
   **aggregate** forecasting, and per-segment is a stub. Do not ship segment numbers as validated.
5. **◑ False precision.** A single 0–100 number invites over-trust. Every score now carries a band
   and abstains when the effect is too small or straddles chance.

## Net
The **method** holes were mostly real and are fixed (CI honesty, effect floor, Spearman power, ties,
rigged demo, subcortical flag). The **product/economic** holes are structural and require the real
validation study, a failed-campaign sample, a benchmark corpus, and actionable timestamped output —
none of which can be honestly faked in code. The single highest-leverage next step remains the
powered `validation/` run: it converts the biggest ⚠ (no real outcome evidence) into a datapoint.
