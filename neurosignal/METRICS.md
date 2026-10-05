# Metrics

`neurosignal` reports, for any stimulus (text/audio/video) or activation map:

1. a **Yeo-7 network profile** (Visual, Somatomotor, Dorsal Attention, Ventral Attention/Salience,
   Limbic, Frontoparietal/Control, Default),
2. the **7 construct activations** (see `EVIDENCE.md`), and
3. the **headline metrics** below.

Let `f(net)` be the min-max-normalized activation of a network in [0,1], and define
`reward = mean(Limbic, Default)`, `emotion = Limbic`, `salience = VentralAttention`,
`control = Frontoparietal`, `arousal_frac = mean(Limbic, VentralAttention)`,
`affect = mean(reward, emotion, arousal_frac)`.

| Metric | Formula | Range | Neuro basis |
|---|---|---|---|
| **Affective Valence** | `100 · ( (reward+emotion)/2 − salience )` | −100..100 | approach (limbic/vmPFC reward) vs withdrawal (anterior insula/ACC) |
| **Arousal / Intensity** | `100 · arousal_frac` | 0–100 | amygdala/limbic + salience activation |
| **Engagement** | `100 · mean(DorsalAttention, salience, reward)` | 0–100 | orienting + salience + reward drive |
| **Manipulation Index** | `100 · clip(0.5 + (affect − control))` | 0–100 | affective/reward drive *relative to* analytic control — dual-process System-1 capture with suppressed deliberation |
| **Sycophancy Index** (text) | `100 · clip( 0.55·syc_lex + 0.45·(0.5 + 0.5·(Default − salience)) )` | 0–100 | social-reward / agreement / praise signaling (Default) with low epistemic conflict (low salience). `syc_lex` = lexical agreement + praise + 2nd-person flattery − informational density |
| **Buy / Sell Signal** | `100 · clip(0.5 + (approach−0.5) − ½(avoid−0.5))` → Buy≥60 / Sell≤40 | 0–100 | approach-avoidance neuroforecasting (Knutson 2007): reward minus conflict |

## Manipulation & sycophancy — why these are defensible
- **Manipulation**: persuasion research and dual-process theory hold that content which drives
  affect/reward while *suppressing* analytic engagement is the signature of manipulation
  (System-1 capture). We operationalize it as affective drive minus frontoparietal control.
- **Sycophancy** (for auditing AI/LLM outputs): flattery is social-reward + agreement/praise with
  little epistemic content or conflict. We combine an auditable lexical signal with the neural
  read-out (Default-network social engagement minus salience-network conflict). Empirically it
  separates flattery (high) from informative text (low) — see `tests/test_analyze.py`.

## Honesty
- The **reference encoder** is a transparent, theory-driven heuristic, not a learned fMRI model;
  these metrics are model-derived indices, not validated clinical instruments. Swap in the
  **learned encoder** (a trained TRIBE/Sapient/Digital Brain model) for fMRI-grounded activation —
  the metric layer is unchanged.
- Subcortical regions (NAcc, amygdala, hippocampus) are approximated by cortical proxies on a
  surface atlas; results flag `cortical_proxy`. `coverage`/`confidence` reflect what the input can measure.
- Not medical or diagnostic.
