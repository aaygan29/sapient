# Evidence base

`neurosignal` maps brain-activation networks to consumer-relevant constructs and an
approach–avoidance buy/sell signal. This document is the literature behind every
construct, weight, and the composite. The code mirror is `neurosignal/constructs.py`.

## The neuroforecasting frame
The buy/sell composite follows **Knutson, Rick, Wimmer, Prelec & Loewenstein (2007),
*Neuron* 53:147** — "Neural predictors of purchases": **nucleus accumbens / mPFC** reward
activation predicts *buying*; **anterior insula** ("pain of paying") predicts *not buying*;
**mPFC** integrates value. We extend this with the supplied consumer-neuroscience papers.

## Constructs → regions → citations
| Construct | Polarity | Canonical regions | Yeo-7 proxy | Key citations |
|---|---|---|---|---|
| **Reward & Value** | +0.30 | vmPFC, OFC, ventral striatum/NAcc | Limbic, Default | Knutson 2007; Bartra et al. 2013 (value meta-analysis); Çakir et al. 2018 (fNIRS buy → fronto-polar/OFC/vmPFC); Kapoor et al. 2023 (ventral striatum, preferred choice) |
| **Emotional Resonance** | +0.22 | amygdala, vmPFC, limbic | Limbic | Vlăsceanu 2014 (somatic markers); Phan et al. 2002 (emotion meta-analysis) |
| **Attention Capture** | +0.18 | IPS, FEF, TPJ | DorsalAttention | Corbetta & Shulman 2002 |
| **Visual & Sensory Salience** | +0.10 | V1–V4, visual streams | Visual | Wandell et al. 2007 (retinotopy); Allen et al. 2022 (NSD) |
| **Memory Encoding** | +0.10 | hippocampus, parahippocampal, DMN | Default | Wagner et al. 1998 (encoding→recall); Raichle 2015 (DMN) |
| **Decision Conflict & Risk** | −0.22 | ACC, anterior insula | VentralAttention | Knutson 2007 (insula); Botvinick et al. 2001 (ACC conflict); Kapoor 2023 (rostral/dorsal ACC, regret); Shang 2018 (N2/N400) |
| **Cognitive Load / Deliberation** | −0.15 | DLPFC, dmPFC | Frontoparietal | Kapoor 2023 (R-DLPFC); Vlăsceanu 2014 (System-2) |

## Composite
Per construct: `score = mean(normalized activation of its present Yeo-7 networks) × 100`
(min-max normalization makes relative cross-network activation comparable).

```
approach = Σ |polarity_i|·score_i / Σ |polarity_i|     over covered POSITIVE constructs
avoid    = Σ |polarity_i|·score_i / Σ |polarity_i|     over covered NEGATIVE constructs
buy_sell = 100 · clip( 0.5 + (approach − 0.5) − 0.5·(avoid − 0.5), 0, 1 )
           Buy ≥ 60 · Hold · Sell ≤ 40
confidence = clip( (0.5 + 0.5·coverage) − 0.3·conflict, 0.2, 0.95 )
```
Reward dominates approach and conflict dominates avoidance, per the neuroforecasting evidence;
confidence rises with construct coverage and falls when conflict is high (Kapoor 2023: conflict
↑ ⇒ slower, harder decisions).

## Limitations (stated plainly)
- Surface atlases (Schaefer-1000/Yeo-7) lack subcortical reward/emotion/memory nuclei (NAcc,
  amygdala, hippocampus); those constructs use cortical proxies and are flagged `cortical_proxy`.
- The Digital Brain image encoder predicts **visual cortex only** — it cannot measure reward or
  conflict; the detector reports those as not-covered when that input is used.
- Per-stimulus activation, not correlation with measured behavior; this is decision-support, not a
  validated purchasing instrument.

## Source papers (supplied)
- Çakir, Çakar, Girisken & Yurdakul (2018). Neural correlates of purchase behavior through fNIRS. *Eur. J. Marketing* 52(1/2):224.
- Kapoor, Sahay, Singh, Pammi & Banerjee (2023). The neural correlates and underlying processes of weak brand choices. *J. Business Research* 154:113230.
- Shang, Deng & Liu (2018). Decision-making neural mechanism of online purchase intention. *NeuroQuantology* 16(5):246.
- Vlăsceanu (2014). New directions in understanding the decision-making process: neuroeconomics and neuromarketing. *Procedia SBS* 127:758.
- Knutson et al. (2007). Neural predictors of purchases. *Neuron* 53:147.
