# Consumer-neuro grounding for the buy/sell signals

The buy/sell layer (`sapient_serving/engine/signals.py`) is not ad-hoc. It implements
the **approach − avoidance** neuroforecasting model and maps the encoder's Yeo-7
network predictions to consumer constructs using the sources below.

## Sources & what we took from each
1. **Knutson, Rick, Wimmer, Prelec & Loewenstein (2007)**, *Neuron* 53:147 — "Neural
   predictors of purchases." NAcc / mPFC reward → buy; **anterior insula** ("pain of
   paying") → don't buy. ⇒ the core approach−avoidance form.
2. **Kapoor, Sahay, Singh, Pammi & Banerjee (2023)**, *J. Business Research* 154:113230
   (fMRI). Preferred/strong choice → **left ventral striatum** (reward) + L-DLPFC;
   hesitant/weak choice → **rostral + dorsal ACC** (conflict, error commission, regret)
   + **R-DLPFC**; ACC conflict scales with longer response time. ⇒ Decision-Conflict
   (salience/ACC/insula) is a **Sell** driver; reward is **Buy**; conflict ⇒ lower confidence.
3. **Çakir, Çakar, Girisken & Yurdakul (2018)**, *Eur. J. Marketing* 52(1/2):224 (fNIRS).
   Positive purchase decisions ↑ **fronto-polar / OFC / vmPFC** (subjective value);
   buy/pass decoded at **85%** once **budget sensitivity** is added. ⇒ reward/value =
   vmPFC/OFC/fronto-polar; price/budget sensitivity is a moderator and a decoding target.
4. **Shang, Deng & Liu (2018)**, *NeuroQuantology* 16(5):246 (ERP). Online purchase
   staged **N2 (risk) → N400 (conflict) → LPP (value/valence)**; **involvement** moderates
   price vs reputation. ⇒ risk/conflict precede value; involvement is a moderator.
5. **Vlăsceanu (2014)**, *Procedia SBS* 127:758 (review). vmPFC + amygdala reward &
   somatic markers; dual-process (System-1 emotion vs System-2 deliberation, gated by
   load); strong brand → prefrontal. ⇒ emotion = approach; DLPFC deliberation/load = friction.

## Construct → network → polarity (see `signals.py:CONSTRUCTS`)
| Construct | Yeo-7 networks | Region rationale | Polarity |
|---|---|---|---|
| Reward & Value | Limbic, Default | vmPFC / OFC / ventral striatum | + (buy) |
| Emotional Resonance | Limbic | amygdala / affect | + |
| Attention Capture | DorsalAttention | orienting | + |
| Visual & Sensory Salience | Visual | retinotopic | + |
| Memory Encoding | Default | brand memory | + |
| Decision Conflict & Risk | VentralAttention | anterior insula + ACC | − (sell) |
| Cognitive Load (Deliberation) | Frontoparietal | DLPFC / dmPFC | − |

`purchase_intent = 100·clip(0.5 + (approach−0.5) − 0.5·(avoid−0.5), 0, 1)`
→ **Buy ≥ 60 · Hold · Sell ≤ 40**. Confidence falls with mean Decision-Conflict
(Kapoor: conflict ↑ ⇒ slower, harder decisions).

## How this should shape the REAL model (task #5)
When the encoder is wired (`engine/real.py`), the buy/sell readout should be a
**region-weighted projection** of predicted activation, not a flat ROI average:
- weight value/reward parcels (vmPFC/OFC + ventral-striatum-adjacent Limbic & medial DMN) **positively**;
- weight salience parcels (anterior insula + ACC → VentralAttention) **negatively** (conflict / pain of paying);
- weight DLPFC/dmPFC (Frontoparietal) **negatively** (deliberation friction).

Build the parcel→construct weight matrix from the canonical Schaefer-1000 atlas labels
(the same atlas used in `postprocess.py`).

### Validation targets for eval
- **Çakir:** buy vs pass decodable from PFC activation (~85% with budget sensitivity) —
  use as a target accuracy for a buy/pass classifier head.
- **Kapoor:** higher Decision-Conflict ⇒ longer predicted deliberation / lower confidence;
  reward higher for chosen/preferred options.
- **Shang:** add an **involvement** moderator (price weight ↑ for low-involvement goods,
  brand/reputation weight ↑ for high-involvement).
- **Knutson/Çakir:** **budget/price sensitivity** and **involvement** are candidate
  conditioning inputs to the readout.
