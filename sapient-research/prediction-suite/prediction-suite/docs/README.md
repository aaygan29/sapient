# Prediction Suite — docs

Background and planning behind the code in `../`. Ordered from research to product.

| Doc | What it is |
|---|---|
| [paper-synthesis-batch1.md](./paper-synthesis-batch1.md) | Synthesis of the first 10 papers (TRIBE v2, Dalla Porta PNAS, CoTZero, PAGRL, HipDETR, CraniMem, hallucination study, NeuroCognition, attachment synchrony, B[FM]²) mapped to the Sapient program. |
| [paper-synthesis-batch2.md](./paper-synthesis-batch2.md) | Synthesis of the next 11 papers (BriLLM, hyperbolic brain graphs, EEG-to-text/image decoders, prompting-brain, LLM alignment, BCNE, cognitive autonomy, CATS Net, 8-dimension language semantics) and the capabilities they unlock. |
| [delivery-and-roadmap.md](./delivery-and-roadmap.md) | The three-feature MVP delivery summary, phased roadmap, expected performance, and integration checklist. |
| [features-and-pricing.md](./features-and-pricing.md) | Per-feature spec, API surface, validation protocol, success criteria, and pricing/go-to-market. |

## How the docs map to the code

- The **paper syntheses** are the idea funnel: which findings are worth building on.
- The **behavioral bridge** (`../engine/behavioral_bridge.py`) and its evidence table
  (`../RESEARCH_BASIS.md`) are what actually shipped from that funnel, restricted to the
  claims the published neuroforecasting literature can support today.
- The **roadmap** and **pricing** docs are forward-looking. Numbers in them are targets and
  projections, not results. Only `../engine/neuroforecasting.py` run on real outcomes earns a
  predictive claim.

Nothing here derives from private or classified work. See `../RESEARCH_BASIS.md` for the
public citation table that grounds every design choice.
