# sapient-2 — Architecture

sapient-2 uses the **exact same architecture** as sapient-1, retrained on different data. The data flow and the 15 locked architecture decisions are identical to [`../sapient1/ARCHITECTURE.md`](../sapient1/ARCHITECTURE.md). This file calls out only what is sapient-2-specific.

## What's different from sapient-1

1. **n_subjects = ~75** (decision #13). Approximately 50 from ds004996 NeuroEngage + 25 from ds001740 Rauchbauer. Verify the exact retained count post-QC during data prep.
2. **Two lineages, two HF releases:**
   - `The-Sapient-Company/sapient-2-scratch-llama` — random initialization.
   - `The-Sapient-Company/sapient-2-ft-llama` — load weights from `The-Sapient-Company/sapient-1-llama`, **drop `subject_embed` on load** (decision #12), train with **`lr=5e-5`** (decision #11). All other hyperparameters unchanged.
3. **Training corpus is pooled.** ds004996 + ds001740 are concatenated into a single "HRI corpus" of ~26.5 h. Subject embeddings separate the populations.
4. **Cross-robot generalization.** ds001740 uses the Furhat robot; ds004996 uses different robot embodiments. Training on both forces sapient-2 to learn robot-agnostic conversation representations.

## What's identical

- Frozen encoders: V-JEPA 2 Gigantic (video, 1280d), W2V-BERT 2.0 (audio, 1024d), Llama-3.2-3B (text, 3072d).
- 8-layer transformer, hidden=1152, heads=8, ff=4608, pre-norm, GELU, dropout=0.1.
- Output: 20,484 fsaverage5 vertices @ 1 Hz.
- Two-stage low-rank head (1152 → 2048, no bias → 20484).
- Modality dropout 0.3 (after projections, before sum), subject dropout 0.1.
- Learnable pos_embed `(1, 100, 1152) × 0.02`, added post-pool.
- HRF offset of 5 TR.
- Masked MSE loss (per-vertex squared error → mean over vertices → mask-weighted mean over time).
- AdamW lr=1e-4 (scratch) or 5e-5 (ft), OneCycleLR with pct_start=0.1, batch 8, grad_accum 2, bf16 mixed.

## Why the `_ft` variant matters

The fine-tuned variant will almost certainly outperform the scratch variant on ds004996, because the model has already learned general stimulus→brain mappings from sapient-1's CNeuroMod training. **Train both, ship the winner.** Both are released for reproducibility and to demonstrate that the fine-tune lineage actually contains sapient-1-derived weights.

See [`../engineering-outline/08_sapient1_sapient2_spec.md`](../engineering-outline/08_sapient1_sapient2_spec.md) §4.6 for the load-state pattern:

```python
sapient1_state = torch.load('sapient1_final.pt')
state = {k: v for k, v in sapient1_state.items() if 'subject_embed' not in k}
model.load_state_dict(state, strict=False)
```
