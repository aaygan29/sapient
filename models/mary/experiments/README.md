# Mary Model — Active Experiments
**Last updated: 2026-06-07**

---

## The Problem

Mary (`mary_multi_v2_s13`) is live in production predicting brain responses to ads.
The research shows three clear weak spots in the current model outputs:

| Brain Network | Current Performance | Noise Ceiling | Gap |
|---|---|---|---|
| **Default Mode** | ~35% of ceiling | r = 0.118 | biggest gap |
| **Limbic** | ~38% of ceiling | r = 0.032 | emotion/memory region |
| **Visual** | ~57% of ceiling | r = 0.237 | best but still room |

These networks matter most for ad analysis — Default Mode is where narrative
and memory consolidation happen; Limbic is emotion. We need to improve them.

Three hypotheses about what's holding the model back. Each gets its own experiment.

---

## Experiment 1 — Sharper Temporal Predictions (`exp_01_temporal_sharpening`)

**Hypothesis:** The model's predictions are over-smoothed in time (autocorrelation
0.95 vs real BOLD ~0.12). Sharpening the loss and reducing the HRF kernel forces
the model to track faster neural dynamics, which should lift Default Mode correlation.

**Changes from current model:**
- `negcorr_weight`: 1.0 → 2.0 (double the correlation-sharpening term)
- `hrf_kernel_size`: 5 → 3 (less temporal blurring from the HRF conv)
- `infonce_weight`: 0.3 → 0.5 (stronger contrastive signal across time)

**Expected outcome:** Higher Default Mode and Limbic r; possible drop in Visual (acceptable tradeoff).

---

## Experiment 2 — Deeper Fusion and Prediction Transformer (`exp_02_deeper_fusion`)

**Hypothesis:** A single fusion layer is a bottleneck for integrating 6 streams
of heterogeneous features. Adding a second fusion pass and a third prediction
layer gives the model more capacity to resolve cross-modal conflicts before
projecting to 20,484 vertices.

**Changes from current model:**
- `n_fusion_layers`: 1 → 2 (second fusion pass over all 6 streams)
- `n_prediction_layers`: 2 → 3 (extra temporal reasoning layer)
- `learning_rate`: 3e-4 → 1.5e-4 (lower LR to compensate for increased depth)

**Expected outcome:** Default Mode and Limbic gains from better cross-modal
integration; the model stops relying on a single stream's signal.

---

## Experiment 3 — Loss Rebalancing Toward Correlation (`exp_03_loss_rebalance`)

**Hypothesis:** MSE loss pushes the model toward predicting mean BOLD amplitude
which inflates loss on low-signal regions (Limbic, Default Mode) and under-weights
regions where *shape* of the response matters more than *magnitude*.

**Changes from current model:**
- `mse_weight`: 0.2 → 0.05 (dramatically reduce magnitude pressure)
- `negcorr_weight`: 1.0 → 2.5 (lead with correlation as primary objective)
- `infonce_weight`: 0.3 → 0.5
- `max_epochs`: 120 → 150 (give the new objective more time to converge)

**Expected outcome:** Biggest potential lift in Default Mode and Limbic.
Riskier — if it diverges, kill it and keep the other two.

---

## How to Launch

```bash
# From sapient-models/mary/
bash experiments/launch_experiments.sh
```

This fires all 3 experiments as **independent detached H100 jobs on Modal**.
Each one is isolated — a crash in one doesn't affect the others.

**Monitor live:** https://modal.com/apps/robert-16572

**Results land at:**
- `/data/checkpoints/exp_01_temporal_sharpening/best.pt`
- `/data/checkpoints/exp_02_attention_pooling/best.pt`
- `/data/checkpoints/exp_03_loss_rebalance/best.pt`

---

## How to Pick the Winner

After all three finish, run:
```bash
modal run eval.py --compare exp_01_temporal_sharpening exp_02_attention_pooling exp_03_loss_rebalance
```

Pick the checkpoint with the highest `default_mode_pearson` + `limbic_pearson` combined.
That version ships to production as the next Mary checkpoint.
