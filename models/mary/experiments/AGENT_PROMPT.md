# Mary — Experiment Cycle Agent

**Model:** `mary_multi_v2_s13` (live in production)
**Goal:** Identify the highest-leverage improvement to Default Mode and Limbic network
prediction, run the three queued experiments in parallel, and return a shipping recommendation.

This is the standard experiment cycle we run after each production eval. Follow the
steps below in order.

---

## Step 1 — Orient (read before touching anything)

Read these in order:

1. `mary/experiments/README.md` — active experiment queue and known weak spots
2. `mary/configs/mary_base.yaml` — architecture constraints (locked)
3. `mary/mary/model.py` — model implementation
4. `mary/mary/losses.py` — composite loss (MSE + NegCorr + InfoNCE)
5. `mary/mary/adapter.py` — 8-step forward pass
6. `mary/train.py` — training loop and Modal GPU entrypoint
7. `sapient-research/Mary-Model-Insights/03-experiments-and-results.md` — benchmark
   numbers from the last eval cycle; where the model underperforms and why

After reading, return a brief orientation summary:
- Current production strengths (cite Yeo-7 numbers)
- Confirmed weak spots with root cause analysis
- Ranked view of the 3 queued experiments by expected impact

Hold for confirmation before Step 2.

---

## Step 2 — Validate configs

Read and sanity-check the three queued experiment configs:
- `mary/experiments/exp_01_temporal_sharpening.yaml`
- `mary/experiments/exp_02_attention_pooling.yaml`
- `mary/experiments/exp_03_loss_rebalance.yaml`

For each, verify:
- Hypothesis is consistent with the benchmark findings (cite the numbers)
- Hyperparameter changes are internally consistent
- No conflicts with locked architecture constraints

Surface any issues. If clean, confirm ready to launch.

---

## Step 3 — Launch

```bash
bash mary/experiments/launch_experiments.sh
```

Fires 3 independent detached H100 jobs on Modal. Each is fully isolated.

Live dashboard: https://modal.com/apps/robert-16572

---

## Step 4 — Results and recommendation

When jobs complete, compare on:
- `default_mode_pearson` — primary metric (narrative + memory; core to ad scoring)
- `limbic_pearson` — secondary (emotion)
- `visual_pearson` — tertiary

Return a comparison table and a single shipping recommendation with justification.
If two runs are within 5% on the primary metric, flag for ensemble consideration.

---

## Standing constraints

- `mary/mary/model.py` and `mary/mary/adapter.py` are read-only — architecture is locked
- `mary/configs/mary_base.yaml` is read-only
- All Modal jobs must use `--detach` — never block the local process
- HF org: `The-Sapient-Company` only
