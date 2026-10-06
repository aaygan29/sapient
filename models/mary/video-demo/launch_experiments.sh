#!/usr/bin/env bash
# ============================================================
# Mary — Parallel Experiment Launcher
# Fires all 3 experiments as independent detached H100 jobs.
# Each job is isolated: a crash in one does not affect others.
#
# Usage:
#   bash video-demo/launch_experiments.sh
#
# Monitor live:
#   https://modal.com/apps/robert-16572
#
# Status: READY — run this to start
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS=(
  "video-demo/exp_01_temporal_sharpening.yaml"
  "video-demo/exp_02_attention_pooling.yaml"
  "video-demo/exp_03_loss_rebalance.yaml"
)

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         Mary — Launching 3 Parallel Experiments     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

for config in "${EXPERIMENTS[@]}"; do
  name=$(grep 'variant_name:' "$config" | awk '{print $2}')
  echo "▶ Submitting: $name"
  modal run --detach train.py --config "$config" 2>&1 \
    | grep -iE "View run|Error|InvalidError" | head -2 || true
  sleep 2
done

echo ""
echo "✓ All 3 experiments submitted."
echo ""
echo "  Monitor:  https://modal.com/apps/robert-16572"
echo "  Results:  /data/checkpoints/{exp_01,exp_02,exp_03}/best.pt"
echo ""
echo "  When done, compare with:"
echo "  modal run eval.py --compare exp_01_temporal_sharpening exp_02_attention_pooling exp_03_loss_rebalance"
echo ""
