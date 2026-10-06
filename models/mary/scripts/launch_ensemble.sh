#!/usr/bin/env bash
# Overnight faithful multi-dataset Mary training — N-seed ensemble, all DETACHED on Modal.
# Each seed = one parallel H100 job writing checkpoints/mary_multi_s{seed}/best.pt.
# Local cost ~0 (each `modal run --detach` submits then exits). Survives laptop sleep/crash.
#
# Usage:  bash scripts/launch_ensemble.sh "13 17 23"
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="${1:-13 17 23 29 37}"   # 5-seed ensemble → 5 parallel H100s (spend-for-speed)
CONFIG="configs/mary_multi.yaml"

echo "Launching ensemble seeds: $SEEDS  (config: $CONFIG)"
for s in $SEEDS; do
  echo "=== seed $s -> detached H100 job ==="
  modal run --detach train.py --config "$CONFIG" --seed "$s" 2>&1 \
    | grep -iE "View run|Error|InvalidError" | head -2 || true
  sleep 3   # stagger submissions
done
echo ""
echo "All ensemble jobs submitted detached. Monitor: https://modal.com/apps/robert-16572"
echo "Checkpoints land at /data/checkpoints/mary_multi_s{seed}/best.pt on volume sapient-data."
