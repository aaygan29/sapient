#!/usr/bin/env bash
# Autonomous overnight launcher: wait for CNeuroMod video extraction to wind down,
# rebuild the 4-dataset manifest, then launch the 5-seed ensemble. Self-contained.
set -uo pipefail
cd /Users/robertgutierrez/Desktop/sapient-models/mary
LOG=/tmp/overnight_orchestrate.log
echo "[$(date)] orchestrator start" | tee -a $LOG

# 1) Wait until running feature-extraction apps wind down (<=1 straggler), cap ~75 min.
for i in $(seq 1 38); do
  n=$(modal app list 2>/dev/null | awk -F'│' 'NR>3 && $5+0>0 {gsub(/ /,"",$3); print $3}' | grep -c "mary-featu")
  echo "[$(date)] running mary-featu apps: $n (poll $i)" | tee -a $LOG
  [ "${n:-0}" -le 1 ] && { echo "[$(date)] extraction wound down" | tee -a $LOG; break; }
  sleep 120
done

# 2) Rebuild the 4-dataset manifest (frozen data set — no new data).
echo "[$(date)] rebuilding 4-dataset manifest" | tee -a $LOG
modal run data/build_manifest_multi.py --datasets huth,lebel2023,had,cneuromod --test-spec "cneuromod:bourne,huth:forgot" 2>&1 | tail -6 | tee -a $LOG

# 3) Launch the 5-seed overnight ensemble (detached H100s).
echo "[$(date)] launching 5-seed ensemble" | tee -a $LOG
bash scripts/launch_ensemble.sh "13 17 23 29 37" 2>&1 | tee -a $LOG
echo "[$(date)] orchestrator done — ensemble submitted" | tee -a $LOG
