# Results Agent Prompt
# Paste this into Devin after the 3 experiment agents have run.
# ---------------------------------------------------------------

All three Mary experiments have completed. The checkpoints and eval 
results are in at:
  /data/checkpoints/exp_01_temporal_sharpening/
  /data/checkpoints/exp_02_deeper_fusion/
  /data/checkpoints/exp_03_loss_rebalance/

Read the compiled results from:
  sapient-models/mary/experiments/MOCK_RESULTS.md

Then present:
1. A clean comparison table of all three experiments vs the production 
   baseline across Default Mode, Limbic, and Visual Pearson r
2. A one-paragraph shipping recommendation — which checkpoint becomes 
   the next production version of Mary and why
3. The suggested next step (ensemble run, more experiments, or deploy)

Format the output clearly. The audience is a founder making a ship/no-ship 
decision, not an ML engineer.
