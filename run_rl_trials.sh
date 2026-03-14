#!/bin/bash
# Run small RL-after-SFT trials to find a setup that improves over SFT baseline.
#
# Get GPU first, then run this script:
#   srun --jobid=85713 --overlap --pty bash
#   conda activate pathformer
#   cd /home/blessedg/Pathformer && bash run_rl_trials.sh

set -e
cd /home/blessedg/Pathformer
SCENARIO="city_47_chicago_3p5"
LOG_DIR="logs/rl_trials"
mkdir -p "$LOG_DIR"

echo "========== SFT baseline (composite metric) =========="
python sft_plus_rl_training.py --scenario "$SCENARIO" --eval_only 2>&1 | tee "$LOG_DIR/baseline.txt"
BASELINE=$(grep 'composite eval metric' "$LOG_DIR/baseline.txt" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
echo "Baseline composite: $BASELINE"

run_trial() {
  local suffix=$1
  local epochs=$2
  local lr=$3
  local sft_coef=$4
  echo ""
  echo "========== Trial: $suffix (epochs=$epochs lr=$lr sft_coef=$sft_coef) =========="
  python sft_plus_rl_training.py --scenario "$SCENARIO" --epochs "$epochs" --lr "$lr" --sft_coef "$sft_coef" --eval_every_epochs 1 --trial_suffix "$suffix" 2>&1 | tee "$LOG_DIR/trial_${suffix}.txt"
  BEST=$(grep 'Best checkpoint saved' "$LOG_DIR/trial_${suffix}.txt" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  echo "Trial $suffix best composite: $BEST"
}

# Short trials (6 epochs) to quickly compare
run_trial "sft02_lr1e5" 6 1e-5 0.2
run_trial "sft15_lr2e5" 6 2e-5 0.15
run_trial "sft10_lr2e5" 6 2e-5 0.1
run_trial "sft05_lr1e5" 6 1e-5 0.05

echo ""
echo "========== Summary (lower composite = better) =========="
echo "Baseline: $BASELINE"
for s in sft02_lr1e5 sft15_lr2e5 sft10_lr2e5 sft05_lr1e5; do
  b=$(grep 'Best checkpoint saved' "$LOG_DIR/trial_${s}.txt" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  [ -n "$b" ] && echo "  $s: $b"
done
