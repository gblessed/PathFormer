#!/usr/bin/env bash
set -eo pipefail

set +u
source ~/.bashrc
set -u
conda activate pathformer

SCENARIO="city_23_beijing_3p5"
TRAIN_RATIOS=(0.2 0.4 0.6 0.8)
SEEDS=(0 1 2)
NOISE_PROB=0.2

LOG_DIR="/home/blessedg/Pathformer/logs"
DIRECT_CKPT_DIR="/home/blessedg/Pathformer/direct_train_ratio_seed_sweep"
FIRST_STEP_CKPT_DIR="/home/blessedg/Pathformer/first_step_train_ratio_seed_sweep"
CORRIDOR_CKPT_DIR="/home/blessedg/Pathformer/corridor_train_ratio_seed_sweep"

mkdir -p "$LOG_DIR" "$DIRECT_CKPT_DIR" "$FIRST_STEP_CKPT_DIR" "$CORRIDOR_CKPT_DIR"

sanitize_ratio() {
  echo "$1" | sed 's/[^a-zA-Z0-9_]/_/g'
}

echo "Launching seed sweep for scenario: $SCENARIO"
echo "Train ratios: ${TRAIN_RATIOS[*]}"
echo "Seeds: ${SEEDS[*]}"

run_family() {
  local family_name="$1"
  local script_path="$2"
  local ckpt_dir="$3"
  local file_prefix="$4"

  echo "========================================"
  echo "Starting family: $family_name"
  echo "========================================"

  for seed in "${SEEDS[@]}"; do
    for train_ratio in "${TRAIN_RATIOS[@]}"; do
      ratio_tag="$(sanitize_ratio "$train_ratio")"
      base="${file_prefix}_${SCENARIO}_r${ratio_tag}_seed${seed}"
      echo "Running ${family_name} | seed=${seed} | train_ratio=${train_ratio}"
      python "$script_path" \
        --scenario "$SCENARIO" \
        --checkpoint-dir "$ckpt_dir" \
        --csv-log-file "$LOG_DIR/${base}.csv" \
        --noise-prob "$NOISE_PROB" \
        --train-ratio "$train_ratio" \
        --seed "$seed" \
        > "$LOG_DIR/${base}.out" \
        2> "$LOG_DIR/${base}.err" &
    done
  done

  wait
  echo "Completed family: $family_name"
}

run_family \
  "direct" \
  "/home/blessedg/Pathformer/multiscenario_direct_training_ablate.py" \
  "$DIRECT_CKPT_DIR" \
  "direct_train_ratio"

run_family \
  "first_step_residual" \
  "/home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_ablate.py" \
  "$FIRST_STEP_CKPT_DIR" \
  "first_step_train_ratio"

run_family \
  "first_step_residual_corridor" \
  "/home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor_ablate.py" \
  "$CORRIDOR_CKPT_DIR" \
  "corridor_train_ratio"

echo "All seed-sweep training runs finished."
