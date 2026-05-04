#!/usr/bin/env bash
set -eo pipefail

set +u
source ~/.bashrc
set -u
conda activate pathformer

SCENARIO="city_23_beijing_3p5"
TRAIN_RATIOS=(0.6 0.8)
SEEDS=(0 1 2)
NOISE_PROB=0.2

LOG_DIR="/home/blessedg/Pathformer/logs"
CKPT_DIR="/home/blessedg/Pathformer/corridor_train_ratio_seed_sweep"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

sanitize_ratio() {
  echo "$1" | sed 's/[^a-zA-Z0-9_]/_/g'
}

for seed in "${SEEDS[@]}"; do
  for train_ratio in "${TRAIN_RATIOS[@]}"; do
    ratio_tag="$(sanitize_ratio "$train_ratio")"
    base="corridor_train_ratio_${SCENARIO}_r${ratio_tag}_seed${seed}"
    echo "Running corridor | seed=${seed} | train_ratio=${train_ratio}"
    python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor_ablate.py \
      --scenario "$SCENARIO" \
      --checkpoint-dir "$CKPT_DIR" \
      --csv-log-file "$LOG_DIR/${base}.csv" \
      --noise-prob "$NOISE_PROB" \
      --train-ratio "$train_ratio" \
      --seed "$seed" \
      > "$LOG_DIR/${base}.out" \
      2> "$LOG_DIR/${base}.err" &
  done
done

wait
echo "Finished missing corridor 0.6/0.8 seed runs."
