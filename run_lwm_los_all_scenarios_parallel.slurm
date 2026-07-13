#!/bin/bash
#SBATCH --job-name=lwm-los-all-par
#SBATCH --output=/home/blessedg/Pathformer/logs/lwm_los_parallel_%j.out
#SBATCH --error=/home/blessedg/Pathformer/logs/lwm_los_parallel_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1

source ~/.bashrc
conda activate pathformer
set -euo pipefail

ROOT_DIR="/home/blessedg/Pathformer"
LWM_DIR="$ROOT_DIR/lwm-competition-2025"
DATA_DIR="/home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc"
RESULTS_CSV="$ROOT_DIR/logs/los_comparisons/lwm_los.csv"
SCENARIO_LOG_DIR="$ROOT_DIR/lwm-competition-2025/logs/lwm_los_parallel"
SCENARIO_OUTPUT_DIR="$ROOT_DIR/lwm-competition-2025/logs/lwm_los_submissions"
MAX_PARALLEL=14

mkdir -p "$ROOT_DIR/logs" "$DATA_DIR" "$SCENARIO_LOG_DIR" "$SCENARIO_OUTPUT_DIR" "$(dirname "$RESULTS_CSV")"

mapfile -t TEST_SCENARIOS < <(
  find "$DATA_DIR" -maxdepth 1 -type f -name '_*_train_data.pt' -printf '%f\n' \
    | sed -E 's/^_(.*)_train_data\.pt$/\1/' \
    | sort
)

if (( ${#TEST_SCENARIOS[@]} == 0 )); then
  echo "[$(date)] No scenario train files found in $DATA_DIR" >&2
  exit 1
fi

run_scenario() {
  local scenario="$1"
  local safe_scenario
  safe_scenario=$(printf '%s' "$scenario" | sed 's/[^[:alnum:]_]/_/g')
  local scenario_log="$SCENARIO_LOG_DIR/${safe_scenario}.log"

  {
    echo "[$(date)] Starting $scenario on host $(hostname)"
    echo "[$(date)] Fine-tuning LWM LoS head for $scenario"

    cd "$LWM_DIR"
    python train_heads_custom.py \
      --scenario "$scenario" \
      --generated-data-dir "$DATA_DIR" \
      --task-idx 1 \
      --results-csv "$RESULTS_CSV" \
      --output-dir "$SCENARIO_OUTPUT_DIR/$safe_scenario" \
      --skip-visualization

    echo "[$(date)] Finished $scenario"
  } >"$scenario_log" 2>&1
}

declare -a pids=()
declare -a scenarios=()
failures=0

for scenario in "${TEST_SCENARIOS[@]}"; do
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    wait "${pids[0]}" || failures=$((failures + 1))
    pids=("${pids[@]:1}")
    scenarios=("${scenarios[@]:1}")
  done

  run_scenario "$scenario" &
  pids+=("$!")
  scenarios+=("$scenario")
done

for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  scenario="${scenarios[$idx]}"
  if wait "$pid"; then
    echo "[$(date)] Scenario succeeded: $scenario"
  else
    echo "[$(date)] Scenario failed: $scenario" >&2
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  echo "[$(date)] $failures scenario(s) failed. Check $SCENARIO_LOG_DIR for details." >&2
  exit 1
fi

echo "[$(date)] All scenarios finished successfully."
