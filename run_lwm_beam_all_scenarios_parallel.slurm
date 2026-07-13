#!/bin/bash
#SBATCH --job-name=lwm-beam-all-par
#SBATCH --output=/home/blessedg/Pathformer/logs/lwm_beam_parallel_%j.out
#SBATCH --error=/home/blessedg/Pathformer/logs/lwm_beam_parallel_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1

source ~/.bashrc
conda activate pathformer
set -euo pipefail

ROOT_DIR="/home/blessedg/Pathformer"
LWM_DIR="$ROOT_DIR/lwm-competition-2025"
DATA_DIR="/home/blessedg/Pathformer/WiFo/dataset/blessed_task/"
RESULTS_CSV="$ROOT_DIR/logs/lwm_beam_results_mmwave.csv"
SCENARIO_LOG_DIR="$ROOT_DIR/logs/lwm_parallel"
SCENARIO_OUTPUT_DIR="$ROOT_DIR/logs/lwm_submissions"

TEST_SCENARIOS=(
    'city_0_newyork_3p5_s'
    'city_1_losangeles_3p5'
    'city_2_chicago_3p5'
    'city_3_houston_3p5'
    'city_4_phoenix_3p5'
    'city_5_philadelphia_3p5'
    'city_6_miami_3p5'
    'city_7_sandiego_3p5'
    'city_8_dallas_3p5'
    'city_95_delhi_3p5'
    'city_96_osaka_3p5'
    'city_10_austin_3p5'
    'city_11_santaclara_3p5'
    'city_12_fortworth_3p5'
    'city_13_columbus_3p5'
    'city_17_seattle_3p5_s'
    'city_18_denver_3p5'
    'city_19_oklahoma_3p5_s'
    # 'city_16_sanfrancisco_3p5_lwm'
    # 'city_23_beijing_3p5'
    # 'city_31_barcelona_3p5'
    # 'city_35_san_francisco_3p5'
    # 'city_47_chicago_3p5'
    # 'city_89_nairobi_3p5'
    # 'city_91_xiangyang_3p5'
    # 'city_92_sãopaulo_3p5'
    # 'boston5g_3p5'
    # 'city_86_ankara_3p5'
    # 'city_72_capetown_3p5'
    # 'city_84_baoding_3p5'
    # 'city_88_tongshan_3p5'
    # 'city_9_sanfrancisco_3p5'
)

TEST_SCENARIOS=(
    'city_1_losangeles_3p5'
    'city_2_chicago_3p5'
    'city_3_houston_3p5'
    'city_4_phoenix_3p5'
    'city_5_philadelphia_3p5'
    'city_6_miami_3p5'
    'city_7_sandiego_3p5'
    'city_8_dallas_3p5'
    'city_9_sanfrancisco_3p5'
    'city_10_austin_3p5'
    'city_11_santaclara_3p5'
    'city_12_fortworth_3p5'
    'city_13_columbus_3p5'
    'city_18_denver_3p5'
    'boston5g_3p5'
)


mkdir -p "$ROOT_DIR/logs" "$DATA_DIR" "$SCENARIO_LOG_DIR" "$SCENARIO_OUTPUT_DIR"

run_scenario() {
  local scenario="$1"
  local safe_scenario
  safe_scenario=$(printf '%s' "$scenario" | sed 's/[^[:alnum:]_]/_/g')
  local scenario_log="$SCENARIO_LOG_DIR/${safe_scenario}.log"

  {
    echo "[$(date)] Starting $scenario on host $(hostname)"

    cd "$ROOT_DIR"
    # python generate_lwm_beam_data.py \
    #   --scenario "$scenario" \
    #   --save-dir "$DATA_DIR/"

    echo "[$(date)] Fine-tuning LWM beam head for $scenario"
    cd "$LWM_DIR"
    python train_heads_custom.py \
      --scenario "$scenario" \
      --generated-data-dir "$DATA_DIR" \
      --task-idx 2 \
      --results-csv "$RESULTS_CSV" \
      --output-dir "$SCENARIO_OUTPUT_DIR/$safe_scenario" \
      --skip-visualization

    echo "[$(date)] Finished $scenario"
  } >"$scenario_log" 2>&1
}

declare -a pids=()
declare -a scenarios=()

for scenario in "${TEST_SCENARIOS[@]}"; do
  run_scenario "$scenario" &
  pids+=("$!")
  scenarios+=("$scenario")
done

failures=0
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


'''
python train_heads_custom.py \
  --scenario "city_2_chicago_3p5" \
  --generated-data-dir /home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc/ \
  --task-idx 5 \
  --results-csv "/home/blessedg/Pathformer/lwm-competition-2025/logs/test.csv" \
  --output-dir "/home/blessedg/Pathformer/lwm-competition-2025/logs/" \
  --skip-visualization

'''