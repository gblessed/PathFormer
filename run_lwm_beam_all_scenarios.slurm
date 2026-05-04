#!/bin/bash
#SBATCH --job-name=lwm-beam-all
#SBATCH --output=/home/blessedg/Pathformer/logs/lwm_beam_%A_%a.out
#SBATCH --error=/home/blessedg/Pathformer/logs/lwm_beam_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:tesla:1


source ~/.bashrc
conda activate pathformer
set -euo pipefail

ROOT_DIR="/home/blessedg/Pathformer"
LWM_DIR="$ROOT_DIR/lwm-competition-2025"
DATA_DIR="$LWM_DIR/blessed_task"
RESULTS_CSV="$ROOT_DIR/logs/lwm_beam_results.csv"

SCENARIOS=(
    'city_0_newyork_3p5_s'
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
    'city_17_seattle_3p5_s'
    'city_18_denver_3p5'
    'city_19_oklahoma_3p5_s'
    'city_16_sanfrancisco_3p5_lwm'
    'city_23_beijing_3p5'
    'city_31_barcelona_3p5'
    'city_35_san_francisco_3p5'
    'city_47_chicago_3p5'
    'city_89_nairobi_3p5'
    'city_91_xiangyang_3p5'
    'city_92_sãopaulo_3p5'
    'boston5g_3p5'
    'city_86_ankara_3p5'
    'city_72_capetown_3p5'
    'city_84_baoding_3p5'
    'city_95_delhi_3p5'
    'city_96_osaka_3p5'
    'city_88_tongshan_3p5'
)

mkdir -p "$ROOT_DIR/logs" "$DATA_DIR"
for SCENARIO in "${SCENARIOS[@]}"; do
  SAFE_SCENARIO=$(printf '%s' "$SCENARIO" | sed 's/[^[:alnum:]_]/_/g')

  echo "[$(date)] Generating beam data for $SCENARIO"
  cd "$ROOT_DIR"
  python generate_lwm_beam_data.py \
    --scenario "$SCENARIO" \
    --save-dir "$DATA_DIR/"

  echo "[$(date)] Fine-tuning LWM beam head for $SCENARIO"
  cd "$LWM_DIR"
  python train_heads_custom.py \
    --scenario "$SCENARIO" \
    --generated-data-dir "$DATA_DIR" \
    --task-idx 2 \
    --results-csv "$RESULTS_CSV" \
    --output-dir "$ROOT_DIR/logs/lwm_submissions/$SAFE_SCENARIO" \
    --skip-visualization

  echo "[$(date)] Finished $SCENARIO"
done
