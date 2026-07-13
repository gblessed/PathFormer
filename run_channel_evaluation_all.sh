#!/bin/bash
source ~/.bashrc
set -euo pipefail
conda activate pathformer

SCENARIOS=(
  "city_1_losangeles_3p5"
  "city_2_chicago_3p5"
  "city_3_houston_3p5"
  "city_4_phoenix_3p5"
  "city_5_philadelphia_3p5"
  "city_6_miami_3p5"
  "city_7_sandiego_3p5"
  "city_8_dallas_3p5"
  "city_9_sanfrancisco_3p5"
  "city_10_austin_3p5"
  "city_11_santaclara_3p5"
  "city_12_fortworth_3p5"
  "city_13_columbus_3p5"
  "city_18_denver_3p5"
)

CSV_OUT="/home/blessedg/Pathformer/logs/channel_eval_fixed_all.csv"
CHECKPOINT="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat/first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth"

cd /home/blessedg/Pathformer
mkdir -p /home/blessedg/Pathformer/logs
rm -f "$CSV_OUT"

for SCENARIO in "${SCENARIOS[@]}"; do
  echo "Evaluating ${SCENARIO}"
  python /home/blessedg/Pathformer/channel_finetune_embedding_all_families_foundation.py \
    --scenario "${SCENARIO}" \
    --skip-train \
    --pretrained-checkpoint "${CHECKPOINT}" \
    --csv-log-file "${CSV_OUT}"
done

echo "Saved evaluation results to ${CSV_OUT}"
