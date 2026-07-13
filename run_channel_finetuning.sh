#!/bin/bash
#SBATCH --job-name=channel-paf
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:h100:1

source ~/.bashrc
conda activate pathformer



SCENARIOS=(
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
)


sanitize_name() {

  printf '%s' "$1" | sed 's/[^[:alnum:]_]/_/g'
}

cd /home/blessedg/Pathformer
PER_SCENARIO_DIR='/home/blessedg/Pathformer/logs/channel_estimation_parallel'
mkdir -p "$PER_SCENARIO_DIR"

for SCENARIO in "${SCENARIOS[@]}"; do
  echo "training for scenario $SCENARIO"
  safe_name="$(sanitize_name "${SCENARIO}")"
  scenario_out="${PER_SCENARIO_DIR}/${safe_name}.out"
  scenario_err="${PER_SCENARIO_DIR}/${safe_name}.err"
  python /home/blessedg/Pathformer/channel_finetune_embedding_all_families_foundation.py \
    --pretrained-checkpoint /home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat/first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth --scenario $SCENARIO   > "${scenario_out}" \
  2> "${scenario_err}" & 
  done
wait



# python /home/blessedg/Pathformer/channel_finetune_embedding_all_families_foundation.py \
#   --pretrained-checkpoint /home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat/first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth --skip-train