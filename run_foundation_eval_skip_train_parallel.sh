#!/bin/bash
#SBATCH --job-name=foundation-eval
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1

source ~/.bashrc
conda activate pathformer

set -euo pipefail

PROJECT_DIR="/home/blessedg/Pathformer"
LOG_DIR="${PROJECT_DIR}/logs"
SCRIPT_PATH="${PROJECT_DIR}/multiscenario_direct_training_first_step_residual_corridor_finetune.py"

CHECKPOINT_DIR_DEFAULT="${PROJECT_DIR}/checkpoints_first_step_residual_corridor_concat"
PRETRAINED_CHECKPOINT_DEFAULT="${CHECKPOINT_DIR_DEFAULT}/first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth"

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${PRETRAINED_CHECKPOINT_DEFAULT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints_first_step_residual_corridor_finetune}"
RESULT_PREFIX="${RESULT_PREFIX:-foundation_skip_train_eval}"
PER_SCENARIO_DIR="${PER_SCENARIO_DIR:-${LOG_DIR}/${RESULT_PREFIX}_per_scenario}"
MERGED_CSV="${MERGED_CSV:-${LOG_DIR}/${RESULT_PREFIX}_all.csv}"
NOISE_PROB="${NOISE_PROB:-0.0}"
N_CLUSTERS="${N_CLUSTERS:-25}"
NEAREST_K="${NEAREST_K:-5}"
CORRIDOR_K="${CORRIDOR_K:-5}"
CORRIDOR_BINS="${CORRIDOR_BINS:-8}"
USE_MATERIAL_FEATURES="${USE_MATERIAL_FEATURES:-1}"
MAX_JOBS="${MAX_JOBS:-32}"

mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${PER_SCENARIO_DIR}"

if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "Pretrained checkpoint not found: ${PRETRAINED_CHECKPOINT}" >&2
    exit 1
fi

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

sanitize_name() {
    printf '%s' "$1" | sed 's/[^[:alnum:]_]/_/g'
}

wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 2
    done
}

build_material_flag() {
    if [[ "${USE_MATERIAL_FEATURES}" == "1" ]]; then
        printf '%s' "--use-material-features"
    else
        printf '%s' "--no-material-features"
    fi
}

MATERIAL_FLAG="$(build_material_flag)"

echo "Using pretrained checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "Writing per-scenario CSVs to: ${PER_SCENARIO_DIR}"
echo "Merged CSV will be: ${MERGED_CSV}"
echo "Running up to ${MAX_JOBS} evaluation jobs in parallel"

for scenario in "${SCENARIOS[@]}"; do
    wait_for_slot

    safe_name="$(sanitize_name "${scenario}")"
    scenario_csv="${PER_SCENARIO_DIR}/${safe_name}.csv"
    scenario_out="${PER_SCENARIO_DIR}/${safe_name}.out"
    scenario_err="${PER_SCENARIO_DIR}/${safe_name}.err"

    python "${SCRIPT_PATH}" \
        --scenario "${scenario}" \
        --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --csv-log-file "${scenario_csv}" \
        --noise-prob "${NOISE_PROB}" \
        --n-clusters "${N_CLUSTERS}" \
        --nearest-k "${NEAREST_K}" \
        --corridor-k "${CORRIDOR_K}" \
        --corridor-bins "${CORRIDOR_BINS}" \
        "${MATERIAL_FLAG}" \
        --skip-train \
        > "${scenario_out}" \
        2> "${scenario_err}" &
done

wait

mapfile -t csv_files < <(find "${PER_SCENARIO_DIR}" -maxdepth 1 -type f -name '*.csv' | sort)

if (( ${#csv_files[@]} == 0 )); then
    echo "No per-scenario CSV files were produced." >&2
    exit 1
fi

rm -f "${MERGED_CSV}"
cat "${csv_files[0]}" > "${MERGED_CSV}"
for csv_file in "${csv_files[@]:1}"; do
    tail -n +2 "${csv_file}" >> "${MERGED_CSV}"
done

echo "Finished evaluating ${#SCENARIOS[@]} scenarios."
echo "Merged results saved to ${MERGED_CSV}"
