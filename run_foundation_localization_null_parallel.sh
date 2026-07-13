#!/bin/bash
#SBATCH --job-name=foundation-localize
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1

set -euo pipefail

set +u
source ~/.bashrc
set -u
conda activate pathformer

PROJECT_DIR="/home/blessedg/Pathformer"
LOG_DIR="${PROJECT_DIR}/logs"
SCRIPT_PATH="${PROJECT_DIR}/pre_train_finetune_user_localization_foundation_null.py"

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${PROJECT_DIR}/checkpoints_first_step_residual_corridor_concat/first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints_foundation_localization_null_corridor}"
RESULT_PREFIX="${RESULT_PREFIX:-foundation_localization_null}"
PER_SCENARIO_DIR="${PER_SCENARIO_DIR:-${LOG_DIR}/${RESULT_PREFIX}_per_scenario}"
MERGED_CSV="${MERGED_CSV:-${LOG_DIR}/${RESULT_PREFIX}_all.csv}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
POOL_MODE="${POOL_MODE:-mean}"
MAX_JOBS="${MAX_JOBS:-8}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
UNFREEZE_BACKBONE="${UNFREEZE_BACKBONE:-0}"
USE_MATERIAL_FEATURES="${USE_MATERIAL_FEATURES:-1}"
NULL_RX_POS="${NULL_RX_POS:-1}"
NULL_SCENE_FEATURES="${NULL_SCENE_FEATURES:-1}"
NULL_CLUSTER_PRIOR="${NULL_CLUSTER_PRIOR:-0}"
N_CLUSTERS="${N_CLUSTERS:-25}"
NEAREST_K="${NEAREST_K:-5}"
CORRIDOR_K="${CORRIDOR_K:-5}"
CORRIDOR_BINS="${CORRIDOR_BINS:-8}"

SCENARIOS=(
    "city_47_chicago_3p5"
    "city_0_newyork_3p5_s"
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
    "city_17_seattle_3p5_s"
    "city_18_denver_3p5"
    "city_19_oklahoma_3p5_s"
    "city_16_sanfrancisco_3p5_lwm"
    "city_23_beijing_3p5"
    "city_31_barcelona_3p5"
    "city_35_san_francisco_3p5"
    "city_89_nairobi_3p5"
    "city_91_xiangyang_3p5"
    "city_92_sãopaulo_3p5"
    "boston5g_3p5"
    "city_86_ankara_3p5"
    "city_72_capetown_3p5"
    "city_84_baoding_3p5"
    "city_95_delhi_3p5"
    "city_96_osaka_3p5"
    "city_88_tongshan_3p5"
)

mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${PER_SCENARIO_DIR}"

if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "Foundation checkpoint not found: ${PRETRAINED_CHECKPOINT}" >&2
    exit 1
fi

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

build_skip_flag() {
    if [[ "${SKIP_TRAIN}" == "1" ]]; then
        printf '%s' "--skip-train"
    else
        printf '%s' ""
    fi
}

build_eval_flag() {
    if [[ "${EVAL_ONLY}" == "1" ]]; then
        printf '%s' "--eval-only"
    else
        printf '%s' ""
    fi
}

build_unfreeze_flag() {
    if [[ "${UNFREEZE_BACKBONE}" == "1" ]]; then
        printf '%s' "--unfreeze-backbone"
    else
        printf '%s' ""
    fi
}

build_rx_flag() {
    if [[ "${NULL_RX_POS}" == "1" ]]; then
        printf '%s' "--null-rx-pos"
    else
        printf '%s' "--keep-rx-pos"
    fi
}

build_scene_flag() {
    if [[ "${NULL_SCENE_FEATURES}" == "1" ]]; then
        printf '%s' "--null-scene-features"
    else
        printf '%s' "--keep-scene-features"
    fi
}

build_cluster_flag() {
    if [[ "${NULL_CLUSTER_PRIOR}" == "1" ]]; then
        printf '%s' "--null-cluster-prior"
    else
        printf '%s' ""
    fi
}

MATERIAL_FLAG="$(build_material_flag)"
SKIP_FLAG="$(build_skip_flag)"
EVAL_FLAG="$(build_eval_flag)"
UNFREEZE_FLAG="$(build_unfreeze_flag)"
RX_FLAG="$(build_rx_flag)"
SCENE_FLAG="$(build_scene_flag)"
CLUSTER_FLAG="$(build_cluster_flag)"

echo "Using foundation checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "Localization checkpoints: ${CHECKPOINT_DIR}"
echo "Per-scenario CSVs: ${PER_SCENARIO_DIR}"
echo "Merged CSV: ${MERGED_CSV}"
echo "Running up to ${MAX_JOBS} scenario jobs in parallel"

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
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --grad-clip-norm "${GRAD_CLIP_NORM}" \
        --pool-mode "${POOL_MODE}" \
        --n-clusters "${N_CLUSTERS}" \
        --nearest-k "${NEAREST_K}" \
        --corridor-k "${CORRIDOR_K}" \
        --corridor-bins "${CORRIDOR_BINS}" \
        "${MATERIAL_FLAG}" \
        "${RX_FLAG}" \
        "${SCENE_FLAG}" \
        ${CLUSTER_FLAG} \
        ${UNFREEZE_FLAG} \
        ${SKIP_FLAG} \
        ${EVAL_FLAG} \
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

echo "Finished localization runs for ${#SCENARIOS[@]} scenarios."
echo "Merged results saved to ${MERGED_CSV}"
