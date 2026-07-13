#!/bin/bash
#SBATCH --job-name=wifo-beam-par
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:h100:1

source ~/.bashrc
conda activate wifo

set -euo pipefail

PROJECT_DIR="/home/blessedg/Pathformer"
SCRIPT_DIR="${PROJECT_DIR}/WiFo/src"
SCRIPT_PATH="${SCRIPT_DIR}/finetune_beam.py"
LOG_DIR="${PROJECT_DIR}/logs"

CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/weights/wifo_base}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/WiFo/dataset/blessed_task}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/WiFo/beam_finetune_runs_parallel}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/WiFo}"
PER_SCENARIO_DIR="${PER_SCENARIO_DIR:-${RESULTS_ROOT}/beam_finetune_per_scenario_parallel}"
MERGED_CSV="${MERGED_CSV:-${RESULTS_ROOT}/beam_finetune_results_parallel_merged_mmwave.csv}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MODEL_SIZE="${MODEL_SIZE:-base}"
PATCH_SIZE="${PATCH_SIZE:-4}"
T_PATCH_SIZE="${T_PATCH_SIZE:-4}"
POS_EMB="${POS_EMB:-SinCos3D}"
SEED="${SEED:-42}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-16}"

# SCENARIOS=(
#     'city_0_newyork_3p5_s'
#     'city_1_losangeles_3p5'
#     'city_2_chicago_3p5'
#     'city_3_houston_3p5'
#     'city_4_phoenix_3p5'
#     'city_5_philadelphia_3p5'
#     'city_6_miami_3p5'
#     'city_7_sandiego_3p5'
#     'city_8_dallas_3p5'
#     'city_9_sanfrancisco_3p5'
#     'city_10_austin_3p5'
#     'city_11_santaclara_3p5'
#     'city_12_fortworth_3p5'
#     'city_13_columbus_3p5'
#     'city_17_seattle_3p5_s'
#     'city_18_denver_3p5'
#     'city_19_oklahoma_3p5_s'
#     'city_16_sanfrancisco_3p5_lwm'
#     'city_23_beijing_3p5'
#     'city_31_barcelona_3p5'
#     'city_35_san_francisco_3p5'
#     'city_47_chicago_3p5'
#     'city_89_nairobi_3p5'
#     'city_91_xiangyang_3p5'
#     'city_92_sãopaulo_3p5'
#     'boston5g_3p5'
#     'city_86_ankara_3p5'
#     'city_72_capetown_3p5'
#     'city_84_baoding_3p5'
#     'city_95_delhi_3p5'
#     'city_96_osaka_3p5'
#     'city_88_tongshan_3p5'
# )

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


mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${PER_SCENARIO_DIR}"

if [[ ! -f "${CHECKPOINT}" && ! -f "${CHECKPOINT}.pkl" ]]; then
    echo "WiFo checkpoint not found: ${CHECKPOINT} or ${CHECKPOINT}.pkl" >&2
    exit 1
fi

sanitize_name() {
    printf '%s' "$1" | sed 's/[^[:alnum:]_]/_/g'
}

wait_for_slot() {
    local max_jobs="$1"
    while (( $(jobs -rp | wc -l) >= max_jobs )); do
        sleep 2
    done
}

run_one_scenario() {
    local scenario="$1"
    local device_id="$2"
    local output_dir="$3"
    local scenario_csv="$4"
    local scenario_out="$5"
    local scenario_err="$6"

    cd "${SCRIPT_DIR}"
    CUDA_VISIBLE_DEVICES="${device_id}" python "${SCRIPT_PATH}" \
        --checkpoint "${CHECKPOINT}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${output_dir}" \
        --results-csv "${scenario_csv}" \
        --scenario "${scenario}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --head-hidden-dim "${HEAD_HIDDEN_DIM}" \
        --head-dropout "${HEAD_DROPOUT}" \
        --num-workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --device-id 0 \
        --size "${MODEL_SIZE}" \
        --patch-size "${PATCH_SIZE}" \
        --t-patch-size "${T_PATCH_SIZE}" \
        --pos-emb "${POS_EMB}" \
        > "${scenario_out}" \
        2> "${scenario_err}"
}

run_split_parallel() {
    local split_name="$1"
    local device_id="$2"
    shift 2
    local scenarios=("$@")

    local split_output_dir="${OUTPUT_ROOT}/${split_name}"
    local split_csv_dir="${PER_SCENARIO_DIR}/${split_name}"
    mkdir -p "${split_output_dir}" "${split_csv_dir}"

    echo "Launching ${#scenarios[@]} scenarios for ${split_name} on GPU ${device_id}"
    printf '  %s\n' "${scenarios[@]}"

    for scenario in "${scenarios[@]}"; do
        wait_for_slot "${MAX_JOBS_PER_GPU}"
        local safe_name
        safe_name="$(sanitize_name "${scenario}")"
        local scenario_csv="${split_csv_dir}/${safe_name}.csv"
        local scenario_out="${split_csv_dir}/${safe_name}.out"
        local scenario_err="${split_csv_dir}/${safe_name}.err"
        local scenario_output_dir="${split_output_dir}/${safe_name}"

        run_one_scenario \
            "${scenario}" \
            "${device_id}" \
            "${scenario_output_dir}" \
            "${scenario_csv}" \
            "${scenario_out}" \
            "${scenario_err}" &
    done

    wait
}

HALF=$(( (${#SCENARIOS[@]} + 1) / 2 ))
SPLIT0_SCENARIOS=("${SCENARIOS[@]}")
# SPLIT0_SCENARIOS=("${SCENARIOS[@]:0:${HALF}}")
# SPLIT1_SCENARIOS=("${SCENARIOS[@]:${HALF}}")

echo "Using WiFo checkpoint: ${CHECKPOINT}"
echo "Per-scenario CSV directory: ${PER_SCENARIO_DIR}"
echo "Merged CSV: ${MERGED_CSV}"
echo "Max parallel scenario jobs per GPU: ${MAX_JOBS_PER_GPU}"

run_split_parallel "split0" 0 "${SPLIT0_SCENARIOS[@]}" &
pid_a=$!

# run_split_parallel "split1" 1 "${SPLIT1_SCENARIOS[@]}" &
# pid_b=$!

wait "${pid_a}"
# wait "${pid_b}"

mapfile -t csv_files < <(find "${PER_SCENARIO_DIR}" -type f -name '*.csv' | sort)

if (( ${#csv_files[@]} == 0 )); then
    echo "No per-scenario CSV files were produced." >&2
    exit 1
fi

rm -f "${MERGED_CSV}"
cat "${csv_files[0]}" > "${MERGED_CSV}"
for csv_file in "${csv_files[@]:1}"; do
    tail -n +2 "${csv_file}" >> "${MERGED_CSV}"
done

echo "Finished WiFo beam finetuning runs."
echo "Merged results saved to ${MERGED_CSV}"
