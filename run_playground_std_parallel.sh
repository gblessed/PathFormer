#!/bin/bash
#SBATCH --job-name=fine-eval
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:1

source ~/.bashrc
conda activate pathformer

mkdir -p /home/blessedg/Pathformer/logs
mkdir -p /home/blessedg/Pathformer/checkpoints_std


# SCENARIOS=(
#   "city_47_chicago_3p5"
#   "city_23_beijing_3p5"
#   "city_91_xiangyang_3p5"
#   "city_17_seattle_3p5_s"
#   "city_12_fortworth_3p5"
# )

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

# SCENARIOS=(
#    'city_19_oklahoma_3p5_s'
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

# SCENARIOS=(
#   "city_92_sãopaulo_3p5"
#   "city_35_san_francisco_3p5"
#   "city_10_florida_villa_7gp_1758095156175"
#   "city_19_oklahoma_3p5_s"
#   "city_74_chiyoda_3p5"
# )
# NOISE_PROBS=(0.1 0.2 0.3 0.4 0.5)
TRAIN_RATIO=(0.2 0.4 0.6)


# for scenario in "${SCENARIOS[@]}"; do
#   base="playground_std_$(echo "$scenario" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/playground_std.py \
#     "$scenario" \
#     --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#     --checkpoint-dir "/home/blessedg/Pathformer/checkpoints_std" \
#     > "/home/blessedg/Pathformer/logs/${base}.out" \
#     2> "/home/blessedg/Pathformer/logs/${base}.err" &
# done


# for scenario in "${SCENARIOS[@]}"; do
#   base="multiscenario_direct_$(echo "$scenario" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training.py \
#   --scenario "$scenario" \
#   --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#   > "/home/blessedg/Pathformer/logs/${base}.out" \
#   2> "/home/blessedg/Pathformer/logs/${base}.err" &
# done

# wait


# for scenario in "${SCENARIOS[@]}"; do
#   base="first_step_$(echo "$scenario" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual.py \
#   --scenario "$scenario" \
#   --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#   --checkpoint-dir "/home/blessedg/Pathformer/checkpoints_first_step_residual" \
#    --noise-prob 0.2 \
#   > "/home/blessedg/Pathformer/logs/${base}.out" \
#   2> "/home/blessedg/Pathformer/logs/${base}.err" &
# done
# wait
# for scenario in "${SCENARIOS[@]}"; do
#   base="corridor_$(echo "$scenario" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor.py \
#   --scenario "$scenario" \
#   --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#   --checkpoint-dir "/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor" \
#    --noise-prob 0.2 \
#   > "/home/blessedg/Pathformer/logs/${base}.out" \
#   2> "/home/blessedg/Pathformer/logs/${base}.err" &
# done

# wait


# echo "All ${#SCENARIOS[@]} playground_std runs finished."



# for train_ratio in "${TRAIN_RATIO[@]}"; do
#   base="direct_train_ratio_city_23_beijing_3p5_ablate_2$(echo "$train_ratio" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training_ablate.py \
#     --scenario city_23_beijing_3p5 \
#     --checkpoint-dir /home/blessedg/Pathformer/direct_train_ratio_2 \
#     --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#     --noise-prob 0.2 \
#     --train-ratio "$train_ratio" \
#     > "/home/blessedg/Pathformer/logs/${base}.out" \
#     2> "/home/blessedg/Pathformer/logs/${base}.err" &
# done

# for train_ratio in "${TRAIN_RATIO[@]}"; do
#   base="first_step_ablate_city_23_beijing_3p5_ablate_2$(echo "$train_ratio" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_ablate.py \
#     --scenario city_23_beijing_3p5 \
#     --checkpoint-dir /home/blessedg/Pathformer/first_step_ablate_2 \
#     --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#     --noise-prob 0.2 \
#     --train-ratio "$train_ratio" \
#     > "/home/blessedg/Pathformer/logs/${base}.out" \
#     2> "/home/blessedg/Pathformer/logs/${base}.err" &
# done

# # # Optional third block
# for train_ratio in "${TRAIN_RATIO[@]}"; do
#   base="corridor_ablate_city_23_beijing_3p5_ablate_2$(echo "$train_ratio" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor_ablate.py \
#     --scenario city_23_beijing_3p5 \
#     --checkpoint-dir /home/blessedg/Pathformer/corridor_train_ratio_2 \
#     --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#     --noise-prob 0.2 \
#     --train-ratio "$train_ratio" \
#     > "/home/blessedg/Pathformer/logs/${base}.out" \
#     2> "/home/blessedg/Pathformer/logs/${base}.err" &
# done

# wait
# echo "All training runs finished."


# echo "All ${#SCENARIOS[@]} playground_std runs finished."



# python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual.py \
#   --scenario city_0_newyork_3p5_s \
#   --scenario city_1_losangeles_3p5 \
#   --scenario city_2_chicago_3p5 \
#   --scenario city_3_houston_3p5 \
#   --scenario city_4_phoenix_3p5 \
#   --scenario city_5_philadelphia_3p5 \
#   --scenario city_6_miami_3p5 \
#   --scenario city_7_sandiego_3p5 \
#   --scenario city_8_dallas_3p5 \
#   --scenario city_9_sanfrancisco_3p5 \
#   --scenario city_10_austin_3p5 \
#   --scenario city_11_santaclara_3p5 \
#   --scenario city_12_fortworth_3p5 \
#   --scenario city_13_columbus_3p5 \
#   --scenario city_17_seattle_3p5_s \
#   --scenario city_18_denver_3p5 \
#   --scenario city_19_oklahoma_3p5_s \
#   --scenario city_16_sanfrancisco_3p5_lwm \
#   --scenario city_23_beijing_3p5 \
#   --scenario city_31_barcelona_3p5 \
#   --scenario city_35_san_francisco_3p5 \
#   --scenario city_47_chicago_3p5 \
#   --scenario city_89_nairobi_3p5 \
#   --scenario city_91_xiangyang_3p5 \
#   --scenario city_92_sãopaulo_3p5 \
#   --scenario boston5g_3p5 \
#   --scenario city_86_ankara_3p5 \
#   --scenario city_86_ankara_3p5 \
#   --scenario city_72_capetown_3p5 \
#   --scenario city_84_baoding_3p5 \
#   --scenario city_95_delhi_3p5 \
#   --scenario city_96_osaka_3p5 \
#   --scenario city_88_tongshan_3p5 \
#   --noise-prob 0.2 \
#   --skip-train \
#   --csv-log-file /home/blessedg/Pathformer/logs/first_residual_results.csv

# for train_ratio in "${TRAIN_RATIO[@]}"; do
#   base="direct_train_ratio_city_23_beijing_3p5_ablate$(echo "$train_ratio" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training_ablate.py \
#     --scenario city_23_beijing_3p5 \
#     --checkpoint-dir "/home/blessedg/Pathformer/direct_train_ratio" \
#     --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#     --noise-prob 0.2 \
#     --train-ratio "$train_ratio" 
# done
# wait

# python /home/blessedg/Pathformer/multiscenario_direct_training.py \
#   --scenario city_0_newyork_3p5_s \
#   --scenario city_1_losangeles_3p5 \
#   --scenario city_2_chicago_3p5 \
#   --scenario city_3_houston_3p5 \
#   --scenario city_4_phoenix_3p5 \
#   --scenario city_5_philadelphia_3p5 \
#   --scenario city_6_miami_3p5 \
#   --scenario city_7_sandiego_3p5 \
#   --scenario city_8_dallas_3p5 \
#   --scenario city_9_sanfrancisco_3p5 \
#   --scenario city_10_austin_3p5 \
#   --scenario city_11_santaclara_3p5 \
#   --scenario city_12_fortworth_3p5 \
#   --scenario city_13_columbus_3p5 \
#   --scenario city_17_seattle_3p5_s \
#   --scenario city_18_denver_3p5 \
#   --scenario city_19_oklahoma_3p5_s \
#   --scenario city_16_sanfrancisco_3p5_lwm \
#   --scenario city_23_beijing_3p5 \
#   --scenario city_31_barcelona_3p5 \
#   --scenario city_35_san_francisco_3p5 \
#   --scenario city_47_chicago_3p5 \
#   --scenario city_89_nairobi_3p5 \
#   --scenario city_91_xiangyang_3p5 \
#   --scenario city_92_sãopaulo_3p5 \
#   --scenario boston5g_3p5 \
#   --scenario city_86_ankara_3p5 \
#   --scenario city_86_ankara_3p5 \
#   --scenario city_72_capetown_3p5 \
#   --scenario city_84_baoding_3p5 \
#   --scenario city_95_delhi_3p5 \
#   --scenario city_96_osaka_3p5 \
#   --scenario city_88_tongshan_3p5 \
#   --noise-prob 0.2 \
#   --skip-train \
#   --csv-log-file /home/blessedg/Pathformer/logs/direct_results.csv



# python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor.py \
#   --scenario city_0_newyork_3p5_s \
#   --scenario city_1_losangeles_3p5 \
#   --scenario city_2_chicago_3p5 \
#   --scenario city_3_houston_3p5 \
#   --scenario city_4_phoenix_3p5 \
#   --scenario city_5_philadelphia_3p5 \
#   --scenario city_6_miami_3p5 \
#   --scenario city_7_sandiego_3p5 \
#   --scenario city_8_dallas_3p5 \
#   --scenario city_9_sanfrancisco_3p5 \
#   --scenario city_10_austin_3p5 \
#   --scenario city_11_santaclara_3p5 \
#   --scenario city_12_fortworth_3p5 \
#   --scenario city_13_columbus_3p5 \
#   --scenario city_17_seattle_3p5_s \
#   --scenario city_18_denver_3p5 \
#   --scenario city_19_oklahoma_3p5_s \
#   --scenario city_16_sanfrancisco_3p5_lwm \
#   --scenario city_23_beijing_3p5 \
#   --scenario city_31_barcelona_3p5 \
#   --scenario city_35_san_francisco_3p5 \
#   --scenario city_47_chicago_3p5 \
#   --scenario city_89_nairobi_3p5 \
#   --scenario city_91_xiangyang_3p5 \
#   --scenario city_92_sãopaulo_3p5 \
#   --scenario boston5g_3p5 \
#   --scenario city_86_ankara_3p5 \
#   --scenario city_72_capetown_3p5 \
#   --scenario city_84_baoding_3p5 \
#   --scenario city_95_delhi_3p5 \
#   --scenario city_96_osaka_3p5 \
#   --scenario city_88_tongshan_3p5 \
#   --noise-prob 0.2 \
#   --skip-train \
#   --csv-log-file /home/blessedg/Pathformer/logs/corridor_results.csv

# python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor_concat.py \
#   --scenario city_0_newyork_3p5_s \
#   --scenario city_1_losangeles_3p5 \
#   --scenario city_2_chicago_3p5 \
#   --scenario city_3_houston_3p5 \
#   --scenario city_4_phoenix_3p5 \
#   --scenario city_5_philadelphia_3p5 \
#   --scenario city_6_miami_3p5 \
#   --scenario city_7_sandiego_3p5 \
#   --scenario city_8_dallas_3p5 \
#   --scenario city_9_sanfrancisco_3p5 \
#   --scenario city_10_austin_3p5 \
#   --scenario city_11_santaclara_3p5 \
#   --scenario city_12_fortworth_3p5 \
#   --scenario city_13_columbus_3p5 \
#   --scenario city_17_seattle_3p5_s \
#   --scenario city_18_denver_3p5 \
#   --scenario city_19_oklahoma_3p5_s \
#   --scenario city_16_sanfrancisco_3p5_lwm \
#   --scenario city_23_beijing_3p5 \
#   --scenario city_31_barcelona_3p5 \
#   --scenario city_35_san_francisco_3p5 \
#   --scenario city_47_chicago_3p5 \
#   --scenario city_89_nairobi_3p5 \
#   --scenario city_91_xiangyang_3p5 \
#   --scenario city_92_sãopaulo_3p5 \
#   --scenario boston5g_3p5 \
#   --scenario city_86_ankara_3p5 \
#   --csv-log-file /home/blessedg/Pathformer/logs/first_step_residual_corridor_concat_all.csv \
#   --checkpoint-dir /home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat \
#   --noise-prob 0.2


  # --scenario city_72_capetown_3p5 \
  # --scenario city_84_baoding_3p5 \
  # --scenario city_95_delhi_3p5 \
  # --scenario city_96_osaka_3p5 \
  # --scenario city_88_tongshan_3p5 
# SCENARIOS=(
# 'city_96_osaka_3p5'
# 'city_88_tongshan_3p5'
# 'city_0_newyork_3p5_s'
# 'city_47_chicago_3p5'
# 'city_23_beijing_3p5'
# )



# for scenario in "${SCENARIOS[@]}"; do
#   base="first_step_residual_corridor_finetune$(echo "$scenario" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor_finetune.py \
#     --scenario "$scenario" \
#     --pretrained-checkpoint /home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat/first_step_residual_corridor_concat_27scenarios_44710a4a_latest_checkpoint.pth \
#     --skip-train \
#     --csv-log-file /home/blessedg/Pathformer/logs/eval_first_step_residual_corridor_finetune_results_$scenario.csv \
#     > "/home/blessedg/Pathformer/logs/${base}.out" \
#     2> "/home/blessedg/Pathformer/logs/${base}.err" &
#   done
# wait


# python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor_concat.py \
#   --scenario city_0_newyork_3p5_s \
#   --scenario city_1_losangeles_3p5 \
#   --scenario city_2_chicago_3p5 \
#   --scenario city_3_houston_3p5 \
#   --scenario city_4_phoenix_3p5 \
#   --scenario city_5_philadelphia_3p5 \
#   --scenario city_6_miami_3p5 \
#   --scenario city_7_sandiego_3p5 \
#   --scenario city_8_dallas_3p5 \
#   --scenario city_9_sanfrancisco_3p5 \
#   --scenario city_10_austin_3p5 \
#   --scenario city_11_santaclara_3p5 \
#   --scenario city_12_fortworth_3p5 \
#   --scenario city_13_columbus_3p5 \
#   --scenario city_17_seattle_3p5_s \
#   --scenario city_18_denver_3p5 \
#   --scenario city_19_oklahoma_3p5_s \
#   --scenario city_16_sanfrancisco_3p5_lwm \
#   --scenario city_23_beijing_3p5 \
#   --scenario city_31_barcelona_3p5 \
#   --scenario city_35_san_francisco_3p5 \
#   --scenario city_47_chicago_3p5 \
#   --scenario city_89_nairobi_3p5 \
#   --scenario city_91_xiangyang_3p5 \
#   --scenario city_92_sãopaulo_3p5 \
#   --scenario boston5g_3p5 \
#   --scenario city_86_ankara_3p5 \
#   --csv-log-file /home/blessedg/Pathformer/logs/first_step_residual_corridor_concat_all.csv \
#   --checkpoint-dir /home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat \
#   --noise-prob 0.2
