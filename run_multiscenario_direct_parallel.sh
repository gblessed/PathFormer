#!/bin/bash
#SBATCH --job-name=channel
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=70:00:00
#SBATCH --gres=gpu:tesla:1

source ~/.bashrc
conda activate pathformer

# python /home/blessedg/Pathformer/channel_finetune_embedding_all_families.py \
#   --epochs 50 \
#   --batch-size 128 \
#   --embed-pool mean \
#   --csv-log-file /home/blessedg/Pathformer/logs/channel_finetune_embedding_all_families.csv
python /home/blessedg/Pathformer/zero_shot_channel_estimation_all_families.py --phase-source groundtruth --csv-log-file "/home/blessedg/Pathformer/logs/zero_shot_gt_phase_channel_estimation_all_families.csv"

# mkdir -p logs
# mkdir -p checkpoints2


# SCENARIOS=(
#   "city_47_chicago_3p5"
#   "city_23_beijing_3p5"
#   "city_91_xiangyang_3p5"
#   "city_17_seattle_3p5_s"
#   "city_12_fortworth_3p5"
# )
# SCENARIOS=(
#   "city_92_sãopaulo_3p5"
#   "city_35_san_francisco_3p5"
#   "city_10_florida_villa_7gp_1758095156175"
#   "city_19_oklahoma_3p5_s"
#   "city_74_chiyoda_3p5"
# )

# for scenario in "${SCENARIOS[@]}"; do
#   base="orig_parallel_$(echo "$scenario" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python /home/blessedg/Pathformer/multiscenario_direct_training.py \
#     "$scenario" \
#     --csv-log-file "/home/blessedg/Pathformer/logs/${base}.csv" \
#     --checkpoint-dir "/home/blessedg/Pathformer/checkpoints2" \
#     > "/home/blessedg/Pathformer/logs/${base}.out" \
#     2> "/home/blessedg/Pathformer/logs/${base}.err" &
# done

# wait
# echo "All ${#SCENARIOS[@]} multiscenario runs finished."
