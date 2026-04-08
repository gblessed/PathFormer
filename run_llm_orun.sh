#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=70:00:00
#SBATCH --gres=gpu:h200:1
source ~/.bashrc

conda activate pathformer

# Scenarios to run in parallel on this single GPU (passed as CLI args to multiscenario_direct_training_parallel.py)


# SCENARIOS=(
#     "city_47_chicago_3p5"
#     "city_91_xiangyang_3p5"
#     "city_23_beijing_3p5"
#     "city_12_fortworth_3p5"
#     "city_17_seattle_3p5_s"
# )
# SCENARIOS=(
#   "city_92_sãopaulo_3p5"
#   "city_35_san_francisco_3p5"
#   "city_10_florida_villa_7gp_1758095156175"
#   "city_19_oklahoma_3p5_s"
#   "city_74_chiyoda_3p5"
# )


# mkdir -p logs
# for i in "${!SCENARIOS[@]}"; do
#   s="${SCENARIOS[$i]}"
#   # Log file name: strip non-alnum for filename (e.g. city_92_saopaulo_3p5)
#   base="parallel_$(echo "$s" | sed 's/[^a-zA-Z0-9_]/_/g')"
#   python3 playground.py "$s" > "logs/${base}.out" 2> "logs/${base}.err" &
# done
# wait
# echo "All ${#SCENARIOS[@]} scenario trainings finished."
# python /home/blessedg/Pathformer/playground_std_fix1_cluster_sweep.py
python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual.py --n-clusters 50
# python pretrain_mixed_cluster.py --n_clusters 15  > "/home/blessedg/Pathformer/logs/pre_train_15.out" 2> "/home/blessedg/Pathformer/logs/pre_train_15.err" & python pretrain_mixed_cluster.py --n_clusters 5 > "/home/blessedg/Pathformer/logs/pre_train_5.out" 2> "/home/blessedg/Pathformer/logs/pre_train_5.err"
wait
# # python zero_pre_train_finetune_beam_prediction.py
# python rl_channel_ppo.py --checkpoint weighted_noise_enc_direct_city_47_chicago_3p5_interacaction_all_inter_str_dec_all_repeat --scenario city_47_chicago_3p5 --lr 5e-6 --init_log_std -3.0 --algorithm grpo --group_size 4
# python channel_finetune_direct_training.py --path_checkpoint /home/blessedg/Pathformer/checkpoints2/true_enc_direct_city_35_san_francisco_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth --eval_only


# source /home/blessedg/miniconda3/etc/profile.d/conda.sh
# conda activate pathformer
python /home/blessedg/Pathformer/multiscenario_direct_training_first_step_residual_corridor.py \
  --scenario city_47_chicago_3p5 \
  --n-clusters 25 \
  --nearest-k 5 \
  --corridor-k 5 \
  --corridor-bins 8 \
  --csv-log-file /home/blessedg/Pathformer/muldims_weighted_first_step_residual_corridor_results.csv \
  --checkpoint-dir /home/blessedg/Pathformer/checkpoints_first_step_residual_corridor
