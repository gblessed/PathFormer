#!/bin/bash
#SBATCH --job-name=train-ratio-seeds
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:1

source ~/.bashrc
conda activate pathformer

cd /home/blessedg/Pathformer

bash /home/blessedg/Pathformer/run_train_ratio_seed_sweep_city23.sh
