#!/bin/bash
#SBATCH --job-name=channel-eval
#SBATCH --output=/home/blessedg/Pathformer/logs/channel_eval_%j.out
#SBATCH --error=/home/blessedg/Pathformer/logs/channel_eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1

source ~/.bashrc
set -euo pipefail
conda activate pathformer

bash /home/blessedg/Pathformer/run_channel_evaluation_all.sh
