#!/bin/bash
#SBATCH --job-name=corridor-missing
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h200:1

cd /home/blessedg/Pathformer
bash /home/blessedg/Pathformer/run_corridor_missing_06_08.sh
