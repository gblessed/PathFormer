#!/bin/bash
#SBATCH --job-name=qwen_orun              # Job name
#SBATCH --output=logs/job_%j.out       # Output log (%j = job ID)
#SBATCH --error=logs/job_%j.err        # Error log (%j = job ID)
#SBATCH --time=05:00:00                # Max runtime (HH:MM:SS)
#SBATCH --gres=gpu:T4:1              # Request 1 H100 GPU
source ~/.bashrc

# Load environment if needed
conda activate pathformer

# Run your training script
python3 pretrain_finetune.py