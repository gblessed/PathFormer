#!/bin/bash
#SBATCH --job-name=delay_only_delta_noise             # Job name
#SBATCH --output=logs/job_%j.out       # Output log (%j = job ID)
#SBATCH --error=logs/job_%j.err        # Error log (%j = job ID)
#SBATCH --time=70:00:00                # Max runtime (HH:MM:SS)
#SBATCH --gres=gpu:tesla:1              # Request 1 H100 GPU
source ~/.bashrc

# Load environment if needed
conda activate pathformer

# Run your training script
python3 playground.py
# python3 multiscenario_direct_training.py
# python3 zero_pre_train_finetune_beam_prediction.py
# python3 pretrain_finetune_user_localization.py

# python3 deep_dive_decoder_inter_str_lists.py
