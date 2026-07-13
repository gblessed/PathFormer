#!/bin/bash
#SBATCH --job-name=wifo-gen-all
#SBATCH --output=/home/blessedg/Pathformer/logs/wifo_channel_%A_%a.out
#SBATCH --error=/home/blessedg/Pathformer/logs/wifo_channel_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:nvidia:1

source ~/.bashrc
conda activate wifo
set -euo pipefail


cd /home/blessedg/Pathformer/WiFo/src/
# python generate_wifo_data.py

python main.py --my_data True --dataset all --device_id 0 --size base --mask_strategy_random none --mask_strategy temporal --file_load_path ./weights/wifo_base --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D