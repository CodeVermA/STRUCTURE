#!/bin/bash
#SBATCH --job-name=train-normal
#SBATCH --output=/home/s2412780/MLP/logs/train_normal_%j.out
#SBATCH --error=/home/s2412780/MLP/logs/train_normal_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Activate environment
cd /home/s2412780/MLP/STRUCTURE
source /home/s2412780/MLP/struct.sh

# 1. Tell Hugging Face to be offline
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 2. Tell Weights & Biases to be offline!
export WANDB_MODE=offline

export HF_HOME=/home/s2412780/.cache/huggingface

# 3. Run the training command
python -m src.train_alignment \
  --config_path configs/losses_lin/clip_structure_1.yaml
