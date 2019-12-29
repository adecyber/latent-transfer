#!/bin/sh
#SBATCH --job-name=finetune_check
#SBATCH --time=96:00:00
#SBATCH --partition=napoli-gpu --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

python latent.py --root_dir "./finetune_forwards_3mil" --finetune
