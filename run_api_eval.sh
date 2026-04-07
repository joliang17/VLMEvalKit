#!/bin/bash

#SBATCH --job-name=gpt_eval
#SBATCH --output=gpt_eval.log
#SBATCH --error=gpt_eval.log
#SBATCH --time=24:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --dependency=afterok:6300972:6301020
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1

source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf

export HF_HOME="/fs/nexus-projects/wilddiffusion/cache"
export HF_DATASETS_CACHE="/fs/nexus-projects/wilddiffusion/cache"
export HF_MODULES_CACHE="/fs/nexus-projects/wilddiffusion/cache"
export TRANSFORMERS_CACHE="/fs/nexus-projects/wilddiffusion/cache"


python3 api_eval.py