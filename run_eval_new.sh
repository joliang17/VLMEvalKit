#!/bin/bash

#SBATCH --job-name=new_eval_2ds
#SBATCH --output=new_eval_2ds.log
#SBATCH --error=new_eval_2ds.log
#SBATCH --time=24:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1

export HF_HOME="/fs/nexus-projects/wilddiffusion/cache"
export HF_DATASETS_CACHE="/fs/nexus-projects/wilddiffusion/cache"
export HF_MODULES_CACHE="/fs/nexus-projects/wilddiffusion/cache"
export TRANSFORMERS_CACHE="/fs/nexus-projects/wilddiffusion/cache"

source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf


python run.py --data Creation_MMBench LogicVista --model Qwen3-VL-8B-qinstruct_new_qa_4k --mode eval --verbose
python run.py --data Creation_MMBench LogicVista --model Qwen3-VL-8B-qinstruct_new_aq_4k --mode eval --verbose
python run.py --data Creation_MMBench LogicVista --model Qwen3-VL-8B-qinstruct_new_iqa_4k --mode eval --verbose
