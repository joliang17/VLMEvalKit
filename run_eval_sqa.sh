#!/bin/bash

#SBATCH --array=0-3
#SBATCH --job-name=sqa_eval
#SBATCH --output=log/sqa_eval_%A_%a.log
#SBATCH --error=log/sqa_eval_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1

export HF_HOME="/fs/nexus-projects/wilddiffusion/cache"
export HF_DATASETS_CACHE="/fs/nexus-projects/wilddiffusion/cache"
export HF_MODULES_CACHE="/fs/nexus-projects/wilddiffusion/cache"
export TRANSFORMERS_CACHE="/fs/nexus-projects/wilddiffusion/cache"
export VLM_EVAL_IMAGE_CACHE="/fs/nexus-projects/wilddiffusion/cache"
export LMUData="/fs/nexus-scratch/yliang17/Research/cache"


source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf


# python run.py --data MMStar --model Qwen2.5-VL-7B-qinstruct_NEW_qs --verbose
# python run.py --data MMStar --model Qwen2.5-VL-7B-qinstruct_NEW_qa --verbose
# python run.py --data MMStar --model Qwen2.5-VL-7B-qinstruct_NEW_qapair --verbose

MODELS=(
    "Qwen3-VL-4B-transfer_sqa_ori_mix_qa_iqa_1k_disjoint"
    "Qwen3-VL-4B-transfer_sqa_ori_qa_only_1k"
    "Qwen3-VL-4B-transfer_sqa_our_mix_qa_iqa_1k_disjoint"
    "Qwen3-VL-4B-transfer_sqa_our_qa_only_1k"
    # "Qwen3-VL-4B-Instruct"
)

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Run evaluation for ${MODEL_NAME} on ScienceQA_TEST"

python run.py --data ScienceQA_TEST --model "${MODEL_NAME}"
