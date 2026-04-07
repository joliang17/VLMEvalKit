#!/bin/bash

#SBATCH --array=0-4
#SBATCH --job-name=logistic_eval
#SBATCH --output=log/logistic_eval_%A_%a.log
#SBATCH --error=log/logistic_eval_%A_%a.log
#SBATCH --time=48:00:00
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


# python run.py --data MMStar --model Qwen2.5-VL-7B-qinstruct_NEW_qs --verbose
# python run.py --data MMStar --model Qwen2.5-VL-7B-qinstruct_NEW_qa --verbose
# python run.py --data MMStar --model Qwen2.5-VL-7B-qinstruct_NEW_qapair --verbose

MODELS=(
    # "Qwen3-VL-4B-transfer_our_mix_qa_8k"
    # "Qwen3-VL-4B-transfer_our_qa_only_8k"
    # "Qwen3-VL-4B-transfer_our_mix_qa_aq_8k_shared"
    # "Qwen3-VL-4B-transfer_our_mix_qa_iqa_8k_shared"
    # "Qwen3-VL-4B-transfer_our_mix_qa_iqa_8k_disjoint"
    # "Qwen3-VL-4B-transfer_our_mix_qa_aq_8k_disjoint"

    # "Qwen3-VL-4B-transfer_ori_mix_qa_iqa_8k_disjoint"
    # "Qwen3-VL-4B-transfer_ori_mix_qa_8k"  
    # "Qwen3-VL-4B-transfer_ori_mix_qa_iqa_8k_shared" 

    # "Qwen3-VL-4B-qinstruct_new_mix_qa_iqa_8k_067033_v2"
    # "Qwen3-VL-4B-transfer_our_mix_qa_iqa_8k_disjoint"
    # "Qwen3-VL-4B-Instruct"

    "Qwen3-VL-4B-transfer_our_v2_mix_qa_aq_8k_disjoint"
    "Qwen3-VL-4B-transfer_our_v2_mix_qa_aq_8k_shared"
    "Qwen3-VL-4B-transfer_our_v2_mix_qa_iqa_8k_disjoint"
    "Qwen3-VL-4B-transfer_our_v2_mix_qa_iqa_8k_shared"
    "Qwen3-VL-4B-transfer_our_v2_qa_full_8k"
    # "Qwen3-VL-4B-transfer_our_v2_qa_only_5360"
)

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Run evaluation for ${MODEL_NAME} on LogicVista"

python run.py --data LogicVista --model "${MODEL_NAME}"
