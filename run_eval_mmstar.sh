#!/bin/bash
#SBATCH --array=0-4
#SBATCH --job-name=new_eval
#SBATCH --output=log/new_eval_%A_%a.log
#SBATCH --error=log/new_eval_%A_%a.log
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


MODELS=(
# "Qwen3-VL-4B-qinstruct_new_qa_12k"
# "Qwen3-VL-4B-qinstruct_ori_qa_12k"
# "Qwen3-VL-4B-qinstruct_mme_qa_12k"
# "Qwen3-VL-4B-qinstruct_new_2turn_qa_12k"

# "Qwen3-VL-4B-qinstruct_new_qa_24k"

# "Qwen3-VL-4B-qinstruct_new_qa_aq_24k"
# "Qwen3-VL-4B-qinstruct_new_qa_iqa_aq_24k"

# "Qwen3-VL-4B-qinstruct_new_qa_aq_4k"
# "Qwen3-VL-4B-qinstruct_new_qa_aq_16k"
# "Qwen3-VL-4B-qinstruct_new_qa_aq_24k"

# "Qwen3-VL-4B-qinstruct_new_qa_aq_48k"

# "Qwen3-VL-4B-qinstruct_ori_qa_24k"
# "Qwen3-VL-4B-qinstruct_halfnew_qa_24k"
# "Qwen3-VL-4B-qinstruct_halfmme_qa_24k"

# "Qwen3-VL-4B-qinstruct_new_aq_24k"
# "Qwen3-VL-4B-qinstruct_ori_qa_aq_24k"
# "Qwen3-VL-4B-qinstruct_mme_qa_aq_24k"


    # "Qwen3-VL-4B-qinstruct_ori_new_qa_8k"
    # "Qwen3-VL-4B-qinstruct_ori_new_qa_aq_8k"
    # "Qwen3-VL-4B-qinstruct_ori_new_qa_aq_iqa_8k"

    # "Qwen3-VL-4B-qinstruct_ori_qa_8k"

    # "Qwen3-VL-4B-qinstruct_new_qa_8k"
    # "Qwen3-VL-4B-qinstruct_new_qa_aq_8k"
    # "Qwen3-VL-4B-qinstruct_new_qa_aq_iqa_8k"

    # "Qwen3-VL-4B-qinstruct_new2_qa_8k"
    # "Qwen3-VL-4B-qinstruct_new2_qa_aq_8k"
    # "Qwen3-VL-4B-qinstruct_new2_qa_aq_iqa_8k"

    # "Qwen3-VL-4B-qinstruct_new_mix_qa_aq_8k"
    # "Qwen3-VL-4B-qinstruct_new_mix_qa_aq_iqa_8k"
    # "Qwen3-VL-4B-qinstruct_ori_mix_qa_aq_8k"
    # "Qwen3-VL-4B-qinstruct_ori_mix_qa_aq_iqa_8k"

    # "Qwen3-VL-4B-qinstruct_ori_qa_8k"
    # "Qwen3-VL-4B-qinstruct_ori_qa_aq_8k"
    # "Qwen3-VL-4B-qinstruct_ori_qa_aq_iqa_8k"

    # "Qwen3-VL-4B-qinstruct_new_mix_qa_aq_iqa_8k_052525"
    # "Qwen3-VL-4B-qinstruct_ori_mix_qa_aq_iqa_8k_052525"
    # "Qwen3-VL-4B-qinstruct_new_both_mix_qa_aq_iqa_8k_052525"

    "Qwen3-VL-4B-qinstruct_new_mix_qa_aq_iqa_4k_052525_v2"
    "Qwen3-VL-4B-qinstruct_new_mix_qa_aq_iqa_8k_052525_v2"
    "Qwen3-VL-4B-qinstruct_ori_mix_qa_aq_iqa_8k_052525_v2"

    "Qwen3-VL-4B-qinstruct_new_mix_qa_iqa_8k_067033_v2"
    "Qwen3-VL-4B-qinstruct_ori_mix_qa_iqa_8k_067033_v2"

)


MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
# for MODEL_NAME in "${MODELS[@]}"; do
# MODEL_NAME="Qwen3-VL-4B-qinstruct_new_qa_aq_iqa_8k"
echo "Run evaluation for ${MODEL_NAME} on MMStar"

python run.py --data MMStar --model "${MODEL_NAME}" --mode infer

# done
cd /fs/nexus-scratch/yliang17/Research/VLM/VLM_toolset/

echo "Run evaluation for ${MODEL_NAME} on MMStar"
python3 eval/mmstar/call_gpt_batch.py --model_name=${MODEL_NAME} --model="gpt-5-nano"
