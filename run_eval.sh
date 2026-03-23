#!/bin/bash

#SBATCH --job-name=ori_eval
#SBATCH --output=ori_eval.log
#SBATCH --error=ori_eval.log
#SBATCH --time=24:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cml31
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

# MMStar

python run.py --data Creation_MMBench LogicVista --model Qwen3-VL-8B-qinstruct_ori_qa_4k --verbose
python run.py --data Creation_MMBench LogicVista --model Qwen3-VL-8B-qinstruct_ori_aq_4k --verbose
python run.py --data Creation_MMBench LogicVista --model Qwen3-VL-8B-qinstruct_ori_iqa_4k --verbose

# python run.py --data MMStar --model Qwen2.5-VL-7B-qinstruct_NEW_qapair --verbose
# python run.py --data MMStar --model Qwen2.5-VL-7B-qinstruct_ORI_qs --verbose

# python run.py --data MMMU_Pro_V --model Qwen2.5-VL-7B-Instruct --verbose
# python run.py --data MMMU_Pro_V --model Qwen2.5-VL-7B-qinstruct_ORI --verbose

# python run.py --data MMMU_Pro_V --model Qwen2.5-VL-7B-qinstruct_QAONLY --verbose
# python run.py --data MMMU_Pro_V --model Qwen2.5-VL-7B-qinstruct_NEW --verbose
