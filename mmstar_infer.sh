#!/bin/bash
#SBATCH --job-name=mmstar_pipeline
#SBATCH --output=logs/mmstar_%x_%j.log
#SBATCH --error=logs/mmstar_%x_%j.log
#SBATCH --time=24:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail

MODEL_NAME="${1:?Need MODEL_NAME}"

mkdir -p logs

echo "===== START ${MODEL_NAME} ====="

# env
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate qwen
source /etc/profile.d/modules.sh
module add cuda/12.4.1

CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export HF_HOME=${CACHE_DIR}
export HF_DATASETS_CACHE=${CACHE_DIR}
export HF_MODULES_CACHE=${CACHE_DIR}
export TRANSFORMERS_CACHE=${CACHE_DIR}

source /fs/nexus-scratch/yliang17/Research/VLA/config/key.conf

# ===== 1. MMStar infer =====
cd /fs/nexus-scratch/yliang17/Research/VLM/VLMEvalKit/
echo "[1/2] Running MMStar infer for ${MODEL_NAME}"
python run.py --data MMStar --model "${MODEL_NAME}" --mode infer

# ===== 2. GPT eval =====
cd /fs/nexus-scratch/yliang17/Research/VLM/VLM_toolset/
echo "[2/2] Running GPT eval for ${MODEL_NAME}"
python3 eval/mmstar/call_gpt_batch.py \
    --model_name="${MODEL_NAME}" \
    --model="gpt-5-nano"

echo "===== DONE ${MODEL_NAME} ====="