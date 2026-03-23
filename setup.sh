#!/bin/bash

module load gcc/11.2.0
module load cuda/12.4.1

# pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install --no-build-isolation flash-attn==2.7.4.post1

pip uninstall -y vllm
pip install -U --no-cache-dir vllm
python -c "import vllm._C; print('vllm _C ok')"