#!/bin/bash
#SBATCH --job-name=flash-attn-build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/flash_attn_build_%j.out
#SBATCH --error=logs/flash_attn_build_%j.err

set -euo pipefail
mkdir -p logs

source /shared/home/raghavan/miniconda3/etc/profile.d/conda.sh
conda activate kvpress

echo "=== Building flash-attn on GPU node ==="
echo "CUDA version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

pip install flash-attn --no-build-isolation 2>&1

echo ""
echo "=== Verifying flash-attn installation ==="
python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__} installed successfully')"
