#!/bin/bash
#SBATCH --job-name=kvpress-tests
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err

set -euo pipefail
mkdir -p logs

source /shared/home/raghavan/miniconda3/etc/profile.d/conda.sh
conda activate kvpress

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "=== Running ZigZagKVPress unit tests ==="
cd /shared/home/raghavan/contributions/kvpress
python -m pytest tests/test_zigzag_press.py -v --tb=long

echo ""
echo "=== Running all press tests ==="
python -m pytest tests/ -v --tb=short -x --ignore=tests/integration

echo ""
echo "=== All tests passed ==="
