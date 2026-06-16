#!/bin/bash
#SBATCH --job-name=zz-longbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/raghavan/contributions/kvpress/logs/longbench_%j.out
#SBATCH --error=/shared/home/raghavan/contributions/kvpress/logs/longbench_%j.err

set -euo pipefail

source /shared/home/raghavan/miniconda3/etc/profile.d/conda.sh
conda activate kvpress
cd /shared/home/raghavan/contributions/kvpress/evaluation

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR="./results/zigzag_bench"
PRESS_NAME="${1:?Usage: sbatch run_longbench.sh <press_name> <compression_ratio>}"
RATIO="${2:?Usage: sbatch run_longbench.sh <press_name> <compression_ratio>}"

TASKS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

echo "============================================"
echo "  LongBench: ${PRESS_NAME} (ratio=${RATIO})"
echo "============================================"

for task in $TASKS; do
    echo ">>> Task: ${task}"
    python evaluate.py --config_file=none \
        --dataset longbench \
        --data_dir "${task}" \
        --model "${MODEL}" \
        --press_name "${PRESS_NAME}" \
        --compression_ratio "${RATIO}" \
        --output_dir "${OUT_DIR}" || echo "FAILED: ${task}"
done

echo "============================================"
echo "  LongBench ${PRESS_NAME} complete!"
echo "============================================"
