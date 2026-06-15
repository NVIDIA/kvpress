#!/bin/bash
#SBATCH --job-name=kvpress-zigzag-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -euo pipefail
mkdir -p logs

source /shared/home/raghavan/miniconda3/etc/profile.d/conda.sh
conda activate kvpress

echo "=== GPU Info ==="
nvidia-smi
echo ""

cd /shared/home/raghavan/contributions/kvpress/evaluation

MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "============================================"
echo "  ZigZagKV Benchmarks on ${MODEL}"
echo "============================================"
echo ""

# --- RULER Benchmark ---
echo ">>> [1/6] RULER - no_press (baseline)"
python evaluate.py --dataset ruler --model "$MODEL" --press_name no_press --compression_ratio 0.0 --output_dir ./results/zigzag_bench

echo ">>> [2/6] RULER - zigzag_observed (0.5 compression)"
python evaluate.py --dataset ruler --model "$MODEL" --press_name zigzag_observed --compression_ratio 0.5 --output_dir ./results/zigzag_bench

echo ">>> [3/6] RULER - zigzag_observed (0.8 compression)"
python evaluate.py --dataset ruler --model "$MODEL" --press_name zigzag_observed --compression_ratio 0.8 --output_dir ./results/zigzag_bench

# --- Compare with SnapKV and KnormPress at the same ratios ---
echo ">>> [4/6] RULER - snapkv (0.5 compression)"
python evaluate.py --dataset ruler --model "$MODEL" --press_name snapkv --compression_ratio 0.5 --output_dir ./results/zigzag_bench

echo ">>> [5/6] RULER - knorm (0.5 compression)"
python evaluate.py --dataset ruler --model "$MODEL" --press_name knorm --compression_ratio 0.5 --output_dir ./results/zigzag_bench

echo ">>> [6/6] RULER - observed_attention (0.5 compression)"
python evaluate.py --dataset ruler --model "$MODEL" --press_name observed_attention --compression_ratio 0.5 --output_dir ./results/zigzag_bench

echo ""
echo "============================================"
echo "  LongBench Benchmarks"
echo "============================================"

echo ">>> [7/10] LongBench - no_press (baseline)"
python evaluate.py --dataset longbench --model "$MODEL" --press_name no_press --compression_ratio 0.0 --output_dir ./results/zigzag_bench

echo ">>> [8/10] LongBench - zigzag_observed (0.5)"
python evaluate.py --dataset longbench --model "$MODEL" --press_name zigzag_observed --compression_ratio 0.5 --output_dir ./results/zigzag_bench

echo ">>> [9/10] LongBench - zigzag_observed (0.8)"
python evaluate.py --dataset longbench --model "$MODEL" --press_name zigzag_observed --compression_ratio 0.8 --output_dir ./results/zigzag_bench

echo ">>> [10/10] LongBench - snapkv (0.5)"
python evaluate.py --dataset longbench --model "$MODEL" --press_name snapkv --compression_ratio 0.5 --output_dir ./results/zigzag_bench

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "  Results in: evaluation/results/zigzag_bench/"
echo "============================================"
