#!/bin/bash
set -euo pipefail
mkdir -p /shared/home/raghavan/contributions/kvpress/logs

COMMON="--partition=gpu --gres=gpu:1 --cpus-per-task=12 --mem=128G --time=04:00:00"
EVAL_DIR="/shared/home/raghavan/contributions/kvpress/evaluation"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR="./results/zigzag_bench"
PREAMBLE='source /shared/home/raghavan/miniconda3/etc/profile.d/conda.sh && conda activate kvpress && cd /shared/home/raghavan/contributions/kvpress/evaluation'

declare -a JOBS=(
  "ruler_baseline:no_press:0.0:ruler"
  "ruler_zigzag50:zigzag_observed:0.5:ruler"
  "ruler_zigzag80:zigzag_observed:0.8:ruler"
  "ruler_snapkv50:snapkv:0.5:ruler"
  "ruler_knorm50:knorm:0.5:ruler"
  "ruler_obsattn50:observed_attention:0.5:ruler"
  "lb_baseline:no_press:0.0:longbench"
  "lb_zigzag50:zigzag_observed:0.5:longbench"
  "lb_zigzag80:zigzag_observed:0.8:longbench"
  "lb_snapkv50:snapkv:0.5:longbench"
)

for entry in "${JOBS[@]}"; do
  IFS=: read -r name press ratio dataset <<< "$entry"
  sbatch $COMMON \
    --job-name="zz-${name}" \
    --output="/shared/home/raghavan/contributions/kvpress/logs/${name}_%j.out" \
    --error="/shared/home/raghavan/contributions/kvpress/logs/${name}_%j.err" \
    --wrap="${PREAMBLE} && python evaluate.py --dataset ${dataset} --model ${MODEL} --press_name ${press} --compression_ratio ${ratio} --output_dir ${OUT_DIR}"
  echo "Submitted: ${name} (${dataset}, ${press}, ratio=${ratio})"
done
