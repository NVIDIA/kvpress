# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to run the leaderboard evaluation on 4 GPUs
dataset="ruler"
data_dir="4096"
model="Qwen/Qwen3-8B"
output_dir="./results_lb"

# Loop 1: presses not requiring to include the questions in the compression
press_names=("random" "knorm" "snapkv" "expected_attention" "streaming_llm" "tova" "observed_attention" "qfilter" "pyramidkv" "lagkv" "keydiff" "adakv_compactor" "cur" "duo_attention" "duo_attention_on_the_fly" "kvzip")

python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name no_press --compression_ratio 0.00 --output_dir $output_dir --device "cuda:0"

for press in "${press_names[@]}"; do  
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio 0.25  --output_dir $output_dir --device "cuda:0" &
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio 0.50  --output_dir $output_dir --device "cuda:1" &
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio 0.75  --output_dir $output_dir --device "cuda:2" &
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio 0.875 --output_dir $output_dir --device "cuda:3" &
    wait
done

# Use -3, -4, -5, -6 for Qwen3-8B and -6, -7, -8, -9 for Llama-3.1-8B-Instruct
for press in "kvzap_linear" "kvzap_mlp"; do
  python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --threshold -3  --output_dir $output_dir --device "cuda:0" &
  python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --threshold -4  --output_dir $output_dir --device "cuda:1" &
  python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --threshold -5  --output_dir $output_dir --device "cuda:2" &
  python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --threshold -6  --output_dir $output_dir --device "cuda:3" &
  wait
done

# Loop 2: presses requiring to compress questions
press_names=("snapkv" "adakv_snapkv" "finch" "chunkkv")
for press in "${press_names[@]}"; do  
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio 0.25  --output_dir $output_dir --device "cuda:0" --query_aware &
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio 0.50  --output_dir $output_dir --device "cuda:1" --query_aware &
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio 0.75  --output_dir $output_dir --device "cuda:2" --query_aware &
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio 0.875 --output_dir $output_dir --device "cuda:3" --query_aware &
    wait
done
