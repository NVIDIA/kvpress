dataset="longbench"
data_dir="narrativeqa"
model="Qwen/Qwen2.5-7B-Instruct"
compression_ratios=(0.5)
press_names=("finch" "finch_with_fix")

# Check if the number of press names is less than or equal to the number of available GPUs
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ ${#press_names[@]} -gt $num_gpus ]; then
  echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
  exit 1
fi

# Iterate over press names and compression ratios
for i in "${!press_names[@]}"; do
  press="${press_names[$i]}"
  
  # Run each press_name on a different GPU in the background
  (
    for compression_ratio in "${compression_ratios[@]}"; do
      echo "Running press_name: $press with compression_ratio: $compression_ratio on GPU cuda:$i"
      python evaluate.py --dataset $dataset --data_dir $data_dir --model $model --press_name $press --compression_ratio $compression_ratio --apply_yarn --compress_questions --device "cuda:$i"
    done
  ) &
done

# Wait for all background jobs to finish
wait
echo "All evaluations completed."
