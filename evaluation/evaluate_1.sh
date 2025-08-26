dataset="needle_in_haystack"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
compression_ratios=(0.25)
press_names=("adakv_expected_attention_e2" "tova" "no_press" "snapkv")
max_context_lengths=(8000 16000 48000 96000 192000)
# Define specific GPUs to use
gpus=(0 1 2 3)

# Check if the number of press names is less than or equal to the number of available GPUs
if [ ${#press_names[@]} -gt ${#gpus[@]} ]; then
  echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs (${#gpus[@]})"
  exit 1
fi

# Iterate over press names - each press gets assigned to one GPU
for i in "${!press_names[@]}"; do
  press="${press_names[$i]}"
  gpu_id="${gpus[$i]}"
  
  # Run this press_name on the assigned GPU for all data_dirs and compression_ratios
  (
    echo "GPU $gpu_id: Starting press_name: $press"
    for max_context_length in "${max_context_lengths[@]}"; do
      for compression_ratio in "${compression_ratios[@]}"; do
        echo "GPU $gpu_id: Running press_name: $press with max_context_length: $max_context_length and compression_ratio: $compression_ratio"
        python evaluate.py --dataset $dataset  --model $model --press_name $press --compression_ratio $compression_ratio --device "cuda:$gpu_id" --max_context_length $max_context_length --output_dir "./results/niah_llama/"
      done
    done
    echo "GPU $gpu_id: Completed press_name: $press"
  ) &
done

# Wait for all background jobs to finish
wait
echo "All evaluations completed."
