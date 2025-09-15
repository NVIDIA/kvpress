dataset="math500"
model="nvidia/OpenMath-Nemotron-14B" 
target_sizes=(500)
press_names=("decoding_knorm" "decoding_adakv_expected_attention_e2")

# 1. Define your list of GPUs
gpus=("cuda:0" "cuda:7")

# 2. Check if the number of press names is less than or equal to the number of provided GPUs
if [ ${#press_names[@]} -gt ${#gpus[@]} ]; then
  echo "Error: The number of press names (${#press_names[@]}) exceeds the number of provided GPUs (${#gpus[@]})"
  exit 1
fi

# 3. Iterate over press names and assign each to a GPU from your list
for i in "${!press_names[@]}"; do
  press="${press_names[$i]}"
  gpu="${gpus[$i]}"

  # Run each press_name on a different GPU in the background
  (
    for target_size in "${target_sizes[@]}"; do
      echo "Running press_name: $press with target_size: $target_size on GPU $gpu"
      python evaluate.py --dataset $dataset --model $model --press_name $press --target_size $target_size --device "$gpu" --max_new_tokens 4096
    done
  ) &
done

# Wait for all background jobs to finish
wait
echo "All evaluations completed."