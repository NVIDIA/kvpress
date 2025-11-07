#!/usr/bin/env python3
"""Test to verify the compression_ratio bug in ComposedPress"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress import ComposedPress, SnapKVPress

# Load a small model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create two presses with known compression ratios
press1 = SnapKVPress(compression_ratio=0.3)  # Remove 30%, keep 70%
press2 = SnapKVPress(compression_ratio=0.2)  # Remove 20%, keep 80%

# Expected behavior:
# - After press1: 70% of tokens remain (keep_ratio = 0.7)
# - After press2: 70% * 80% = 56% of tokens remain
# - Overall prune fraction: 1 - 0.56 = 0.44 (44% removed)

composed_press = ComposedPress([press1, press2])

# Generate some text
prompt = "The quick brown fox jumps over the lazy dog. " * 20  # Create a longer sequence
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"\nInput sequence length: {inputs['input_ids'].shape[1]}")
print(f"Press 1 compression_ratio: {press1.compression_ratio}")
print(f"Press 2 compression_ratio: {press2.compression_ratio}")

with composed_press(model):
    cache = DynamicCache()
    outputs = model(**inputs, past_key_values=cache, use_cache=True)

    final_cache_length = cache.get_seq_length()
    print(f"\nFinal cache length: {final_cache_length}")
    print(f"ComposedPress reports compression_ratio: {composed_press.compression_ratio}")

    # Calculate actual compression
    original_length = inputs['input_ids'].shape[1]
    actual_keep_ratio = final_cache_length / original_length
    actual_prune_ratio = 1 - actual_keep_ratio

    print(f"\nActual keep ratio: {actual_keep_ratio:.4f}")
    print(f"Actual prune ratio (compression_ratio): {actual_prune_ratio:.4f}")

    # Expected values
    expected_keep_ratio = (1 - 0.3) * (1 - 0.2)  # 0.7 * 0.8 = 0.56
    expected_prune_ratio = 1 - expected_keep_ratio  # 0.44

    print(f"\nExpected keep ratio: {expected_keep_ratio:.4f}")
    print(f"Expected prune ratio: {expected_prune_ratio:.4f}")

    print(f"\n{'='*60}")
    print(f"BUG VERIFICATION:")
    print(f"ComposedPress reports: {composed_press.compression_ratio:.4f}")
    print(f"Should report: {expected_prune_ratio:.4f}")
    print(f"Difference: {abs(composed_press.compression_ratio - expected_prune_ratio):.4f}")

    if abs(composed_press.compression_ratio - expected_prune_ratio) > 0.01:
        print(f"\n❌ BUG CONFIRMED: ComposedPress incorrectly multiplies prune fractions!")
        print(f"   It's computing: 0.3 * 0.2 = {0.3 * 0.2}")
        print(f"   Should compute: 1 - (0.7 * 0.8) = {expected_prune_ratio}")
    else:
        print(f"\n✅ No bug detected.")

