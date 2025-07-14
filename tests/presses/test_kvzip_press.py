# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

from transformers import AutoTokenizer, DynamicCache

from kvpress import KVzipPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_kvzip_press(unit_test_model):  # noqa: F811
    tokenizer = AutoTokenizer.from_pretrained("MaxJeblick/llama2-0b-unit-test")
    compression_ratio = 0.7

    for press in [
        KVzipPress(compression_ratio),
        KVzipPress(compression_ratio, layerwise=True),  # uniform compression ratios across layers
    ]:

        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        context = " ".join(random.choices(words, k=3000))  # dummy text
        context_ids = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        context_length = context_ids.shape[1]

        press.context = context
        press.suffix = "Dummy"

        # Test initial prefill
        cache = DynamicCache()
        with press(unit_test_model):
            unit_test_model(
                input_ids=context_ids,
                past_key_values=cache,
                num_logits_to_keep=1,
            )

        n_kv_full = 0
        for keys in cache.key_cache:
            n_kv_full += keys[..., 0].numel()  # head_dim dimension

        assert cache.key_cache[0].shape[-2] == context_length, print(
            "cache seq_length does not match the original context_length"
        )

        n_pruned = 0
        for layer in unit_test_model.model.layers:
            module = layer.self_attn
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            n_pruned += len(seq_indices)

        pruned_ratio = n_pruned / n_kv_full

        # Allow up to 0.1% error
        tolerance = 0.001  # 0.1%
        assert (
            abs(pruned_ratio - compression_ratio) <= tolerance
        ), f"pruned: {pruned_ratio:.3f}, setting: {compression_ratio:.3f}"
