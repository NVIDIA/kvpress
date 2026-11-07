# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import CriticalAdaKVPress, KnormPress


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("compression_ratio, n_pruned", [(0.25, 4), (0.5, 8), (0.75, 12)])
def test_critical_adakv_batch_matches_independent_compression(batch_size, compression_ratio, n_pruned):
    """One prompt's head budgets must not depend on other prompts in its batch."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            _attn_implementation="sdpa",
        )
    ).eval()
    module = model.model.layers[0].self_attn
    press = CriticalAdaKVPress(press=KnormPress(compression_ratio=compression_ratio))
    input_ids = torch.arange(batch_size * 8).view(batch_size, 8)

    with torch.no_grad():
        output = model(input_ids, use_cache=True)
        cache = output.past_key_values.layers[0]
        keys, values = cache.keys, cache.values
        expected_masks = []
        for sample in range(batch_size):
            press.compress(module, None, keys[sample : sample + 1], values[sample : sample + 1], None, {})
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            mask = torch.zeros((1, 2, 8), dtype=torch.bool)
            mask[batch_indices, head_indices, seq_indices] = True
            expected_masks.append(mask)

        compressed_keys, compressed_values = press.compress(module, None, keys, values, None, {})

    # Adaptive compression marks positions for the attention implementation;
    # it does not physically remove entries from the rectangular cache.
    assert compressed_keys is keys
    assert compressed_values is values
    batch_indices, head_indices, seq_indices = module.masked_key_indices
    mask = torch.zeros((batch_size, 2, 8), dtype=torch.bool)
    mask[batch_indices, head_indices, seq_indices] = True
    assert (mask.sum(dim=(1, 2)) == n_pruned).all()
    assert (mask.sum(dim=-1) < 8).all()
    torch.testing.assert_close(mask, torch.cat(expected_masks))
