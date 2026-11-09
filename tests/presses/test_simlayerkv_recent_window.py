# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import SimLayerKVPress


@pytest.mark.parametrize(
    "n_recent, n_last, expected_positions, expected_ratio",
    [
        (1, 1, [0, 1], 14 / 16),
        (2, 2, [0, 1], 14 / 16),
        (4, 4, [0, 1], 14 / 16),
        (4, 1, [0, 1, 13, 14, 15], 11 / 16),
        (4, 2, [0, 1, 14, 15], 12 / 16),
    ],
)
def test_simlayer_retains_only_sinks_when_recent_window_is_empty(n_recent, n_last, expected_positions, expected_ratio):
    """A zero-length recent window must not duplicate the entire original cache."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            _attn_implementation="sdpa",
        )
    ).eval()
    press = SimLayerKVPress(lazy_threshold=0.0, n_initial=2, n_recent=n_recent, n_last=n_last)
    input_ids = torch.arange(16).unsqueeze(0)

    with torch.no_grad():
        reference = model(input_ids, use_cache=True).past_key_values.layers[0]
        with press(model):
            compressed = model(input_ids, use_cache=True).past_key_values.layers[0]

    assert compressed.keys.shape[2] == len(expected_positions)
    assert compressed.values.shape[2] == len(expected_positions)
    torch.testing.assert_close(compressed.keys, reference.keys[:, :, expected_positions])
    torch.testing.assert_close(compressed.values, reference.values[:, :, expected_positions])
    assert press.compression_ratios == pytest.approx([expected_ratio])
    assert press.compression_ratio == pytest.approx(expected_ratio)
