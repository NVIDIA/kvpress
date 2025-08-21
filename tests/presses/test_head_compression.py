# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import DynamicCache

from kvpress import AdaKVPress, CriticalAdaKVPress, KnormPress, KVzipPress
from tests.fixtures import unit_test_model  # noqa: F401


def compute_masked_percentage(module, batch_size, num_key_value_heads, seq_len):
    """
    Compute the percentage of masked indices from module.masked_key_indices.
    """
    if module.masked_key_indices is None:
        return 0.0

    batch_indices, head_indices, seq_indices = module.masked_key_indices
    num_masked = len(batch_indices)
    total_positions = batch_size * num_key_value_heads * seq_len
    masked_percentage = num_masked / total_positions
    return masked_percentage


@pytest.mark.parametrize("wrapper_press", [AdaKVPress, CriticalAdaKVPress])
@pytest.mark.parametrize("compression_ratio", [0.2, 0.4, 0.6, 0.8])
def test_wrapper_head_compression(unit_test_model, wrapper_press, compression_ratio):  # noqa: F811
    p = KnormPress(compression_ratio=compression_ratio)
    press = wrapper_press(press=p)
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, 128))
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    assert unit_test_model.model.layers[0].self_attn.masked_key_indices is not None
    headwise_compression_ratio = 0.0
    for layer in unit_test_model.model.layers:
        cr = compute_masked_percentage(layer.self_attn, 1, unit_test_model.config.num_key_value_heads, 128)
        headwise_compression_ratio += cr
    cumulative_compression_ratio = headwise_compression_ratio / len(unit_test_model.model.layers)
    assert abs(cumulative_compression_ratio - press.compression_ratio) < 1e-2  # tolerate small differences


# Only for KVzipPress, since it's the only non-wrapper press with head compression (apart from Duo)
@pytest.mark.parametrize("press", [KVzipPress])
@pytest.mark.parametrize("compression_ratio", [0.2, 0.4, 0.6, 0.8])
@pytest.mark.parametrize("layerwise", [True, False])
def test_head_compression(unit_test_model, press, compression_ratio, layerwise):  # noqa: F811
    press = KVzipPress(compression_ratio=compression_ratio, layerwise=layerwise)
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, 128))
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    assert unit_test_model.model.layers[0].self_attn.masked_key_indices is not None
    headwise_compression_ratio = 0.0
    for layer in unit_test_model.model.layers:
        cr = compute_masked_percentage(layer.self_attn, 1, unit_test_model.config.num_key_value_heads, 128)
        headwise_compression_ratio += cr
    cumulative_compression_ratio = headwise_compression_ratio / len(unit_test_model.model.layers)
    assert abs(cumulative_compression_ratio - press.compression_ratio) < 1e-2  # tolerate small differences
