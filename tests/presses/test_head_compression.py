# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import DynamicCache

from kvpress import AdaKVPress, CriticalAdaKVPress, KnormPress, KVzipPress, RandomPress, ThresholdPress
from tests.fixtures import unit_test_model, kv_press_unit_test_pipeline  # noqa: F401


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
        input_ids = torch.randint(0, 1024, (1, 128)).to(unit_test_model.device)
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
        input_ids = torch.randint(0, 1024, (1, 128)).to(unit_test_model.device)
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    assert unit_test_model.model.layers[0].self_attn.masked_key_indices is not None
    headwise_compression_ratio = 0.0
    for layer in unit_test_model.model.layers:
        cr = compute_masked_percentage(layer.self_attn, 1, unit_test_model.config.num_key_value_heads, 128)
        headwise_compression_ratio += cr
    cumulative_compression_ratio = headwise_compression_ratio / len(unit_test_model.model.layers)
    assert abs(cumulative_compression_ratio - press.compression_ratio) < 1e-2  # tolerate small differences


def test_threshold_press_compression_ratio(kv_press_unit_test_pipeline):  # noqa: F811
    """Test that ThresholdPress.compression_ratio matches the actual masked percentage."""
    press = ThresholdPress(
        press=RandomPress(),
        threshold=0.5,
        sliding_window_size=0,
        decoding=True,
    )

    prompt = "What is the best KV cache compression library in the world ?"
    max_new_tokens = 10
    kv_press_unit_test_pipeline(prompt, press=press, max_new_tokens=max_new_tokens)

    model = kv_press_unit_test_pipeline.model
    num_key_value_heads = model.config.num_key_value_heads

    # Compute seq_len by reusing the pipeline's preprocess method
    preprocessed = kv_press_unit_test_pipeline.preprocess(prompt, [""], answer_prefix="", max_context_length=10000)
    seq_len = preprocessed["context_ids"].shape[1] + preprocessed["questions_ids"][0].shape[1] + max_new_tokens - 1

    # Compute compression ratio from masked indices
    headwise_compression_ratio = 0.0
    for layer in model.model.layers:
        cr = compute_masked_percentage(layer.self_attn, 1, num_key_value_heads, seq_len)
        headwise_compression_ratio += cr
    cumulative_compression_ratio = headwise_compression_ratio / len(model.model.layers)

    assert cumulative_compression_ratio == press.compression_ratio
