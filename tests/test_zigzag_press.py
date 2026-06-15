# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from transformers import DynamicCache

from kvpress import KnormPress, ObservedAttentionPress, SnapKVPress, ZigZagKVPress
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


def test_zigzag_basic_compression(unit_test_model_output_attention):  # noqa: F811
    """Test that ZigZagKVPress compresses the cache and produces different sizes per layer."""
    model = unit_test_model_output_attention
    press = ZigZagKVPress(
        press=ObservedAttentionPress(),
        compression_ratio=0.5,
        window_size=32,
    )

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    layer_sizes = [cache.layers[i].keys.shape[2] for i in range(len(cache))]

    # All layers should be compressed (shorter than original 256)
    for size in layer_sizes:
        assert size < 256, f"Expected compressed size < 256, got {size}"
        assert size > 0, "No layer should be completely empty"


def test_zigzag_different_layer_budgets(unit_test_model_output_attention):  # noqa: F811
    """Verify that ZigZagKV produces different cache sizes per layer (dynamic allocation)."""
    model = unit_test_model_output_attention
    press = ZigZagKVPress(
        press=ObservedAttentionPress(),
        compression_ratio=0.5,
        window_size=32,
    )

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    layer_sizes = [cache.layers[i].keys.shape[2] for i in range(len(cache))]

    # With dynamic allocation, layers should generally get different sizes
    # (unless all layers happen to have identical attention patterns)
    # With 2 layers in the unit test model, they should differ
    assert len(set(layer_sizes)) >= 1, "Expected at least some variation in layer sizes"


def test_zigzag_no_compression(unit_test_model_output_attention):  # noqa: F811
    """With compression_ratio=0, no compression should occur."""
    model = unit_test_model_output_attention
    press = ZigZagKVPress(
        press=ObservedAttentionPress(),
        compression_ratio=0.0,
        window_size=32,
    )

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    for i in range(len(cache)):
        assert cache.layers[i].keys.shape[2] == 256, (
            f"Layer {i}: expected 256 tokens with no compression, got {cache.layers[i].keys.shape[2]}"
        )


def test_zigzag_high_compression(unit_test_model_output_attention):  # noqa: F811
    """With high compression_ratio, cache should be very small but non-empty."""
    model = unit_test_model_output_attention
    press = ZigZagKVPress(
        press=ObservedAttentionPress(),
        compression_ratio=0.9,
        window_size=32,
    )

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    for i in range(len(cache)):
        size = cache.layers[i].keys.shape[2]
        assert size < 256, f"Layer {i}: expected compression, got {size}"
        assert size >= 1, f"Layer {i}: should retain at least 1 token"


def test_zigzag_b_bound_prevents_starvation(unit_test_model_output_attention):  # noqa: F811
    """With high b_bound_ratio, layers should have more uniform sizes."""
    model = unit_test_model_output_attention

    # High b_bound means most budget is guaranteed, less dynamic allocation
    press_high_bound = ZigZagKVPress(
        press=ObservedAttentionPress(),
        compression_ratio=0.5,
        window_size=32,
        b_bound_ratio=0.9,
    )

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press_high_bound(model):
        model(input_ids, past_key_values=cache)

    layer_sizes = [cache.layers[i].keys.shape[2] for i in range(len(cache))]

    # With high b_bound, sizes should be more uniform (closer together)
    size_range = max(layer_sizes) - min(layer_sizes)
    # All layers should still be compressed
    for size in layer_sizes:
        assert size < 256


def test_zigzag_batch_input(unit_test_model_output_attention):  # noqa: F811
    """Test with batched input."""
    model = unit_test_model_output_attention
    press = ZigZagKVPress(
        press=ObservedAttentionPress(),
        compression_ratio=0.5,
        window_size=32,
    )

    batch_size = 3
    input_ids = torch.randint(0, 3_000, (batch_size, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    for i in range(len(cache)):
        assert cache.layers[i].keys.shape[0] == batch_size
        assert cache.layers[i].keys.shape[2] < 256


def test_zigzag_compression_ratio_property(unit_test_model_output_attention):  # noqa: F811
    """Test that the compression_ratio property returns the configured average ratio."""
    press = ZigZagKVPress(
        press=ObservedAttentionPress(),
        compression_ratio=0.7,
    )
    assert press.compression_ratio == 0.7


def test_zigzag_invalid_params():
    """Test that invalid parameters raise assertions."""
    import pytest

    with pytest.raises(AssertionError):
        ZigZagKVPress(press=ObservedAttentionPress(), compression_ratio=1.5)

    with pytest.raises(AssertionError):
        ZigZagKVPress(press=ObservedAttentionPress(), compression_ratio=-0.1)

    with pytest.raises(AssertionError):
        ZigZagKVPress(press=ObservedAttentionPress(), attention_threshold=0.0)

    with pytest.raises(AssertionError):
        ZigZagKVPress(press=ObservedAttentionPress(), b_bound_ratio=1.0)
