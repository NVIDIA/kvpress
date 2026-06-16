# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from transformers import DynamicCache

from kvpress import SnapKVPress, ZigZagKVPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_zigzag_basic_compression(unit_test_model):  # noqa: F811
    """ZigZagKVPress compresses the cache (flash-compatible, no eager required)."""
    model = unit_test_model
    press = ZigZagKVPress(press=SnapKVPress(window_size=32), compression_ratio=0.5)

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    layer_sizes = [cache.layers[i].keys.shape[2] for i in range(len(cache))]
    for size in layer_sizes:
        assert size < 256, f"Expected compressed size < 256, got {size}"
        assert size > 0, "No layer should be completely empty"


def test_zigzag_average_ratio_preserved(unit_test_model):  # noqa: F811
    """The mean per-layer budget should match the target compression ratio."""
    model = unit_test_model
    seq_len = 256
    press = ZigZagKVPress(press=SnapKVPress(window_size=32), compression_ratio=0.5)

    input_ids = torch.randint(0, 3_000, (1, seq_len), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    layer_sizes = [cache.layers[i].keys.shape[2] for i in range(len(cache))]
    mean_size = sum(layer_sizes) / len(layer_sizes)
    target = seq_len * (1 - 0.5)
    # Allow tolerance for integer rounding and clamping.
    assert abs(mean_size - target) <= 0.2 * target, f"mean={mean_size}, target={target}"


def test_zigzag_no_compression(unit_test_model):  # noqa: F811
    """With compression_ratio=0, no compression should occur."""
    model = unit_test_model
    press = ZigZagKVPress(press=SnapKVPress(window_size=32), compression_ratio=0.0)

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    for i in range(len(cache)):
        assert cache.layers[i].keys.shape[2] == 256, (
            f"Layer {i}: expected 256 tokens with no compression, got {cache.layers[i].keys.shape[2]}"
        )


def test_zigzag_high_compression(unit_test_model):  # noqa: F811
    """With high compression_ratio, cache should be very small but non-empty."""
    model = unit_test_model
    press = ZigZagKVPress(press=SnapKVPress(window_size=32), compression_ratio=0.9)

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    for i in range(len(cache)):
        size = cache.layers[i].keys.shape[2]
        assert size < 256, f"Layer {i}: expected compression, got {size}"
        assert size >= 1, f"Layer {i}: should retain at least 1 token"


def test_zigzag_b_bound_floor(unit_test_model):  # noqa: F811
    """b_bound_ratio guarantees a minimum per-layer budget."""
    model = unit_test_model
    seq_len = 256
    ratio = 0.5
    b_bound_ratio = 0.4
    press = ZigZagKVPress(
        press=SnapKVPress(window_size=32),
        compression_ratio=ratio,
        b_bound_ratio=b_bound_ratio,
    )

    input_ids = torch.randint(0, 3_000, (1, seq_len), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    b_avg = seq_len * (1 - ratio)
    floor = int(round(b_avg * b_bound_ratio)) - 1  # allow rounding slack
    for i in range(len(cache)):
        assert cache.layers[i].keys.shape[2] >= floor


def test_zigzag_batch_input(unit_test_model):  # noqa: F811
    """Test with batched input."""
    model = unit_test_model
    press = ZigZagKVPress(press=SnapKVPress(window_size=32), compression_ratio=0.5)

    batch_size = 3
    input_ids = torch.randint(0, 3_000, (batch_size, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    for i in range(len(cache)):
        assert cache.layers[i].keys.shape[0] == batch_size
        assert cache.layers[i].keys.shape[2] < 256


def test_zigzag_generation_after_compression(unit_test_model):  # noqa: F811
    """A follow-up forward pass must work with variable per-layer cache sizes."""
    model = unit_test_model
    press = ZigZagKVPress(press=SnapKVPress(window_size=32), compression_ratio=0.5)

    input_ids = torch.randint(0, 3_000, (1, 256), device=model.device)
    cache = DynamicCache()

    with press(model):
        model(input_ids, past_key_values=cache)

    ctx_len = cache.get_seq_length()
    question = torch.randint(0, 3_000, (1, 8), device=model.device)
    position_ids = torch.arange(ctx_len, ctx_len + 8, device=model.device).unsqueeze(0)
    out = model(question, past_key_values=cache, position_ids=position_ids)
    assert out.logits.shape[1] == 8


def test_zigzag_compression_ratio_attribute(unit_test_model):  # noqa: F811
    """The configured average ratio is exposed as a plain attribute."""
    press = ZigZagKVPress(press=SnapKVPress(), compression_ratio=0.7)
    assert press.compression_ratio == 0.7


def test_zigzag_invalid_params():
    """Invalid parameters raise assertions."""
    with pytest.raises(AssertionError):
        ZigZagKVPress(press=SnapKVPress(), compression_ratio=1.5)

    with pytest.raises(AssertionError):
        ZigZagKVPress(press=SnapKVPress(), compression_ratio=-0.1)

    with pytest.raises(AssertionError):
        ZigZagKVPress(press=SnapKVPress(), attention_threshold=0.0)

    with pytest.raises(AssertionError):
        ZigZagKVPress(press=SnapKVPress(), b_bound_ratio=1.0)
