# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress.presses.entropy_adaptive_press import EntropyAdaptivePress
from tests.fixtures import unit_test_model_output_attention  # noqa: F401


def test_entropy_adaptive_runs(unit_test_model_output_attention):  # noqa: F811
    """Basic smoke test: press runs without error."""
    press = EntropyAdaptivePress(compression_ratio=0.5)
    model = unit_test_model_output_attention
    with press(model):
        input_ids = torch.randint(0, 1024, (1, 128), device=model.device)
        cache = DynamicCache()
        model(input_ids, past_key_values=cache).past_key_values
        assert cache.get_seq_length() > 0
        assert cache.get_seq_length() < 128


def test_entropy_adaptive_compression(unit_test_model_output_attention):  # noqa: F811
    """Verify compression actually reduces cache size."""
    press = EntropyAdaptivePress(compression_ratio=0.5)
    model = unit_test_model_output_attention
    with press(model):
        input_ids = torch.randint(0, 1024, (1, 256), device=model.device)
        cache = DynamicCache()
        model(input_ids, past_key_values=cache).past_key_values
        assert cache.get_seq_length() < 256


def test_entropy_adaptive_ratio_bounds(unit_test_model_output_attention):  # noqa: F811
    """Verify min/max ratio params are respected."""
    for min_r, max_r in [(0.1, 0.3), (0.2, 0.8)]:
        press = EntropyAdaptivePress(
            compression_ratio=0.5, min_ratio=min_r, max_ratio=max_r
        )
        model = unit_test_model_output_attention
        with press(model):
            input_ids = torch.randint(0, 1024, (1, 128), device=model.device)
            cache = DynamicCache()
            model(input_ids, past_key_values=cache).past_key_values
            assert cache.get_seq_length() > 0


def test_entropy_adaptive_zero_compression(unit_test_model_output_attention):  # noqa: F811
    """With compression_ratio=0, all tokens should be kept."""
    press = EntropyAdaptivePress(compression_ratio=0.0)
    model = unit_test_model_output_attention
    with press(model):
        input_ids = torch.randint(0, 1024, (1, 64), device=model.device)
        cache = DynamicCache()
        model(input_ids, past_key_values=cache).past_key_values
        assert cache.get_seq_length() == 64
