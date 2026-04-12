# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for CriticalKVPress head_dim fallback.

Models like Qwen2 do not expose `config.head_dim` as an explicit attribute.
Without the fallback to `hidden_size // num_attention_heads`, CriticalKVPress
raises AttributeError in `vwl1norm`.
"""

import pytest
import torch
from transformers import DynamicCache

from kvpress import CriticalAdaKVPress, CriticalKVPress, KnormPress
from tests.fixtures import unit_test_model  # noqa: F401


@pytest.fixture
def model_without_head_dim(unit_test_model):  # noqa: F811
    """Return the unit-test model with config.head_dim deleted to simulate Qwen2-like configs."""
    config = unit_test_model.config
    original = getattr(config, "head_dim", None)
    if hasattr(config, "head_dim"):
        delattr(config, "head_dim")
    yield unit_test_model
    # Restore
    if original is not None:
        config.head_dim = original


def test_criticalkv_without_head_dim(model_without_head_dim):
    """CriticalKVPress must work when config.head_dim is absent."""
    assert not hasattr(model_without_head_dim.config, "head_dim")
    press = CriticalKVPress(press=KnormPress(compression_ratio=0.5))
    with press(model_without_head_dim):
        input_ids = torch.randint(0, 1024, (1, 128), device=model_without_head_dim.device)
        model_without_head_dim(input_ids, past_key_values=DynamicCache()).past_key_values


def test_criticaladakv_without_head_dim(model_without_head_dim):
    """CriticalAdaKVPress must work when config.head_dim is absent."""
    assert not hasattr(model_without_head_dim.config, "head_dim")
    press = CriticalAdaKVPress(press=KnormPress(compression_ratio=0.5))
    with press(model_without_head_dim):
        input_ids = torch.randint(0, 1024, (1, 128), device=model_without_head_dim.device)
        model_without_head_dim(input_ids, past_key_values=DynamicCache()).past_key_values
