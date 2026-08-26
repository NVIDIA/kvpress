# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch.nn as nn

from kvpress.presses.pyramidkv_press import PyramidKVPress


class MockConfig:
    def __init__(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers


class MockModule(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx


def scorer_press_layer_budget(q_len, compression_ratio):
    return round(q_len * (1 - compression_ratio))


@pytest.mark.parametrize("layer_budget_func", ["pyramidkv_press_layer_budget", "scorer_press_layer_budget"])
@pytest.mark.parametrize("num_hidden_layers", [32, 64, 128])
@pytest.mark.parametrize("compression_ratio", [0.1, 0.25, 0.3, 0.5, 0.6, 0.75, 0.8])
@pytest.mark.parametrize("q_len", [1024, 2787, 4096, 6591, 8192])
def test_mean_layer_budget(layer_budget_func, num_hidden_layers, compression_ratio, q_len):
    total_n_kept = 0

    if layer_budget_func == "pyramidkv_press_layer_budget":
        config = MockConfig(num_hidden_layers)
        press = PyramidKVPress()
        press.compression_ratio = compression_ratio

    for layer_idx in range(num_hidden_layers):
        if layer_budget_func == "pyramidkv_press_layer_budget":
            module = MockModule(config, layer_idx)
            n_kept = press.get_layer_budget(module, q_len)
        elif layer_budget_func == "scorer_press_layer_budget":
            n_kept = scorer_press_layer_budget(q_len, compression_ratio)
        else:
            raise ValueError(f"Unsupported layer_budget_func: {layer_budget_func}")
        total_n_kept += n_kept

    mean_n_kept = total_n_kept / num_hidden_layers
    assert mean_n_kept == pytest.approx(q_len * (1 - compression_ratio), rel=1e-3)


def test_get_layer_budget_fallback_floors_to_one():
    # A short context with a high compression ratio fails the pyramid-shape guard and falls
    # back to the SnapKV formula, which must never floor to 0 (that would empty the cache).
    press = PyramidKVPress()
    press.compression_ratio = 0.96
    module = MockModule(MockConfig(32), layer_idx=0)
    assert press.get_layer_budget(module, q_len=10) >= 1


def test_get_layer_budget_single_layer_no_division_by_zero():
    press = PyramidKVPress()
    press.compression_ratio = 0.5
    module = MockModule(MockConfig(num_hidden_layers=1), layer_idx=0)
    n_kept = press.get_layer_budget(module, q_len=4096)
    assert n_kept >= 1
