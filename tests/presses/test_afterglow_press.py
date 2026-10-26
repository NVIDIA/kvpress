# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import AdaKVPress, AfterglowPress, CriticalAdaKVPress, KnormPress, SnapKVPress
from kvpress.presses.afterglow_press import afterglow
from tests.fixtures import unit_test_model  # noqa: F401


def test_afterglow():
    scores = torch.rand(2, 3, 50) ** 4
    expected = scores.clone()
    for t in range(50):
        for j in range(4, t):
            expected[..., t] = torch.maximum(expected[..., t], 0.9 ** (t - j) * scores[..., j])
    assert torch.allclose(afterglow(scores, gamma=0.9, n_sink=4), expected, rtol=1e-4)
    with pytest.raises(ValueError):
        afterglow(-scores, gamma=0.9, n_sink=4)


@pytest.mark.parametrize("wrapper", [None, AdaKVPress, CriticalAdaKVPress])
def test_afterglow_press(unit_test_model, wrapper):  # noqa: F811
    press = AfterglowPress(SnapKVPress(window_size=2))
    press = press if wrapper is None else wrapper(press)
    press.compression_ratio = 0.5
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)
        cache = DynamicCache()
        unit_test_model(input_ids, past_key_values=cache)
    if wrapper is None:
        assert cache.get_seq_length() == 64
    else:
        n_masked = sum(len(layer.self_attn.masked_key_indices[0]) for layer in unit_test_model.model.layers)
        assert n_masked == unit_test_model.config.num_hidden_layers * unit_test_model.config.num_key_value_heads * 64


def test_afterglow_press_negative_scores(unit_test_model):  # noqa: F811
    with pytest.raises(ValueError):
        with AfterglowPress(KnormPress(compression_ratio=0.5))(unit_test_model):
            unit_test_model(torch.randint(0, 1024, (1, 128), device=unit_test_model.device))
