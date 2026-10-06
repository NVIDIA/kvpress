# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import QuerySpanPress
from kvpress.presses.query_span_press import forward_propagate
from tests.fixtures import unit_test_model  # noqa: F401


def test_forward_propagate_matches_recursion():
    scores = torch.rand(2, 3, 50)
    gamma = 0.8
    expected = scores.clone()
    for t in range(1, scores.shape[-1]):
        expected[..., t] = torch.maximum(scores[..., t], gamma * expected[..., t - 1])
    torch.testing.assert_close(forward_propagate(scores, gamma), expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("compression_ratio", [0.2, 0.5, 0.9])
def test_query_span_press_budget_and_sinks(unit_test_model, compression_ratio):  # noqa: F811
    press = QuerySpanPress(compression_ratio=compression_ratio, window_size=8)
    seq_len = 128
    input_ids = torch.randint(0, 1024, (1, seq_len), device=unit_test_model.device)
    cache = DynamicCache()
    with press(unit_test_model):
        unit_test_model(input_ids, past_key_values=cache)

    layers = unit_test_model.model.layers
    num_kv_heads = unit_test_model.config.num_key_value_heads
    n_kept = int(seq_len * (1 - compression_ratio))
    n_pruned = 0
    for layer in layers:
        _, head_indices, seq_indices = layer.self_attn.masked_key_indices
        assert (seq_indices >= press.n_sink).all(), "attention sinks must never be pruned"
        assert (head_indices < num_kv_heads).all()
        n_pruned += len(seq_indices)
    # The budget is shared across layers and heads
    assert n_pruned == len(layers) * num_kv_heads * (seq_len - n_kept)
    # The cache itself is not shortened (pruned pairs are masked in the attention)
    assert cache.get_seq_length() == seq_len


def test_query_span_press_no_compression(unit_test_model):  # noqa: F811
    press = QuerySpanPress(compression_ratio=0.0)
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
    for layer in unit_test_model.model.layers:
        layer.self_attn.masked_key_indices = None
    with press(unit_test_model):
        unit_test_model(input_ids, past_key_values=DynamicCache())
    assert all(layer.self_attn.masked_key_indices is None for layer in unit_test_model.model.layers)
