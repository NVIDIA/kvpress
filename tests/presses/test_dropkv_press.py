# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.nn import functional as F
from transformers import DynamicCache

from kvpress import DropKVPress
from kvpress.utils import compute_n_kept
from tests.fixtures import unit_test_model  # noqa: F401
from tests.fixtures import unit_test_model_output_attention  # noqa: F401


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


def _reference_scores(
    query_states: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Straightforward query-head implementation used as a test oracle."""
    batch_size, num_query_heads, window_size, head_dim = query_states.shape
    num_kv_heads = keys.shape[1]
    seq_len = keys.shape[2]
    num_groups = num_query_heads // num_kv_heads

    repeated_keys = _repeat_kv(keys, num_groups)
    repeated_values = _repeat_kv(values, num_groups)
    logits = torch.matmul(query_states, repeated_keys.transpose(-1, -2)) / math.sqrt(head_dim)

    causal_mask = torch.full((window_size, seq_len), float("-inf"), device=query_states.device)
    causal_mask = torch.triu(causal_mask, diagonal=seq_len - window_size + 1)
    logits += causal_mask
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(probabilities, repeated_values)

    probabilities_float = probabilities.float()
    weights = (probabilities_float / (1.0 - probabilities_float + epsilon)).square()

    weight_sum = weights.sum(dim=-2)
    value_norm_squared = repeated_values.float().square().sum(dim=-1)
    term_a = weight_sum * value_norm_squared

    weighted_outputs = torch.matmul(weights.transpose(-1, -2), attention_output.float())
    term_b = 2.0 * torch.sum(repeated_values.float() * weighted_outputs, dim=-1)

    output_norm_squared = attention_output.float().square().sum(dim=-1)
    term_c = torch.matmul(weights.transpose(-1, -2), output_norm_squared.unsqueeze(-1)).squeeze(-1)

    scores = term_a - term_b + term_c
    return scores.view(batch_size, num_kv_heads, num_groups, seq_len).mean(dim=2)


@pytest.mark.parametrize(
    "num_query_heads,num_kv_heads",
    [
        (8, 8),
        (8, 2),
        (14, 2),
    ],
)
def test_dropkv_scores_match_reference(num_query_heads, num_kv_heads):
    torch.manual_seed(0)
    batch_size, seq_len, window_size, head_dim = 2, 17, 4, 8
    query_states = torch.randn(batch_size, num_query_heads, window_size, head_dim)
    keys = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    values = torch.randn_like(keys)
    press = DropKVPress(window_size=window_size)

    expected = _reference_scores(query_states, keys, values, press.epsilon)
    probabilities = press._compute_window_probabilities(query_states, keys)
    actual = press._compute_scores(probabilities, keys, values)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0},
        {"kernel_size": 0},
        {"kernel_size": 4},
        {"epsilon": 0},
    ],
)
def test_dropkv_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        DropKVPress(**kwargs)


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [64, 4])
def test_dropkv_press_keeps_the_shared_budget(unit_test_model, seq_len):  # noqa: F811
    """The budget follows ScorerPress, including when the context is shorter than the window."""
    compression_ratio = 0.5
    press = DropKVPress(compression_ratio=compression_ratio, window_size=8, kernel_size=3)
    input_ids = torch.randint(0, 1024, (1, seq_len), device=unit_test_model.device)

    with press(unit_test_model):
        cache = DynamicCache()
        unit_test_model(input_ids, past_key_values=cache)

    expected = compute_n_kept(seq_len, compression_ratio)
    for layer in cache.layers:
        assert layer.keys.shape[2] == expected
        assert layer.values.shape[2] == expected


@torch.no_grad()
def test_dropkv_score_protects_recent_window(unit_test_model):  # noqa: F811
    """The observation window outranks every other position, so top-k always keeps it."""
    window_size = 8
    recorded = []

    class RecordingDropKVPress(DropKVPress):
        def score(self, *args, **kwargs):
            scores = super().score(*args, **kwargs)
            recorded.append(scores.clone())
            return scores

    press = RecordingDropKVPress(compression_ratio=0.5, window_size=window_size, kernel_size=3)
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

    with press(unit_test_model):
        unit_test_model(input_ids, past_key_values=DynamicCache())

    assert recorded
    for scores in recorded:
        window, rest = scores[..., -window_size:], scores[..., :-window_size]
        assert torch.all(window == window[..., :1])
        assert window.min() > rest.max()


@torch.no_grad()
def test_dropkv_reuses_eager_attention_weights(unit_test_model_output_attention):  # noqa: F811
    """With eager attention the scores must match the ones recomputed from the queries."""
    window_size = 8
    model = unit_test_model_output_attention
    input_ids = torch.randint(0, 1024, (1, 40), device=model.device)

    recorded: dict[str, list] = {"reused": [], "recomputed": []}

    class RecordingDropKVPress(DropKVPress):
        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            reused = super().score(module, hidden_states, keys, values, attentions, kwargs)
            recomputed = super().score(module, hidden_states, keys, values, None, kwargs)
            recorded["reused"].append(reused.clone())
            recorded["recomputed"].append(recomputed.clone())
            return reused

    press = RecordingDropKVPress(compression_ratio=0.5, window_size=window_size, kernel_size=3)
    with press(model):
        model(input_ids, past_key_values=DynamicCache(), output_attentions=True)

    assert recorded["reused"], "the press never saw eager attention weights"
    for reused, recomputed in zip(recorded["reused"], recorded["recomputed"]):
        torch.testing.assert_close(reused, recomputed, rtol=1e-4, atol=1e-4)
