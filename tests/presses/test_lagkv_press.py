# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import LagKVPress


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lagkv_constant_reference_scores_are_finite(dtype):
    press = LagKVPress(lag_size=4, cross_scoring=True)
    states = torch.ones((1, 1, 2, 4, 8), dtype=dtype)

    scores = press._get_states_score(states)

    torch.testing.assert_close(scores, torch.full_like(scores, 0.25))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "previous,reference,expected",
    [
        ([[-32768.0, -32768.0]] * 4, [[32768.0, 32768.0]] * 4, [0.25] * 4),
        (
            [[-32768.0, -32768.0], [-32768.0, 32768.0], [32768.0, -32768.0], [32768.0, 32768.0]],
            [[32768.0, 32768.0]] * 4,
            [0.0, 0.5, 0.5, 0.0],
        ),
        (
            [[-32768.0, -32768.0], [-32768.0, 32768.0], [32768.0, -32768.0], [32768.0, 32768.0]],
            [[-32768.0, -32768.0], [32768.0, 32768.0]] * 2,
            [0.1651192253, 0.3348807747, 0.3348807747, 0.1651192253],
        ),
    ],
    ids=["constant-uniform", "constant-nonuniform", "wide-reference-range"],
)
def test_lagkv_large_finite_states_preserve_relative_scores(dtype, previous, reference, expected):
    press = LagKVPress(lag_size=4, cross_scoring=True)
    states = torch.tensor([[previous, reference]], dtype=dtype).unsqueeze(0)
    original = states.clone()

    scores = press._get_states_score(states)

    # Centering can exceed float16's range; the final probabilities are representable in every dtype.
    torch.testing.assert_close(scores, torch.tensor([[[expected]]], dtype=dtype))
    torch.testing.assert_close(states, original)


def test_lagkv_large_float16_values_preserve_protected_cache():
    press = LagKVPress(compression_ratio=0.25, n_sink=4, lag_size=4, cross_scoring=True)
    keys = torch.arange(24, dtype=torch.float16).reshape(1, 1, 12, 2)
    values = torch.zeros_like(keys)
    values[..., 4:8, :] = -32768
    values[..., 8:, :] = 32768

    scores = press.score(None, None, keys, values, None, {})
    compressed_keys, compressed_values = press.compress(SimpleNamespace(head_dim=2), None, keys, values, None, {})

    assert torch.isfinite(scores).all()
    assert compressed_keys.shape == (1, 1, 9, 2)
    assert compressed_keys.dtype == compressed_values.dtype == torch.float16
    protected_keys = torch.cat((keys[..., :4, :], keys[..., -4:, :]), dim=-2)
    matches = (protected_keys.unsqueeze(-2) == compressed_keys.unsqueeze(-3)).all(dim=-1)
    assert matches.any(dim=-1).all()
    # The first channel identifies each key, so its matching value can be checked without relying on topk tie order.
    indices = (compressed_keys[..., 0] / 2).long().unsqueeze(-1).expand_as(compressed_keys)
    torch.testing.assert_close(compressed_values, values.gather(2, indices))


@pytest.mark.parametrize("scale", [1.0, 1e-8])
def test_lagkv_nonconstant_reference_preserves_relative_scores(scale):
    press = LagKVPress(lag_size=4, cross_scoring=True)
    states = (
        torch.tensor(
            [[[[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]]]
        )
        * scale
    )

    scores = press._get_states_score(states)

    # Unit-range normalization gives standard deviations [0, 1/sqrt(2), 1/sqrt(2), 0].
    expected = torch.tensor([[[[0.1651192253, 0.3348807747, 0.3348807747, 0.1651192253]]]])
    torch.testing.assert_close(scores, expected)


@pytest.mark.parametrize("cross_scoring", [False, True])
def test_lagkv_repeated_token_prefill_preserves_protected_cache(cross_scoring):
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    ).eval()
    press = LagKVPress(compression_ratio=0.25, n_sink=4, lag_size=4, cross_scoring=cross_scoring)

    with torch.no_grad():
        output = model(torch.ones((1, 12), dtype=torch.long), use_cache=True)
        cache = output.past_key_values.layers[0]
        module = model.model.layers[0].self_attn
        scores = press.score(module, None, cache.keys, cache.values, None, {})
        compressed_keys, _ = press.compress(module, None, cache.keys, cache.values, None, {})

    assert compressed_keys.shape[-2] == 9
    protected_keys = torch.cat((cache.keys[..., :4, :], cache.keys[..., -4:, :]), dim=-2)
    matches = (protected_keys.unsqueeze(-2) == compressed_keys.unsqueeze(-3)).all(dim=-1)
    assert matches.any(dim=-1).all()
    assert scores.shape == (1, 1, 12)
    assert torch.isfinite(scores).all()
    assert (scores[..., :4] == 1).all()
    assert (scores[..., -4:] == 1).all()
    assert (scores[..., 4:8] < 1).all()
