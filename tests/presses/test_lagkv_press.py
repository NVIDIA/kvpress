# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
