# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import ExpectedAttentionPress


@pytest.fixture(scope="module")
def attention_module():
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    module = model.model.layers[0].self_attn
    module.rotary_emb = model.model.rotary_emb
    return module


def make_inputs(module, n_hidden_states, n_keys):
    torch.manual_seed(1)
    hidden_states = torch.randn(1, n_hidden_states, module.config.hidden_size)
    keys, values = torch.randn(2, 1, module.config.num_key_value_heads, n_keys, module.head_dim).unbind(0)
    return hidden_states, keys, values


def reference_scores(press, module, hidden_states, keys, values, next_position):
    """Scores computed with the future queries placed after next_position by construction."""
    reference = dataclasses.replace(press)
    apply_avg_rope = reference.apply_avg_rope
    reference.apply_avg_rope = lambda module, mu, cov, q_len: apply_avg_rope(module, mu, cov, next_position)
    kwargs = {"position_ids": torch.arange(hidden_states.shape[1]).unsqueeze(0)}
    return reference.score(module, hidden_states, keys, values, None, kwargs)


@torch.no_grad()
def test_prefill_scores_are_unchanged(attention_module):
    press = ExpectedAttentionPress(compression_ratio=0.5)
    hidden_states, keys, values = make_inputs(attention_module, 32, 32)
    kwargs = {"position_ids": torch.arange(32).unsqueeze(0), "cache_position": torch.arange(32)}

    scores = press.score(attention_module, hidden_states, keys, values, None, kwargs)
    assert torch.equal(scores, reference_scores(press, attention_module, hidden_states, keys, values, 32))


@pytest.mark.parametrize("use_covariance", [True, False])
@torch.no_grad()
def test_future_positions_follow_the_absolute_position(attention_module, use_covariance):
    press = ExpectedAttentionPress(compression_ratio=0.5, use_covariance=use_covariance)
    hidden_states, keys, values = make_inputs(attention_module, 16, 140)
    # e.g. 16 buffered hidden states during decoding, with a compressed cache shorter than the positions
    kwargs = {"position_ids": torch.arange(184, 200).unsqueeze(0), "cache_position": torch.arange(124, 140)}

    scores = press.score(attention_module, hidden_states, keys, values, None, kwargs)
    expected = reference_scores(press, attention_module, hidden_states, keys, values, 200)
    torch.testing.assert_close(scores, expected)
    assert not torch.allclose(scores, reference_scores(press, attention_module, hidden_states, keys, values, 16))
