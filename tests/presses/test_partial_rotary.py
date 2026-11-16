# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache, Phi3Config, Phi3ForCausalLM
from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb

from kvpress import (
    CapPress,
    CompactorPress,
    DropKVPress,
    ExpectedAttentionPress,
    KeyRerotationPress,
    KnormPress,
    NonCausalAttnPress,
    SnapKVPress,
    ThinKPress,
    TOVAPress,
)
from kvpress.utils import (
    apply_rope,
    get_prerope_key_states,
    get_prerope_query_states,
    get_query_states,
    get_rope_matrices,
)


@pytest.fixture(scope="module")
def partial_rotary_model():
    """Tiny Phi3 model that rotates only 12 of its 16 head channels, the layout of Phi-4-mini."""
    torch.manual_seed(0)
    config = Phi3Config(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        partial_rotary_factor=0.75,
    )
    return Phi3ForCausalLM(config).eval()


@torch.no_grad()
def test_query_and_key_rope_match_transformers(partial_rotary_model):
    module = partial_rotary_model.model.layers[0].self_attn
    hidden_states = torch.randn(2, 7, 64)
    cos, sin = partial_rotary_model.model.rotary_emb(hidden_states, torch.arange(7).unsqueeze(0))
    assert cos.shape[-1] == 12

    queries = get_prerope_query_states(module, hidden_states)
    keys = get_prerope_key_states(module, hidden_states)
    expected_queries, expected_keys = apply_rotary_pos_emb(queries, keys, cos, sin)

    torch.testing.assert_close(get_query_states(module, hidden_states, (cos, sin)), expected_queries, rtol=0, atol=0)
    torch.testing.assert_close(apply_rope(keys, cos.unsqueeze(1), sin.unsqueeze(1)), expected_keys, rtol=0, atol=0)


@pytest.mark.parametrize("rotary_dim", [12, 16])
def test_rope_matrices_match_apply_rope(rotary_dim):
    # As in transformers, the two halves of the rotated channels share their angles
    angles = torch.randn(5, rotary_dim // 2).repeat(1, 2)
    cos, sin = angles.cos(), angles.sin()
    x = torch.randn(5, 16)
    matrices = get_rope_matrices(cos, sin, head_dim=16)
    torch.testing.assert_close(torch.einsum("nij,nj->ni", matrices, x), apply_rope(x, cos, sin))


@torch.no_grad()
def test_rerotate_keys_with_partial_rotary(partial_rotary_model):
    """Rerotated keys must match the keys RoPE gives the kept tokens at their new positions."""
    module = partial_rotary_model.model.layers[0].self_attn
    rotary_emb = partial_rotary_model.model.rotary_emb
    module.rotary_emb = rotary_emb
    hidden_states = torch.randn(1, 20, 64)
    keys = get_prerope_key_states(module, hidden_states)
    cos, sin = rotary_emb(hidden_states, torch.arange(20).unsqueeze(0))
    rotated_keys = apply_rotary_pos_emb(keys, keys, cos, sin)[1]

    indices = torch.stack([torch.randperm(20)[:8].sort().values for _ in range(2)]).unsqueeze(0)
    rerotated_keys = KeyRerotationPress.rerotate_keys(module, indices, rotated_keys)

    kept_keys = keys.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1]))
    cos, sin = rotary_emb(hidden_states, torch.arange(8).unsqueeze(0))
    expected = apply_rotary_pos_emb(kept_keys, kept_keys, cos, sin)[1]
    torch.testing.assert_close(rerotated_keys, expected)


@pytest.mark.parametrize(
    "press",
    [
        SnapKVPress(compression_ratio=0.5, window_size=8),
        TOVAPress(compression_ratio=0.5),
        ThinKPress(key_channel_compression_ratio=0.5, window_size=8),
        DropKVPress(compression_ratio=0.5, window_size=8),
        CompactorPress(compression_ratio=0.5),
        NonCausalAttnPress(compression_ratio=0.5, chunk_size=16),
        ExpectedAttentionPress(compression_ratio=0.5),
        CapPress(compression_ratio=0.5),
        KeyRerotationPress(press=KnormPress(compression_ratio=0.5)),
    ],
    ids=lambda press: type(press).__name__,
)
@torch.no_grad()
def test_presses_run_with_partial_rotary(partial_rotary_model, press):
    with press(partial_rotary_model):
        input_ids = torch.randint(0, 1024, (1, 64))
        partial_rotary_model(input_ids, past_key_values=DynamicCache())
