# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F
from transformers import DynamicCache, Gemma3ForCausalLM, Gemma3TextConfig, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress import CAMPress, ExpectedAttentionPress, KnormPress, SnapKVPress


@pytest.fixture(scope="module")
def gemma3_attention_inputs():
    """Inputs of an eager Gemma3 attention layer whose scaling differs from 1 / sqrt(head_dim)."""
    config = Gemma3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        query_pre_attn_scalar=64,
        vocab_size=128,
        layer_types=["full_attention"],
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Gemma3ForCausalLM(config).eval()
    module = model.model.layers[0].self_attn
    assert module.scaling != module.head_dim**-0.5

    captured = {}

    def hook(module, args, kwargs, output):
        captured.update(kwargs=kwargs, attentions=output[1])

    handle = module.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        cache = DynamicCache()
        model(torch.randint(0, 128, (1, 24)), past_key_values=cache)
    handle.remove()
    module.rotary_emb = model.model.rotary_emb
    return module, captured["kwargs"], cache.layers[0].keys, cache.layers[0].values, captured["attentions"]


def test_snapkv_window_attention_matches_eager_attention(gemma3_attention_inputs):
    module, kwargs, keys, _, attentions = gemma3_attention_inputs
    window_size = 4
    attn_weights = SnapKVPress.compute_window_attention(
        module, kwargs["hidden_states"], keys, window_size, kwargs["position_embeddings"]
    )
    torch.testing.assert_close(attn_weights, attentions[..., -window_size:, :-window_size])


def test_cam_current_token_attention_matches_eager_attention(gemma3_attention_inputs):
    module, kwargs, keys, _, attentions = gemma3_attention_inputs
    attn_weights = CAMPress(base_press=KnormPress())._compute_current_token_attention(
        module, kwargs["hidden_states"], keys, kwargs
    )
    torch.testing.assert_close(attn_weights, attentions[..., -1:, :])


def test_expected_attention_uses_module_scaling():
    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2
    )
    model = LlamaForCausalLM(config).eval()
    module = model.model.layers[0].self_attn
    module.rotary_emb = model.model.rotary_emb
    module.scaling = 0.1
    hidden_states = torch.randn(1, 24, config.hidden_size)
    keys, values = torch.randn(2, 1, config.num_key_value_heads, 24, module.head_dim).unbind(0)

    press = ExpectedAttentionPress(n_sink=0, use_vnorm=False)
    kwargs = {"cache_position": torch.arange(24)}
    with torch.no_grad():
        scores = press.score(module, hidden_states, keys, values, None, kwargs)
        mean_query, cov_query = press.get_query_statistics(module, hidden_states)
    num_key_value_groups = module.config.num_attention_heads // keys.shape[1]
    repeated_keys = repeat_kv(keys, num_key_value_groups)
    logits = torch.einsum("bhd,bhnd->bhn", mean_query, repeated_keys) * module.scaling
    logits += torch.einsum("bhnd,bhde,bhne->bhn", repeated_keys, cov_query, repeated_keys) * module.scaling**2 / 2
    expected = F.softmax(logits, dim=-1).view(1, keys.shape[1], num_key_value_groups, -1).mean(dim=2)
    torch.testing.assert_close(scores, expected)
