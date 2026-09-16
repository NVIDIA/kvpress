# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    Qwen3_5ForCausalLM,
    apply_rotary_pos_emb,
)

from kvpress import KnormPress, SnapKVPress
from kvpress.adapters import Qwen3_5Adapter, get_adapter, get_adapter_from_module, has_adapter
from tests.fixtures import get_device

FULL_ATTENTION_LAYERS = [1, 3]


@pytest.fixture(scope="module")
def tiny_qwen3_5_model():
    """A randomly initialised Qwen3.5 stack: two linear-attention and two full-attention layers."""
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=2,
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.25,
        },
    )
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config).eval()
    return model.to(get_device())


def test_adapter_is_registered_for_qwen3_5(tiny_qwen3_5_model):
    assert has_adapter(tiny_qwen3_5_model)
    assert isinstance(get_adapter(tiny_qwen3_5_model), Qwen3_5Adapter)


def test_only_full_attention_layers_are_compressible(tiny_qwen3_5_model):
    """Linear-attention layers hold a recurrent state, not a KV cache, so they must be skipped."""
    adapter = get_adapter(tiny_qwen3_5_model)
    modules = adapter.compressible_modules(tiny_qwen3_5_model)
    assert [m.layer_idx for m in modules] == FULL_ATTENTION_LAYERS
    assert isinstance(get_adapter_from_module(modules[0]), Qwen3_5Adapter)


def test_hooks_only_land_on_full_attention_layers(tiny_qwen3_5_model):
    press = KnormPress(compression_ratio=0.5)
    with press(tiny_qwen3_5_model):
        for idx, layer in enumerate(tiny_qwen3_5_model.model.layers):
            if idx in FULL_ATTENTION_LAYERS:
                assert len(layer.self_attn._forward_hooks) == 1
            else:
                assert not hasattr(layer, "self_attn")
    for idx in FULL_ATTENTION_LAYERS:
        assert len(tiny_qwen3_5_model.model.layers[idx].self_attn._forward_hooks) == 0


def test_make_cache_returns_hybrid_cache(tiny_qwen3_5_model):
    cache = get_adapter(tiny_qwen3_5_model).make_cache(tiny_qwen3_5_model)
    assert isinstance(cache, Qwen3_5DynamicCache)
    assert len(cache) == tiny_qwen3_5_model.config.num_hidden_layers


@pytest.mark.parametrize("press_cls", [KnormPress, SnapKVPress])
def test_press_compresses_only_attention_layers(tiny_qwen3_5_model, press_cls):
    adapter = get_adapter(tiny_qwen3_5_model)
    seq_len = 128
    input_ids = torch.randint(0, 128, (1, seq_len), device=tiny_qwen3_5_model.device)

    cache = adapter.make_cache(tiny_qwen3_5_model)
    with press_cls(compression_ratio=0.5)(tiny_qwen3_5_model), torch.no_grad():
        tiny_qwen3_5_model.model(input_ids=input_ids, past_key_values=cache)

    kept = seq_len // 2
    assert cache.get_seq_length() == kept
    for layer_idx in range(tiny_qwen3_5_model.config.num_hidden_layers):
        if layer_idx in FULL_ATTENTION_LAYERS:
            assert cache.key_cache[layer_idx].shape[2] == kept
            assert cache.value_cache[layer_idx].shape[2] == kept
        else:
            assert cache.key_cache[layer_idx] is None


def test_prerope_states_match_the_model(tiny_qwen3_5_model):
    """The adapter must reproduce the gated q_proj split and the q/k norms exactly."""
    adapter = get_adapter(tiny_qwen3_5_model)
    module = adapter.compressible_modules(tiny_qwen3_5_model)[0]
    hidden_states = torch.randn(1, 8, tiny_qwen3_5_model.config.hidden_size, device=tiny_qwen3_5_model.device)

    with torch.no_grad():
        queries = adapter.prerope_queries(module, hidden_states)
        keys = adapter.prerope_keys(module, hidden_states)

        # Recompute the way Qwen3_5Attention.forward does.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        expected_q, _gate = torch.chunk(
            module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
        )
        expected_q = module.q_norm(expected_q.view(hidden_shape)).transpose(1, 2)
        expected_k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

    assert queries.shape == (1, tiny_qwen3_5_model.config.num_attention_heads, 8, module.head_dim)
    assert keys.shape == (1, tiny_qwen3_5_model.config.num_key_value_heads, 8, module.head_dim)
    torch.testing.assert_close(queries, expected_q)
    torch.testing.assert_close(keys, expected_k)


def test_apply_rope_matches_model_partial_rotary(tiny_qwen3_5_model):
    """Qwen3.5 rotates only part of the head dim, so a Llama-style full rotation would be wrong."""
    adapter = get_adapter(tiny_qwen3_5_model)
    module = adapter.compressible_modules(tiny_qwen3_5_model)[0]
    bsz, seq_len = 1, 8
    shape = (bsz, tiny_qwen3_5_model.config.num_attention_heads, seq_len, module.head_dim)
    queries = torch.randn(shape, device=tiny_qwen3_5_model.device)

    position_ids = torch.arange(seq_len, device=tiny_qwen3_5_model.device).view(1, 1, -1).expand(3, bsz, -1)
    with torch.no_grad():
        cos, sin = tiny_qwen3_5_model.model.rotary_emb(queries, position_ids)
        expected, _ = apply_rotary_pos_emb(queries, queries, cos, sin)
        actual = adapter.apply_rope(module, queries, cos, sin)

    assert cos.shape[-1] < module.head_dim, "expected a partial rotary embedding"
    torch.testing.assert_close(actual, expected)


def test_sequential_continuation_matches_a_single_pass(tiny_qwen3_5_model):
    """Extending a populated cache one token at a time must match one full-length forward.

    ``Qwen3_5GatedDeltaNet`` only reuses its recurrent state when ``seq_len == 1``, so a
    single multi-token continuation restarts the recurrence and drops the prefix. The
    size of that error depends on the learned decay rates, so it is not observable on a
    randomly initialised model; what is checked here is that the path the adapter opts
    into is the exact one.
    """
    adapter = get_adapter(tiny_qwen3_5_model)
    assert not adapter.supports_multi_token_continuation()

    ids = torch.randint(0, 128, (1, 40), device=tiny_qwen3_5_model.device)
    context_ids, rest_ids = ids[:, :32], ids[:, 32:]

    def recurrent_state(prefill_ids, continuation_ids, step):
        cache = adapter.make_cache(tiny_qwen3_5_model)
        with torch.no_grad():
            logits = tiny_qwen3_5_model(input_ids=prefill_ids, past_key_values=cache).logits
            for i in range(0, continuation_ids.shape[1], step):
                logits = tiny_qwen3_5_model(
                    input_ids=continuation_ids[:, i : i + step], past_key_values=cache
                ).logits
        return cache.recurrent_states[0], logits[0, -1]

    reference_state, reference_logits = recurrent_state(ids, ids[:, :0], 1)
    sequential_state, sequential_logits = recurrent_state(context_ids, rest_ids, 1)

    torch.testing.assert_close(sequential_state, reference_state, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(sequential_logits, reference_logits, rtol=1e-3, atol=1e-3)


def test_rewind_cache_restores_prefill_lengths(tiny_qwen3_5_model):
    adapter = get_adapter(tiny_qwen3_5_model)
    input_ids = torch.randint(0, 128, (1, 32), device=tiny_qwen3_5_model.device)
    cache = adapter.make_cache(tiny_qwen3_5_model)
    with torch.no_grad():
        tiny_qwen3_5_model.model(input_ids=input_ids, past_key_values=cache)
    prefill_lengths = [cache.get_seq_length(i) for i in range(len(cache))]

    with torch.no_grad():
        tiny_qwen3_5_model.model(
            input_ids=torch.randint(0, 128, (1, 4), device=tiny_qwen3_5_model.device),
            past_key_values=cache,
        )
    assert cache.get_seq_length() == 36

    adapter.rewind_cache(cache, prefill_lengths)
    assert cache.get_seq_length() == 32
    for layer_idx in FULL_ATTENTION_LAYERS:
        assert cache.key_cache[layer_idx].shape[2] == 32
