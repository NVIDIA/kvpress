# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from kvpress import KVComposePress


@pytest.fixture(params=["cpu", "cuda"])
def model(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        attn_implementation="sdpa",
    )
    return LlamaForCausalLM(config).to(request.param).eval()


@pytest.mark.parametrize("fail_on_forward", [1, 2])
def test_forward_error_restores_model_and_allows_reuse(model, fail_on_forward):
    press = KVComposePress(compression_ratio=0.5)
    original_forward = model.model.forward
    error = RuntimeError("injected forward failure")
    calls = 0

    def fail_once(module, args):
        nonlocal calls
        calls += 1
        if calls == fail_on_forward:
            raise error

    handle = model.model.layers[0].register_forward_pre_hook(fail_once)
    input_ids = torch.arange(8, device=model.device).unsqueeze(0)
    try:
        with pytest.raises(RuntimeError) as exc, torch.no_grad():
            with press(model):
                model(input_ids, past_key_values=DynamicCache())
        assert exc.value is error
    finally:
        handle.remove()

    assert model.config._attn_implementation == "sdpa"
    assert model.model.forward == original_forward
    assert not hasattr(model, "original_forward_KVComposePress")
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
    assert press.cache is None

    with torch.no_grad(), press(model):
        cache = model(input_ids, past_key_values=DynamicCache()).past_key_values
    assert 0 < cache.get_seq_length() < input_ids.shape[1]
    assert model.config._attn_implementation == "sdpa"


def test_context_error_before_forward_is_preserved(model):
    press = KVComposePress(compression_ratio=0.5)
    error = RuntimeError("context failed before prefill")
    with pytest.raises(RuntimeError) as exc:
        with press(model):
            raise error
    assert exc.value is error
    assert model.config._attn_implementation == "sdpa"
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
    assert press.cache is None


def test_context_without_forward_restores_model(model):
    press = KVComposePress(compression_ratio=0.5)
    original_forward = model.model.forward
    with press(model):
        pass
    assert model.model.forward == original_forward
    assert model.config._attn_implementation == "sdpa"
    assert press.cache is None
