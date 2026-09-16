# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import DynamicCache

from kvpress import KnormPress
from kvpress.adapters import LlamaLikeAdapter, get_adapter, get_adapter_from_module
from tests.fixtures import unit_test_model  # noqa: F401


def test_unit_test_model_uses_llama_like_adapter(unit_test_model):  # noqa: F811
    adapter = get_adapter(unit_test_model)
    assert isinstance(adapter, LlamaLikeAdapter)
    modules = adapter.compressible_modules(unit_test_model)
    assert len(modules) == len(unit_test_model.model.layers)
    assert get_adapter_from_module(modules[0]).__class__ is LlamaLikeAdapter


def test_llama_adapter_hooks_match_layer_count(unit_test_model):  # noqa: F811
    press = KnormPress(compression_ratio=0.2)
    with press(unit_test_model):
        for layer in unit_test_model.model.layers:
            assert len(layer.self_attn._forward_hooks) == 1
    for layer in unit_test_model.model.layers:
        assert len(layer.self_attn._forward_hooks) == 0


def test_llama_adapter_knorm_cache_length(unit_test_model):  # noqa: F811
    press = KnormPress(compression_ratio=0.2)
    input_ids = unit_test_model.dummy_inputs["input_ids"].to(unit_test_model.device)
    with press(unit_test_model):
        cache = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values
    seq_len = input_ids.shape[-1]
    kept = int(seq_len * 0.8)
    for layer in cache.layers:
        assert layer.keys.shape[2] == kept == cache.get_seq_length()
