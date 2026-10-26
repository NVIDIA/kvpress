# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import Cache, DynamicCache, QuantizedCache
from transformers.cache_utils import QuantizedLayer

from kvpress import CAMPress, DecodingPress, KnormPress
from kvpress.utils import extract_keys_and_values
from tests.fixtures import unit_test_model  # noqa: F401


class IdentityQuantizedLayer(QuantizedLayer):
    """Quantized layer without quantization error, so that no quantization backend is needed."""

    def _quantize(self, tensor, axis):
        return tensor.clone()

    def _dequantize(self, q_tensor):
        return q_tensor


def make_quantized_cache(num_layers: int, residual_length: int = 128) -> QuantizedCache:
    cache = QuantizedCache.__new__(QuantizedCache)
    Cache.__init__(cache, layers=[IdentityQuantizedLayer(residual_length=residual_length) for _ in range(num_layers)])
    return cache


def test_extract_keys_and_values_includes_the_full_precision_residual():
    keys, values = torch.randn(2, 1, 2, 10, 4).unbind(0)
    cache = make_quantized_cache(num_layers=1, residual_length=8)
    cache.update(keys[:, :, :6], values[:, :, :6], layer_idx=0)
    for i in range(6, 10):
        cache.update(keys[:, :, i : i + 1], values[:, :, i : i + 1], layer_idx=0)
    assert cache.layers[0].keys.shape[2] == 4

    extracted_keys, extracted_values = extract_keys_and_values(cache, 0)
    assert torch.equal(extracted_keys, keys)
    assert torch.equal(extracted_values, values)


@torch.no_grad()
def test_prefill_compression_with_quantized_cache(unit_test_model):  # noqa: F811
    input_ids = torch.randint(0, 1000, (1, 32), device=unit_test_model.device)
    dynamic_cache = DynamicCache()
    quantized_cache = make_quantized_cache(unit_test_model.config.num_hidden_layers)
    for cache in [dynamic_cache, quantized_cache]:
        with KnormPress(compression_ratio=0.5)(unit_test_model):
            unit_test_model(input_ids, past_key_values=cache)

    for layer_idx, layer in enumerate(quantized_cache.layers):
        keys, values = extract_keys_and_values(quantized_cache, layer_idx)
        assert torch.equal(keys, dynamic_cache.layers[layer_idx].keys)
        assert torch.equal(values, dynamic_cache.layers[layer_idx].values)
        assert layer.keys.numel() == layer.values.numel() == 0
        assert layer.get_seq_length() == 16


@pytest.mark.parametrize("press_cls", [DecodingPress, CAMPress])
@torch.no_grad()
def test_decoding_compression_with_quantized_cache_matches_dynamic_cache(unit_test_model, press_cls):  # noqa: F811
    input_ids = torch.randint(0, 1000, (1, 32), device=unit_test_model.device)
    dynamic_cache = DynamicCache()
    quantized_cache = make_quantized_cache(unit_test_model.config.num_hidden_layers)
    outputs = []
    for cache in [dynamic_cache, quantized_cache]:
        torch.manual_seed(0)
        press = press_cls(base_press=KnormPress(), compression_interval=4, target_size=24)
        with press(unit_test_model):
            outputs.append(
                unit_test_model.generate(input_ids, past_key_values=cache, max_new_tokens=10, do_sample=False)
            )

    assert torch.equal(outputs[0], outputs[1])
    for layer_idx, layer in enumerate(quantized_cache.layers):
        keys, values = extract_keys_and_values(quantized_cache, layer_idx)
        assert torch.equal(keys, dynamic_cache.layers[layer_idx].keys)
        assert torch.equal(values, dynamic_cache.layers[layer_idx].values)
        assert layer.get_seq_length() == keys.shape[2]
