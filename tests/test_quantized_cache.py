# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import Cache, DynamicCache, QuantizedCache
from transformers.cache_utils import QuantizedLayer

from kvpress import DecodingPress, KnormPress, KVzipPress
from kvpress.utils import extract_keys_and_values
from tests.fixtures import unit_test_model  # noqa: F401


class IdentityQuantizedLayer(QuantizedLayer):
    """QuantizedLayer storing tensors as is, so that a QuantizedCache can be compared exactly to a DynamicCache."""

    def _quantize(self, tensor, axis):
        return tensor

    def _dequantize(self, q_tensor):
        return q_tensor


class IdentityQuantizedCache(QuantizedCache):
    def __init__(self, config, residual_length: int):
        layers = [IdentityQuantizedLayer(residual_length=residual_length) for _ in range(config.num_hidden_layers)]
        Cache.__init__(self, layers=layers)


@torch.no_grad()
def test_decoding_press_compresses_quantized_cache_residual(unit_test_model):  # noqa: F811
    """The most recent tokens of a QuantizedCache are stored in full precision (the residual) until residual_length
    is reached. Decoding compression must keep them, so the QuantizedCache must match the DynamicCache."""
    input_ids = torch.randint(0, 1000, (1, 120), device=unit_test_model.device)
    caches = [DynamicCache(), IdentityQuantizedCache(unit_test_model.config, residual_length=32)]
    for cache in caches:
        press = DecodingPress(base_press=KnormPress(), compression_interval=10, target_size=48)
        unit_test_model(input_ids[:, :40], past_key_values=cache)
        with press(unit_test_model):
            for position in range(40, input_ids.shape[1]):
                position_ids = torch.tensor([[position]], device=unit_test_model.device)
                unit_test_model(input_ids[:, position : position + 1], past_key_values=cache, position_ids=position_ids)

    dynamic_cache, quantized_cache = caches
    assert quantized_cache.get_seq_length() == dynamic_cache.get_seq_length()
    for layer_idx, layer in enumerate(dynamic_cache.layers):
        keys, values = extract_keys_and_values(quantized_cache, layer_idx)
        assert torch.equal(keys, layer.keys)
        assert torch.equal(values, layer.values)


@torch.no_grad()
def test_kvzip_press_scores_quantized_cache_like_dynamic_cache(unit_test_model):  # noqa: F811
    """KVzip appends the reconstruction tokens to the cache, a QuantizedCache keeps them in its residual."""
    input_ids = torch.randint(0, 1000, (1, 100), device=unit_test_model.device)
    masked_key_indices = []
    for cache in [DynamicCache(), IdentityQuantizedCache(unit_test_model.config, residual_length=128)]:
        with KVzipPress(compression_ratio=0.5)(unit_test_model):
            unit_test_model.model(input_ids=input_ids, past_key_values=cache)
        masked_key_indices.append([layer.self_attn.masked_key_indices for layer in unit_test_model.model.layers])

    for dynamic_indices, quantized_indices in zip(*masked_key_indices):
        assert all(torch.equal(a, b) for a, b in zip(dynamic_indices, quantized_indices))
