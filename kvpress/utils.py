# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from transformers import Cache, QuantizedCache


def get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract pre-RoPE query states; dispatched through the model adapter."""
    from kvpress.adapters import get_adapter_from_module

    return get_adapter_from_module(module).prerope_queries(module, hidden_states)


def get_prerope_key_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract pre-RoPE key states; dispatched through the model adapter."""
    from kvpress.adapters import get_adapter_from_module

    return get_adapter_from_module(module).prerope_keys(module, hidden_states)


def dequantize_layer(cache_layer) -> tuple[torch.Tensor, torch.Tensor]:
    keys = cache_layer._dequantize(cache_layer._quantized_keys)
    values = cache_layer._dequantize(cache_layer._quantized_values)
    return keys, values


def compute_n_kept(k_len: int, compression_ratio: float) -> int:
    """Number of KV pairs to keep after applying ``compression_ratio``.

    Returns ``0`` when ``compression_ratio >= 1.0`` (the caller asks to evict
    the entire layer by design). Otherwise clamps to ``>= 1`` so that a short
    context or float-truncation cannot silently empty the cache when the caller
    only meant to compress it partially.
    """
    if compression_ratio >= 1.0:
        return 0
    return max(1, int(k_len * (1 - compression_ratio)))


def extract_keys_and_values(cache: Cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the keys and values from a given cache layer,
    handling both quantized and unquantized caches.
    """
    if isinstance(cache, QuantizedCache):
        keys, values = dequantize_layer(cache.layers[layer_idx])
    else:
        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values
    return keys, values
