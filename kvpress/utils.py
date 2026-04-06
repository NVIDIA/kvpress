# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from transformers import Cache, QuantizedCache
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


def get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Extracts the query states from a given attention module and hidden states tensor.

    This function supports multiple attention module types: Phi3Attention, Qwen3Attention, Gemma3Attention,
    and Llama-like modules. It handles the appropriate projection and reshaping to obtain the query states
    in the expected format.

    Parameters
    ----------
    module : nn.Module
        The attention module from which to extract query states. Must be one of
        Phi3Attention, Qwen3Attention, Gemma3Attention, or a Llama-like attention module
        with a 'q_proj' attribute.
    hidden_states : torch.Tensor
        The input hidden states of shape (batch_size, seq_len, hidden_dim).

    Returns
    -------
    query_states : torch.Tensor
        The extracted query states of shape (batch_size, num_heads, seq_len, head_dim).
    """
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    if isinstance(module, Phi3Attention):
        qkv = module.qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        # Assume Llama-like attention layer
        query_states = module.q_proj(hidden_states)
    else:
        raise NotImplementedError(f"Press not yet implemented for {module.__class__}.")

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    # Support for Qwen3 and Gemma3 QK norm
    if isinstance(module, (Qwen3Attention, Gemma3Attention)):
        query_states = module.q_norm(query_states)

    return query_states


def get_prerope_key_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Extracts the key states from a given attention module and hidden states tensor.

    This function supports multiple attention module types: Phi3Attention, Qwen3Attention, Gemma3Attention,
    and Llama-like modules. It handles the appropriate projection and reshaping to obtain the key states
    in the expected format.

    Parameters
    ----------
    module : nn.Module
        The attention module from which to extract key states. Must be one of
        Phi3Attention, Qwen3Attention, Gemma3Attention, or a Llama-like attention module
        with a 'k_proj' attribute.
    hidden_states : torch.Tensor
        The input hidden states of shape (batch_size, seq_len, hidden_dim).

    Returns
    -------
    key_states : torch.Tensor
        The extracted key states of shape (batch_size, num_heads, seq_len, head_dim).
    """
    bsz, k_len, _ = hidden_states.shape
    head_dim = module.head_dim
    if isinstance(module, Phi3Attention):
        qkv = module.qkv_proj(hidden_states)
        query_pos = module.config.num_attention_heads * module.head_dim
        key_states = qkv[..., query_pos : query_pos + module.num_key_value_heads * module.head_dim]
    elif hasattr(module, "k_proj"):
        # Assume Llama-like attention layer
        key_states = module.k_proj(hidden_states)
    else:
        raise NotImplementedError(f"Press not yet implemented for {module.__class__}.")

    key_states = key_states.view(bsz, k_len, -1, head_dim).transpose(1, 2)

    # Support for Qwen3 and Gemma3 QK norm
    if isinstance(module, (Qwen3Attention, Gemma3Attention)):
        key_states = module.k_norm(key_states)
    return key_states


def dequantize_layer(cache_layer) -> tuple[torch.Tensor, torch.Tensor]:
    keys = cache_layer._dequantize(cache_layer._quantized_keys)
    values = cache_layer._dequantize(cache_layer._quantized_values)
    return keys, values


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
