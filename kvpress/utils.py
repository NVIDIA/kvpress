# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from transformers import Cache, QuantizedCache
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.llama.modeling_llama import rotate_half
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


def get_query_states(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Extracts the query states from an attention module and applies rotary embeddings.

    Thin wrapper around :func:`get_prerope_query_states` for presses that score with
    post-RoPE queries. Callers slice ``hidden_states`` and ``position_embeddings``
    themselves, e.g. ``hidden_states[:, -window_size:]`` together with
    ``(cos[:, -window_size:], sin[:, -window_size:])``.

    Parameters
    ----------
    module : nn.Module
        The attention module from which to extract query states.
    hidden_states : torch.Tensor
        The input hidden states of shape (batch_size, seq_len, hidden_dim).
    position_embeddings : tuple[torch.Tensor, torch.Tensor]
        The (cos, sin) rotary embeddings, each of shape (batch_size, seq_len, head_dim).

    Returns
    -------
    query_states : torch.Tensor
        The rotated query states of shape (batch_size, num_heads, seq_len, head_dim).
    """
    cos, sin = position_embeddings
    query_states = get_prerope_query_states(module, hidden_states)
    return apply_rope(query_states, cos.unsqueeze(1), sin.unsqueeze(1))


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary embeddings to query or key states.

    Models with a partial rotary embedding (``partial_rotary_factor < 1``, e.g. Phi-4-mini) only rotate
    the first ``cos.shape[-1]`` channels of each head and leave the others unchanged, as in transformers.

    Parameters
    ----------
    x : torch.Tensor
        Query or key states of shape (..., head_dim).
    cos, sin : torch.Tensor
        Rotary embeddings of shape (..., rotary_dim), broadcastable to ``x[..., :rotary_dim]``.

    Returns
    -------
    torch.Tensor
        The rotated states, with the same shape as ``x``.
    """
    rotary_dim = cos.shape[-1]
    if rotary_dim == x.shape[-1]:
        return (x * cos) + (rotate_half(x) * sin)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    return torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)


def get_rope_matrices(cos: torch.Tensor, sin: torch.Tensor, head_dim: int) -> torch.Tensor:
    """
    Returns the rotary embedding of each position as a matrix, such that ``R[i] @ x`` equals
    ``apply_rope(x, cos[i], sin[i])`` for a head vector ``x``. Channels that a partial rotary
    embedding leaves unrotated get the identity.

    Parameters
    ----------
    cos, sin : torch.Tensor
        Rotary embeddings of shape (n_positions, rotary_dim).
    head_dim : int
        The head dimension.

    Returns
    -------
    torch.Tensor
        Rotation matrices of shape (n_positions, head_dim, head_dim).
    """
    rotary_dim, half_dim = cos.shape[-1], cos.shape[-1] // 2
    factory_kwargs = {"device": cos.device, "dtype": cos.dtype}
    rotate_half_matrix = torch.zeros((rotary_dim, rotary_dim), **factory_kwargs)
    rotate_half_matrix[half_dim:, :half_dim] = torch.eye(half_dim, **factory_kwargs)
    rotate_half_matrix[:half_dim, half_dim:] = -torch.eye(half_dim, **factory_kwargs)
    rotations = cos.unsqueeze(1) * torch.eye(rotary_dim, **factory_kwargs) + sin.unsqueeze(1) * rotate_half_matrix
    if rotary_dim == head_dim:
        return rotations
    matrices = torch.eye(head_dim, **factory_kwargs).repeat(cos.shape[0], 1, 1)
    matrices[:, :rotary_dim, :rotary_dim] = rotations
    return matrices


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
