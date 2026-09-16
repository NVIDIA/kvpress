# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter for the Qwen3.5 hybrid stack (Gated DeltaNet + periodic full attention).

Three things differ from a dense Llama-style stack and motivate a dedicated adapter:

1. Only a minority of layers hold a KV cache. The rest are ``linear_attention``
   (Gated DeltaNet) layers whose state is a fixed-size recurrent state, so there is
   nothing to compress there.
2. ``Qwen3_5DynamicCache`` is not a ``transformers.Cache``; it stores plain
   ``key_cache`` / ``value_cache`` lists with ``None`` entries for linear layers.
3. Queries come out of a *gated* ``q_proj`` (which projects to ``2 * head_dim`` per
   head), and RoPE is interleaved mRoPE applied to only part of the head dimension.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache, rotate_half

from kvpress.adapters.base import CacheLike, ModelAdapter, register_adapter


@register_adapter("qwen3_5", "qwen3_5_text")
class Qwen3_5Adapter(ModelAdapter):
    def language_model(self, model: PreTrainedModel) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        return model.model

    def compressible_modules(self, model: PreTrainedModel) -> list[nn.Module]:
        """Only ``full_attention`` layers own a KV cache; linear-attention layers are skipped."""
        return [
            layer.self_attn
            for layer in self.language_model(model).layers
            if getattr(layer, "layer_type", None) == "full_attention" and hasattr(layer, "self_attn")
        ]

    def prepare_module(self, model: PreTrainedModel, module: nn.Module) -> None:
        rotary_emb = getattr(self.language_model(model), "rotary_emb", None)
        if rotary_emb is not None:
            module.rotary_emb = rotary_emb

    def get_keys_values(self, cache: CacheLike, module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        return cache.key_cache[module.layer_idx], cache.value_cache[module.layer_idx]

    def set_keys_values(
        self, cache: CacheLike, module: nn.Module, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        cache.key_cache[module.layer_idx] = keys
        cache.value_cache[module.layer_idx] = values

    def make_cache(self, model: PreTrainedModel) -> CacheLike:
        config = model.config
        return Qwen3_5DynamicCache(getattr(config, "text_config", config))

    def supports_multi_token_continuation(self) -> bool:
        """Qwen3.5's Gated DeltaNet can only carry its recurrent state forward one token at a time.

        ``Qwen3_5GatedDeltaNet.forward`` only reuses the cached recurrent state when
        ``seq_len == 1``; a longer forward pass over an already-populated cache silently
        restarts the recurrence from ``initial_state=None``, discarding everything the
        linear-attention layers learned during prefill.
        """
        return False

    def rewind_cache(self, cache: CacheLike, seq_lengths: list[int]) -> None:
        """Truncate the attention cache back to ``seq_lengths``.

        Linear-attention layers hold no per-token state to truncate, and their
        recurrent state is left untouched: it has already absorbed the generated
        tokens and cannot be rewound, which is why multi-question reuse of one
        prefilled cache is not exact for this architecture.
        """
        for layer_idx, sequence_length in enumerate(seq_lengths):
            if cache.key_cache[layer_idx] is None:
                continue
            cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :sequence_length]
            cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :sequence_length]

    def prerope_queries(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """Split the gated ``q_proj`` output and drop the gate, keeping the query half."""
        input_shape = hidden_states.shape[:-1]
        query_states, _gate = torch.chunk(
            module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
        )
        return module.q_norm(query_states.view(*input_shape, -1, module.head_dim)).transpose(1, 2)

    def prerope_keys(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        key_states = module.k_proj(hidden_states).view(*input_shape, -1, module.head_dim)
        return module.k_norm(key_states).transpose(1, 2)

    def apply_rope(
        self, module: nn.Module, states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Partial rotary embedding: only the leading ``cos.shape[-1]`` dims are rotated."""
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]
        rotated, passthrough = states[..., :rotary_dim], states[..., rotary_dim:]
        rotated = (rotated * cos) + (rotate_half(rotated) * sin)
        return torch.cat([rotated, passthrough], dim=-1)
