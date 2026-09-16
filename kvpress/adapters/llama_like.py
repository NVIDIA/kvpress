# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default adapter: dense Llama-style decoder stacks (Llama, Mistral, Phi3, Qwen2/3, Gemma3)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import DynamicCache, PreTrainedModel, QuantizedCache
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, Gemma3ForConditionalGeneration
from transformers.models.llama.modeling_llama import rotate_half
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress.adapters.base import CacheLike, ModelAdapter, register_adapter
from kvpress.utils import extract_keys_and_values


@register_adapter(
    "llama",
    "mistral",
    "mixtral",
    "phi3",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    default=True,
)
class LlamaLikeAdapter(ModelAdapter):
    def language_model(self, model: PreTrainedModel) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        return model.model

    def compressible_modules(self, model: PreTrainedModel) -> list[nn.Module]:
        language_model = self.language_model(model)
        modules = []
        for layer in language_model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if isinstance(model, Gemma3ForConditionalGeneration) and getattr(attn, "is_sliding", False):
                continue
            modules.append(attn)
        return modules

    def prepare_module(self, model: PreTrainedModel, module: nn.Module) -> None:
        language_model = self.language_model(model)
        rotary_emb = getattr(language_model, "rotary_emb", None)
        if rotary_emb is not None:
            module.rotary_emb = rotary_emb

    def get_keys_values(self, cache: CacheLike, module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        return extract_keys_and_values(cache, int(module.layer_idx))

    def set_keys_values(
        self, cache: CacheLike, module: nn.Module, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        cache_layer: Any = cache.layers[int(module.layer_idx)]
        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

    def make_cache(self, model: PreTrainedModel) -> CacheLike:
        return DynamicCache()

    def rewind_cache(self, cache: CacheLike, seq_lengths: list[int]) -> None:
        for layer_idx, sequence_length in enumerate(seq_lengths):
            layer: Any = cache.layers[layer_idx]
            layer.keys = layer.keys[:, :, :sequence_length]
            layer.values = layer.values[:, :, :sequence_length]
            if isinstance(cache, QuantizedCache):
                layer._quantized_keys = layer._quantized_keys[:, :, :sequence_length]
                layer._quantized_values = layer._quantized_values[:, :, :sequence_length]

    def prerope_queries(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim

        if isinstance(module, Phi3Attention):
            qkv = module.qkv_proj(hidden_states)
            query_states = qkv[..., : num_heads * head_dim]
        elif hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states)
        else:
            raise NotImplementedError(f"Press not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            query_states = module.q_norm(query_states)

        return query_states

    def prerope_keys(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, k_len, _ = hidden_states.shape
        head_dim = module.head_dim
        if isinstance(module, Phi3Attention):
            qkv = module.qkv_proj(hidden_states)
            query_pos = module.config.num_attention_heads * module.head_dim
            key_states = qkv[..., query_pos : query_pos + module.num_key_value_heads * module.head_dim]
        elif hasattr(module, "k_proj"):
            key_states = module.k_proj(hidden_states)
        else:
            raise NotImplementedError(f"Press not yet implemented for {module.__class__}.")

        key_states = key_states.view(bsz, k_len, -1, head_dim).transpose(1, 2)

        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            key_states = module.k_norm(key_states)
        return key_states

    def apply_rope(
        self, module: nn.Module, states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return (states * cos.unsqueeze(1)) + (rotate_half(states) * sin.unsqueeze(1))
