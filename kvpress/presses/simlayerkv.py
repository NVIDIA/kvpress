# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import math
import inspect

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from kvpress.presses.snapkv_press import BasePress

@dataclass
class SimLayerKVPress(BasePress):
    initial_tokens: int = 4
    recent_tokens: int = 1024 # according to the paper 
    w_last: int = 32
    compression_ratio: float = 0.9 
    window_size: int = 32

    def compute_window_attention(module, hidden_states, keys,window_size):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, q_len, _ = hidden_states.shape

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -window_size :])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -window_size :])
            query_states = qkv[..., : module.num_heads * module.head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, window_size, module.num_heads, module.head_dim).transpose(1, 2)

        # Apply RoPE
        if "position_ids" in inspect.signature(module.rotary_emb.forward).parameters:
            position_ids = torch.arange(q_len - window_size, q_len).unsqueeze(0).to(query_states.device)
            cos, sin = module.rotary_emb(query_states, position_ids)
        else:
            cos, sin = module.rotary_emb(query_states, q_len)
            cos, sin = cos[-window_size :].unsqueeze(0), sin[-window_size :].unsqueeze(0)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, module.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., : -window_size]

        return attn_weights



    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        # Get basic dimensions
        bsz, num_heads, seq_len, _ = keys.shape
        device, dtype = keys.device, keys.dtype

        # Get attention weights
        if attentions is None:
            attn_weights = compute_window_attention(module, hidden_states, keys, self.window_size)
        else:
            attn_weights = attentions

        # Identify lazy layers
        initial_recent_attn = attn_weights[:, :, :, :self.initial_tokens].sum(dim=-1) + \
                              attn_weights[:, :, :, -self.recent_tokens:].sum(dim=-1)

        avg_attn = initial_recent_attn[:, :, -self.w_last:].mean()

        is_lazy_layer = avg_attn > self.compression_ratio

        # Create scores tensor
        scores = torch.zeros(bsz, num_heads, seq_len, device=device, dtype=dtype)

        if is_lazy_layer:
            scores[:, :, :self.initial_tokens] = 1.0
            scores[:, :, -self.recent_tokens:] = 1.0
        else:
            scores[:, :, :] = 1.0

        return scores