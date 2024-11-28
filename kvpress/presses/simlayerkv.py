# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv , LlamaAttention 

from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class SlimLayerKVPress(SnapKVPress):
    """
    SlimLayerKV identifies lazy layers based on attention patterns and optimizes KV cache usage
    by trimming the cache for these layers while retaining only initial and recent tokens.
    """
    
    window_size: int = 1
    threshold: float = 0.7
    initial_tokens: int = 5
    recent_tokens: int = 5

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        
        bsz, num_heads, seq_len, _ = keys.shape

        # Get attention weights for scoring
        if attentions is not None:
            attn_weights = attentions[..., -1:, :-1]
        else:
            attn_weights = self.compute_window_attention(module, hidden_states, keys)

        # Calculate attention to initial and recent tokens
        initial_weights = attn_weights[..., :self.initial_tokens].mean()
        recent_weights = attn_weights[..., -self.recent_tokens:].mean()
        
        # Create scores based on lazy layer identification
        scores = torch.ones((bsz, num_heads, seq_len), device=keys.device)
        if (initial_weights + recent_weights) > self.threshold:
            # For lazy layers, only keep initial and recent tokens
            mask = torch.zeros(seq_len, device=keys.device)
            mask[:self.initial_tokens] = 1
            mask[-self.recent_tokens:] = 1
            scores = scores * mask

        return scores