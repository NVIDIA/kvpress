# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class SlimLayerKVPress(SnapKVPress):
    initial_tokens: int = 4  
    recent_tokens: int = 1024   # according to paper 
    w_last: int = 32  
    compression_ratio: float = 0.9  

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
        
        if attentions is not None:
            attn_weights = attentions[..., -1:, :-1]
        else:
            attn_weights = self.compute_window_attention(module, hidden_states, keys)
        

        # Initialize scores with a small value instead of zeros
        scores = torch.full((bsz, num_heads, seq_len), -1e4, device=keys.device)
        # Always keep initial and recent tokens by giving them high scores
        scores[..., :self.initial_tokens] = float('inf')  # Keep initial tokens
        if seq_len > self.recent_tokens:
            scores[..., -self.recent_tokens:] = float('inf')  # Keep recent tokens
        
        if seq_len > 1:  # Prefilling phase
            last_window = attn_weights[..., -self.w_last:, :]
            initial_attn = last_window[..., :self.initial_tokens].mean(dim=(-2, -1))
            recent_attn = last_window[..., -self.recent_tokens:].mean(dim=(-2, -1))
            total_attn = (initial_attn + recent_attn) / self.w_last
            
            is_lazy = total_attn > self.compression_ratio
            
            if not is_lazy.any():
                # If not lazy, use attention scores for middle tokens
                middle_scores = attn_weights.mean(dim=-2)
                scores[..., self.initial_tokens:-self.recent_tokens] = middle_scores[..., self.initial_tokens:-self.recent_tokens]
                
        else:  # Decoding phase
            initial_attn = attn_weights[..., :self.initial_tokens].mean(dim=-1)
            recent_attn = attn_weights[..., -self.recent_tokens:].mean(dim=-1)
            total_attn = initial_attn + recent_attn
            
            is_lazy = total_attn > self.compression_ratio
            
            if not is_lazy.any():
                # If not lazy, use attention scores for middle tokens
                middle_scores = attn_weights.squeeze(-2)
                scores[..., self.initial_tokens:-self.recent_tokens] = middle_scores[..., self.initial_tokens:-self.recent_tokens]
        
        return scores