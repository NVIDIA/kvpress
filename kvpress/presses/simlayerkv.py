# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.snapkv_press import SnapKVPress



@dataclass
class SlimLayerKVPress(SnapKVPress):
    threshold: float = 0.7
    initial_tokens: int = 5
    recent_tokens: int = 5
    compression_ratio: float = 0.25  

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

        initial_weights = attn_weights[..., :self.initial_tokens].mean()
        recent_weights = attn_weights[..., -self.recent_tokens:].mean()
        
        scores = torch.zeros((bsz, num_heads, seq_len), device=keys.device)
        
        if (initial_weights + recent_weights) > self.threshold:
            keep_tokens = int(seq_len * (1 - self.compression_ratio))
            scores[:, :, :self.initial_tokens] = 1.0  
            scores[:, :, -self.recent_tokens:] = 1.0  
            
            remaining_slots = keep_tokens - (self.initial_tokens + self.recent_tokens)
            if remaining_slots > 0:
                middle_scores = attn_weights.mean(dim=1)  
                middle_indices = torch.topk(middle_scores[:, self.initial_tokens:-self.recent_tokens], 
                                         k=remaining_slots, dim=-1).indices
                scores[:, :, middle_indices + self.initial_tokens] = 1.0
        else:
            scores.fill_(1.0)  #
  
        
        return scores