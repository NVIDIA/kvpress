# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class AdaKVPress(BasePress):
    """
    AdaKV: Adaptive head-wise KV cache compression.
    
    Based on AdaKV (https://arxiv.org/abs/2407.11550), this method performs head-specific
    compression by selecting the top-k keys and values across all heads in a layer based
    on importance scores. Unlike uniform compression, AdaKV allows different heads to
    retain different numbers of tokens based on their individual importance patterns.
    
    The method works by:
    1. Computing importance scores for all tokens across all heads
    2. Selecting the globally most important tokens across heads
    3. Applying a safeguard to ensure each head retains a minimum fraction of tokens
    4. Masking less important tokens during attention computation
    
    Key advantages:
    - Adapts compression to each head's specific attention patterns
    - Maintains model performance better than uniform compression
    - Preserves critical tokens that multiple heads find important
    
    This implementation has been reviewed by Yuan Feng, first author of AdaKV.
    """

    press: ScorerPress
    """The underlying scoring method used to evaluate token importance."""
    
    alpha_safeguard: float = 0.20
    """
    Minimum fraction of KV pairs that each head must retain.
    
    This safeguard parameter ensures that no attention head is compressed too
    aggressively, which could severely impact its functionality. Even if a head's
    tokens receive low global importance scores, it will still retain at least
    `alpha_safeguard` fraction of its original tokens.
    
    Values should be between 0.0 and 1.0:
    - 0.0: No safeguard (heads could lose all tokens - not recommended)
    - 0.2: Each head keeps at least 20% of tokens (default)
    - 0.5: Each head keeps at least 50% of tokens (conservative)
    
    Higher values provide more protection for individual heads but may reduce
    overall compression effectiveness. The default of 0.2 provides a good
    balance between compression and head functionality preservation.
    """

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "AdaKVPress requires a ScorerPress as input"
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in [0, 1]"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values

        assert module.config._attn_implementation != "eager", "eager mode not supported"

        # Compute scores
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, q_len = scores.shape

        # Make sure to keep at least alpha * (1 - compression_ratio) KV pairs per head
        n_kept = int(q_len * (1 - self.compression_ratio))  # ScorerPress definition
        n_safe = int(n_kept * self.alpha_safeguard)
        top_indices = torch.topk(scores, n_safe, dim=-1).indices
        scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)

        # Compute bottom-k across heads
        n_pruned = num_key_value_heads * (q_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for more details
        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // q_len
        seq_indices = indices % q_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values
