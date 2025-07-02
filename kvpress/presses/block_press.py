# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class BlockPress(BasePress):
    """
    Block-wise iterative KV cache compression method.
    
    Simulates block prompt processing as described in KeyDiff (https://arxiv.org/abs/2504.15364).
    This method segments the input sequence into non-overlapping blocks and applies compression
    iteratively, maintaining a limited memory overhead for long context inference.
    
    The algorithm works by:
    1. Starting with an empty set of kept tokens
    2. Processing each block of tokens sequentially
    3. For each block, scoring all tokens (kept + current block)
    4. Selecting the top-k tokens to keep for the next iteration
    5. Continuing until all blocks are processed
    
    This approach is particularly effective for very long sequences where global scoring
    would be computationally expensive or memory-intensive.
    """

    press: ScorerPress
    """The underlying scoring method used to evaluate token importance within each block."""
    
    block_size: int = 128
    """
    Size of each processing block in tokens.
    
    Larger blocks:
    - Provide more context for scoring decisions
    - Require more memory during processing
    - May be slower for very long sequences
    
    Smaller blocks:
    - Use less memory per iteration
    - Process faster but with less context
    - May make suboptimal compression decisions
    
    Typical values range from 64 to 512 tokens depending on available memory and sequence length.
    """

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "BlockPress requires a ScorerPress"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.press.compression_ratio == 0:
            return keys, values

        assert attentions is None, "BlockPress does not support attentions."

        bsz, num_key_value_heads, q_len, head_dim = keys.shape

        block_size = self.block_size if self.block_size < q_len else q_len
        n_kept = int(q_len * (1 - self.compression_ratio))

        kept_indices = torch.arange(n_kept, device=keys.device).expand(bsz, num_key_value_heads, -1)

        # Reshape hidden states to match the kept_indices
        states = hidden_states.view(bsz, q_len, num_key_value_heads, -1).transpose(1, 2)

        for i in range(n_kept, q_len, block_size):
            end = min(i + block_size, q_len)
            current_indices = torch.arange(i, end, device=keys.device).expand(bsz, num_key_value_heads, -1)
            current_indices = torch.cat([kept_indices, current_indices], dim=-1)

            # Gather hidden states for the selected indices, then restore the shape
            # Check tests/presses/test_block_press.py for correctness verification of gathered hidden states
            current_states = states.gather(2, current_indices.unsqueeze(-1).expand(-1, -1, -1, states.shape[-1]))
            current_states = current_states.transpose(1, 2).reshape(bsz, -1, hidden_states.shape[-1])

            scores = self.press.score(
                module,
                current_states,
                keys.gather(2, current_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)),
                values.gather(2, current_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)),
                attentions,
                kwargs,
            )
            topk_indices = scores.topk(n_kept, dim=-1).indices
            kept_indices = current_indices.gather(-1, topk_indices)

        kept_indices = kept_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        keys = keys.gather(2, kept_indices).contiguous()
        values = values.gather(2, kept_indices).contiguous()

        return keys, values
