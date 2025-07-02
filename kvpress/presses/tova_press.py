# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class TOVAPress(ScorerPress):
    """
    TOVA: Token-wise Optimal Value Attention for KV cache compression.
    
    Based on TOVA (https://arxiv.org/abs/2401.06104), this method uses the attention
    weights of the last token (averaged across all attention heads) to estimate the
    importance of previous key-value pairs. The intuition is that the attention pattern
    of the most recent token provides a good indicator of which historical tokens
    are important for the current context.
    
    The method works by:
    1. Computing attention weights between the last query and all previous keys
    2. Averaging these attention weights across all attention heads
    3. Using the averaged attention as importance scores for compression
    4. Adding back the last token with maximum score to ensure it's never pruned
    
    This approach is particularly effective because:
    - The last token's attention pattern reflects current context needs
    - Averaging across heads provides a robust importance estimate
    - It's computationally efficient (only requires one attention computation)
    - It naturally handles different types of attention patterns
    
    This implementation has been reviewed by Michael Hassid, one of the authors
    of the TOVA paper. Official implementation: 
    https://github.com/schwartz-lab-NLP/TOVA/blob/main/src/tova_cache.py
    """

    compression_ratio: float = 0.0
    """
    Fraction of key-value pairs to remove during compression.
    See ScorerPress.compression_ratio for detailed description.
    """

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        if attentions is not None:
            attn_weights = attentions[..., -1:, :-1]
        else:
            attn_weights = SnapKVPress.compute_window_attention(
                module, hidden_states, keys, 1, kwargs["position_embeddings"]
            )

        # Average across heads and repeat num_key_value_head times
        scores = attn_weights.mean(1)
        scores = scores.repeat(1, keys.shape[1], 1)

        # Add back the last token. Use max score to make sure the window is not pruned.
        # This is a very slight difference from TOVA that don't enforce it, but the
        # last attention weight is usually very high so it should not change the results.
        scores = F.pad(scores, (0, 1), value=scores.max().item())

        return scores
