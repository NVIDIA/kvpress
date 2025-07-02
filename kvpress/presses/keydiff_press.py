# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KeyDiffPress(ScorerPress):
    """
    KeyDiff: Key similarity-based KV cache compression.
    
    Based on KeyDiff (https://arxiv.org/abs/2504.15364), this method evicts tokens
    based solely on key vector similarity. The approach identifies tokens whose
    key representations are most similar to the average key pattern and removes
    them, keeping tokens with more distinctive key vectors.
    
    The method works by:
    1. Computing the average (anchor) key vector across all positions
    2. Normalizing both individual keys and the anchor using L2 normalization
    3. Computing cosine similarity between each key and the anchor
    4. Assigning negative similarity scores (so similar keys get lower scores)
    5. Pruning tokens with highest similarity (most redundant keys)
    
    This approach is based on the intuition that:
    - Keys with similar patterns provide redundant information
    - Distinctive keys are more likely to be important for attention
    - Key similarity is a good proxy for token redundancy
    - The method is computationally efficient (no attention computation needed)
    
    Key characteristics:
    - Fast computation (only requires similarity calculation)
    - No dependency on attention weights or query patterns
    - Focuses on key diversity rather than attention patterns
    - Can identify redundant tokens effectively
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
        anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)
        return -F.cosine_similarity(keys, anchor, dim=-1)
