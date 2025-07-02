# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KnormPress(ScorerPress):
    """
    Key norm-based KV cache compression.
    
    Based on the observation from https://arxiv.org/pdf/2406.11430, this method
    prunes key-value pairs based on the L2 norm of their key vectors. The intuition
    is that keys with higher norms tend to have larger magnitudes in the attention
    computation and may be more important for maintaining model performance.
    
    The method works by:
    1. Computing the L2 norm of each key vector
    2. Assigning negative norm values as scores (so higher norms get lower scores)
    3. Pruning tokens with the highest key norms (lowest scores)
    
    This approach is simple, efficient, and doesn't require attention computation,
    making it suitable for scenarios where computational overhead must be minimized.
    
    Key characteristics:
    - Fast computation (only requires norm calculation)
    - No dependency on attention weights or complex patterns
    - Works well as a baseline compression method
    - Can be combined with other methods in ComposedPress
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
        return -keys.norm(dim=-1)
