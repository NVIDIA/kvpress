# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class RandomPress(ScorerPress):
    """
    Random KV cache compression for baseline comparison.
    
    This method randomly selects which key-value pairs to prune, without considering
    their importance or attention patterns. It serves as a baseline for comparing
    the effectiveness of more sophisticated compression methods.
    
    While not practical for production use, RandomPress is valuable for:
    - Establishing baseline performance metrics
    - Validating that other methods perform better than random selection
    - Testing compression infrastructure and pipelines
    - Research and ablation studies
    """

    compression_ratio: float = 0.0
    """
    Fraction of key-value pairs to remove during compression.
    See ScorerPress.compression_ratio for detailed description.
    """
    
    seed: Optional[int] = None
    """
    Random seed for reproducible compression results.
    
    If provided, this seed is used to initialize the random number generator
    before selecting which tokens to prune. This ensures that the same input
    will always result in the same compression pattern.
    
    - None: Use default random state (non-reproducible)
    - Any integer: Use as seed for reproducible results
    
    Setting a seed is recommended for research and debugging purposes.
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
        if self.seed is not None:
            torch.manual_seed(self.seed)
        return torch.rand(*keys.shape[:-1]).to(keys.device, keys.dtype)
