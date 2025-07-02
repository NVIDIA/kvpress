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
    
    Randomly selects which key-value pairs to prune, without considering
    importance or attention patterns. Useful for establishing baseline
    performance metrics and validating other compression methods.
    """

    compression_ratio: float = 0.0
    """Fraction of key-value pairs to remove during compression."""
    
    seed: Optional[int] = None
    """Random seed for reproducible compression results."""
    
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
