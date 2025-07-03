# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.snapkv_press import SnapKVPress

logger = logging.getLogger(__name__)


@dataclass
class PyramidKVPress(SnapKVPress):
    """
    PyramidKV: Layer-wise adaptive KV cache allocation with pyramid structure.

    Dynamically adjusts KV cache sizes across transformer layers, allocating
    more tokens to lower layers and fewer to higher layers. Based on the
    observation that lower layers need more context while higher layers
    can work with less. Based on PyramidKV (https://arxiv.org/abs/2406.02069).
    """

    compression_ratio: float = 0.0
    """Fraction of key-value pairs to remove during compression."""

    window_size: int = 64
    """Base window size for attention computation, used in pyramid budget calculation."""

    kernel_size: int = 5
    """Size of the pooling kernel for attention smoothing (inherited from SnapKV)."""

    beta: int = 20
    """
    Hyperparameter controlling the pyramid's shape and steepness.
    
    Larger values create steeper pyramids with more dramatic differences between
    layers. Smaller values create gentler, more balanced allocation across layers.
    """

    def get_layer_budget(
        self,
        module: nn.Module,
        q_len: int,
    ) -> int:
        """
        Compute the budget for each layer based on the pyramid shape.
        """
        assert self.beta >= 1, "Beta should >= 1"

        # Ensure the total budget meets the compression_ratio requirements
        max_capacity_prompt = self.window_size + q_len * (1 - self.compression_ratio)

        min_num = (max_capacity_prompt - self.window_size) / self.beta
        max_num = (max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (max_capacity_prompt - self.window_size) * 2 - max_num

        if not (q_len >= max_num >= min_num >= self.window_size):
            # Fall back to SnapKV
            return round(q_len * (1 - self.compression_ratio))

        steps = (max_num - min_num) / (module.config.num_hidden_layers - 1)
        return round(max_num - module.layer_idx * steps)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = self.get_layer_budget(module, q_len)
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
