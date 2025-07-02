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
    
    Based on PyramidKV (https://arxiv.org/abs/2406.02069), this method dynamically
    adjusts KV cache sizes across different transformer layers, following a pyramid
    structure where lower layers retain more tokens and higher layers retain fewer.
    
    The pyramid allocation is based on the observation that:
    - Lower layers capture more basic linguistic patterns and need more context
    - Higher layers focus on more abstract representations and can work with less context
    - A gradual reduction from bottom to top layers optimizes the memory-performance trade-off
    
    The method works by:
    1. Computing a layer-specific budget using the pyramid formula
    2. Allocating more cache capacity to lower layers (larger budgets)
    3. Allocating less cache capacity to higher layers (smaller budgets)
    4. Using SnapKV-style attention computation within each layer's budget
    5. Applying the beta parameter to control the pyramid's steepness
    
    Budget calculation follows the formula:
    ```
    max_capacity_prompt = window_size + query_length * (1 - compression_ratio)
    ```
    
    Key advantages:
    - Optimizes memory usage across the entire model depth
    - Maintains performance by preserving more context where it's most needed
    - Provides fine-grained control over layer-wise compression
    - Builds upon the proven SnapKV attention mechanism
    
    Note: This implementation always applies the specified compression_ratio,
    unlike the original code which may disable compression for short queries.
    """

    compression_ratio: float = 0.0
    """
    Fraction of key-value pairs to remove during compression.
    
    This parameter controls the overall compression level across all layers.
    The actual compression per layer varies according to the pyramid structure,
    but the total compression across all layers averages to this ratio.
    
    See ScorerPress.compression_ratio for detailed description.
    """
    
    window_size: int = 64
    """
    Base window size for attention computation.
    
    This parameter works similarly to SnapKVPress.window_size but is used
    in the pyramid budget calculation. Each layer's effective window size
    may vary based on the pyramid allocation.
    
    See SnapKVPress.window_size for detailed description.
    """
    
    kernel_size: int = 5
    """
    Size of the pooling kernel for attention smoothing.
    
    This parameter is inherited from SnapKVPress and controls the pooling
    operation applied to attention weights for more stable importance scores.
    
    See SnapKVPress.kernel_size for detailed description.
    """
    
    beta: int = 20
    """
    Hyperparameter controlling the pyramid's shape and steepness.
    
    This parameter adjusts how aggressively the cache allocation decreases
    from lower to higher layers. It controls the "steepness" of the pyramid:
    
    Larger beta values:
    - Create a steeper pyramid (more dramatic difference between layers)
    - Allocate much more cache to lower layers, much less to higher layers
    - May provide better compression but could hurt higher-layer performance
    
    Smaller beta values:
    - Create a gentler pyramid (more gradual difference between layers)
    - Provide more balanced allocation across layers
    - May be safer but less aggressive in compression
    
    The default value of 20 provides a good balance for most models and tasks.
    Fine-tuning this parameter can help optimize the trade-off between
    memory usage and model performance for specific use cases.
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
