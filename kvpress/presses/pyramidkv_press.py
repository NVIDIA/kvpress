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
    PyramidKV (https://arxiv.org/abs/2406.02069) dynamically adjusts KV cache sizes across layers,
    allocating more in lower layers and less in higher layers.

    We use the budget calculation formula from:
    https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py#L197

    This implementation always applies compression_ratio,
    instead of disabling compression or keeping fixed budget for short queries like the original code.

    max_capacity_prompt is calculated as:
    max_num + min_num   &= (max_capacity_prompt - window_size) * 2
    total_kvcache_size  &= \frac{(max_num + min_num) * num_layers}{2}
                        &= (max_capacity_prompt - window_size) * num_layers
    total_kvcache_size  &= query_length * num_layers * (1 - compression_ratio)
    max_capacity_prompt &= window_size + query_length * (1 - compression_ratio)
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5
    beta: int = 20  # a hyperparameter to adjust the pyramid’s shape

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
