# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress


@dataclass
class ThinKPress(BasePress):
    """
    ThinK (https://arxiv.org/pdf/2407.21018) compresses the dimensions of the keys, and not the sequence length.
    Hence it can be combined with any other press that compresses the sequence length, e.g.
    press = ComposedPress([SnapKVPress(0.5), ThinKPress(0.5)])

    Here, we zero out the pruned dimensions resulting in no memory gain (the shape of the keys remains the same).
    To achieve memory savings, several options can be considered (see https://github.com/NVIDIA/kvpress/pull/18/),
    we might implement them in the future, especially if other similar presses are requested.

    This press has been reviewed by Yuhui Xu, first author of the ThinK paper.
    """

    key_channel_compression_ratio: float = 0.0
    window_size: int = 32

    def compute_window_queries(self, module, hidden_states, position_embeddings):
        """
        Re-compute the last window_size query states
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -self.window_size :])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -self.window_size :])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, self.window_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -self.window_size :], sin[:, -self.window_size :]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        return query_states

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        If other similar presses are requested, we might create a generic compress method for dimension pruning
        to avoid code duplication.
        """

        if self.key_channel_compression_ratio == 0:
            return keys, values

        # Compute scores per dimension
        bsz, num_key_value_heads, q_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        queries = self.compute_window_queries(module, kwargs["hidden_states"], kwargs["position_embeddings"])
        queries_norm = torch.pow(queries, 2).mean(dim=2)  # (bsz, num_heads, head_dim)
        queries_norm = queries_norm.view(bsz, num_key_value_heads, num_key_value_groups, module.head_dim).mean(2)
        keys_norm = torch.pow(keys, 2).mean(dim=2)
        key_scores = queries_norm * keys_norm  # (bsz, num_key_value_heads, head_dim)

        # Prune dimensions with the lowest scores by setting them to 0
        n_pruned = int(head_dim * self.key_channel_compression_ratio)
        indices = key_scores.topk(n_pruned, dim=-1, largest=False).indices
        indices = indices.unsqueeze(2).expand(-1, -1, q_len, -1)
        keys = keys.scatter_(-1, indices, 0)

        return keys, values

    @property
    def compression_ratio(self):
        return self.key_channel_compression_ratio / 2

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")
