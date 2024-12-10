# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.cache_utils import QuantizedCache
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress


@dataclass
class ThinKPress(BasePress):
    """
    ThinK (https://arxiv.org/pdf/2407.21018) compresses the dimensions of the keys, and not the sequence length.
    Hence it can be combined with any other press that compresses the sequence length, e.g.
    press = ThinKPress(compression_ratio=0.5, inner_press=SnapKVPress(compression_ratio=0.5))

    Here, we zero out the pruned dimensions resulting in no memory gain (the shape of the keys remains the same).
    To achieve memory savings, several options can be considered (see https://github.com/NVIDIA/kvpress/pull/18/),
    we might implement them in the future, especially if other similar presses are requested.

    This press has been reviewed by Yuhui Xu, first author of the ThinK paper.
    """

    key_channel_compression_ratio: float = 0.0
    inner_press: Optional[BasePress] = None
    window_size: int = 32

    def compute_window_queries(self, module, hidden_states):
        """
        Re-compute the last window_size query states
        """
        bsz, q_len, _ = hidden_states.shape

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -self.window_size :])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -self.window_size :])
            query_states = qkv[..., : module.num_heads * module.head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, self.window_size, module.num_heads, module.head_dim).transpose(1, 2)

        # Apply RoPE
        position_ids = torch.arange(q_len - self.window_size, q_len).unsqueeze(0).to(query_states.device)
        cos, sin = module.rotary_emb(query_states, position_ids)
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        return query_states

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        We first apply the inner press, then we prune the key dimensions. If other similar presses are requested,
        we will create a dedicated DimensionBasePress class to avoid code duplication.
        """

        # Apply the forward hook of the inner press
        if self.inner_press is not None:
            output = self.inner_press.forward_hook(module, input, kwargs, output)

        # Don't compress if the compression ratio is 0 or this is not pre-filling
        cache = output[-1]
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]
        assert q_len > self.window_size, "Query length should be greater than the window size"

        if (self.key_channel_compression_ratio == 0) or (cache.seen_tokens > q_len):
            return output

        # Get keys
        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
        bsz, num_key_value_heads, q_len, head_dim = keys.shape

        # ThinK specific code
        queries = self.compute_window_queries(module, kwargs["hidden_states"])

        # Compute scores per dimension
        queries_norm = torch.pow(queries, 2).mean(dim=2)  # (bsz, num_heads, head_dim)
        queries_norm = queries_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, module.head_dim).mean(2)
        keys_norm = torch.pow(keys, 2).mean(dim=2)
        key_scores = queries_norm * keys_norm  # (bsz, num_key_value_heads, head_dim)

        # Prune dimensions with the lowest scores by setting them to 0
        n_pruned = int(head_dim * self.key_channel_compression_ratio)
        indices = key_scores.topk(n_pruned, dim=-1, largest=False).indices
        indices = indices.unsqueeze(2).expand(-1, -1, q_len, -1)
        keys = keys.scatter_(-1, indices, 0)

        # Update cache
        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
        else:
            cache.key_cache[module.layer_idx] = keys

        return output

    @property
    def compression_ratio(self):
        compression_ratio = self.key_channel_compression_ratio / 2
        if self.inner_press is not None and hasattr(self.inner_press, "compression_ratio"):
            compression_ratio += self.inner_press.compression_ratio
        return compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(
            "Cannot set the compression ratio of ThinKPress directly. " "Set key_channel_compression_ratio instead."
        )
