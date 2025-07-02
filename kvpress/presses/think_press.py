# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.phi3.modeling_phi3 import Phi3Attention

from kvpress.presses.base_press import BasePress


@dataclass
class ThinKPress(BasePress):
    """
    ThinK: Channel-wise key compression for transformer attention.
    
    Based on ThinK (https://arxiv.org/pdf/2407.21018), this method compresses the dimensions
    of the keys rather than the sequence length. Unlike other methods that remove entire
    tokens, ThinK reduces the dimensionality of each key vector by zeroing out less
    important channels.
    
    This approach is complementary to sequence-length compression methods and can be
    combined with them for even greater compression. For example:
    ```python
    press = ComposedPress([SnapKVPress(0.5), ThinKPress(0.5)])
    ```
    
    Key characteristics:
    - Compresses key dimensions, not sequence length
    - Can be combined with other compression methods
    - Uses recent query patterns to identify important key channels
    - Currently zeros out pruned dimensions (no memory savings in this implementation)
    
    Note: This implementation zeros out pruned dimensions rather than removing them,
    so memory usage remains the same while computation may be reduced.
    
    This press has been reviewed by Yuhui Xu, first author of the ThinK paper.
    """

    key_channel_compression_ratio: float = 0.0
    """
    Fraction of key channels (dimensions) to remove during compression.
    
    Must be between 0.0 and 1.0:
    - 0.0: No compression (keep all key dimensions)
    - 0.3: Remove 30% of key channels (keep 70%)
    - 0.7: Remove 70% of key channels (keep 30%)
    
    Unlike sequence compression, this affects the dimensionality of each key vector
    rather than the number of tokens. Higher values result in more aggressive
    dimensional reduction but may impact attention quality.
    """
    
    window_size: int = 32
    """
    Number of recent tokens to use for computing key channel importance.
    
    This parameter determines how many of the most recent query tokens are used
    to compute attention patterns with keys, which helps identify which key
    channels are most important for recent attention computations.
    
    Larger window sizes:
    - Provide more comprehensive patterns for channel importance
    - May better capture which dimensions are consistently important
    - Require more computation for the importance analysis
    
    Smaller window sizes:
    - Are more computationally efficient
    - Focus on very recent attention patterns
    - May miss some important channel dependencies
    
    Default value of 32 provides a good balance between accuracy and efficiency.
    """

    def compute_window_queries(self, module, hidden_states, position_embeddings):
        """
        Re-compute the last window_size query states
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim

        # Get last self.window_size queries
        if isinstance(module, Phi3Attention):
            qkv = module.qkv_proj(hidden_states[:, -self.window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        elif hasattr(module, "q_proj"):
            # Assume Llama-like attention layer
            query_states = module.q_proj(hidden_states[:, -self.window_size:])
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, self.window_size, num_heads, head_dim).transpose(1, 2)

        # Support for Qwen3 and Gemma3 QK norm
        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            query_states = module.q_norm(query_states)

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
