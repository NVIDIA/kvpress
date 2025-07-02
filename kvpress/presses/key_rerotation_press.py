# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KeyRerotationPress(BasePress):
    """
    Key Rerotation: RoPE-aware compression wrapper for maintaining positional encoding.
    
    This wrapper enhances any ScorerPress by applying key rerotation after compression
    to maintain proper RoPE (Rotary Position Embedding) representations. When tokens
    are pruned from the sequence, the remaining tokens need their positional encodings
    adjusted to reflect their new positions in the compressed sequence.
    
    The rerotation process works by:
    1. Applying the underlying ScorerPress to identify tokens to keep
    2. Inverse-rotating the selected keys to remove their original RoPE encoding
    3. Re-applying RoPE with the new consecutive positions after compression
    4. Ensuring proper positional relationships in the compressed sequence
    
    This method is essential for compression techniques that need to maintain
    accurate positional information, and is used in several key-value cache
    compression methods including:
    - SinkCache implementation in Hugging Face's transformers library
    - FINCH: Prompt-guided Key-Value Cache Compression for Large Language Models
    - StreamingLLM with proper positional encoding
    
    The rerotation ensures that attention computations remain accurate after
    compression by preserving the relative positional relationships between
    the remaining tokens.
    
    Key benefits:
    - Maintains accurate positional encoding after token removal
    - Preserves attention quality in compressed sequences
    - Can be applied to any underlying ScorerPress method
    - Essential for models that rely heavily on positional information
    """

    press: ScorerPress
    """
    The underlying scoring method to enhance with key rerotation.
    
    This should be any ScorerPress subclass that identifies which tokens to
    keep during compression. The KeyRerotationPress wrapper will apply the
    scoring method and then rerotate the keys of the selected tokens to
    maintain proper RoPE positional encoding.
    
    The rerotation is applied after the underlying press determines which
    tokens to keep, ensuring that the compressed sequence maintains accurate
    positional relationships for attention computation.
    """

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.press.compression_ratio == 0:
            return keys, values

        # Compute scores from base press
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = int(q_len * (1 - self.press.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = torch.sort(indices, dim=2).values
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        cos, sin = kwargs["position_embeddings"]
        # Rerotate as follows
        #  1. keys = RoPE(W_k * hidden_states)
        #  2. keys_unrotated = RoPE^-1(keys)
        #  3. keys_pruned = prune(keys_unrotated)
        #  4. keys = RoPE(keys_pruned)

        # 2. Inverse of rotation matrix is equivalent to setting sin -> -sin in the equation below
        keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * (-sin.unsqueeze(1)))
        # 3. Prune keys
        keys = keys.gather(2, indices).contiguous()
        # 4. Apply RoPE
        cos, sin = cos[:, :n_kept], sin[:, :n_kept]
        keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * sin.unsqueeze(1))

        values = values.gather(2, indices).contiguous()
        return keys, values
