# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class ChunkKVPress(BasePress):
    """
    ChunkKV: Semantic-preserving compression with chunk-wise token selection.
    
    Based on ChunkKV (https://arxiv.org/abs/2502.00299), this method enhances any
    ScorerPress by applying chunk-wise token selection instead of global selection.
    The approach first computes global importance scores, then selects tokens
    chunk by chunk to preserve semantic coherence within local contexts.
    
    The method works by:
    1. Computing global importance scores using the underlying ScorerPress
    2. Dividing the sequence into fixed-size chunks
    3. For each chunk, selecting the most important tokens based on global scores
    4. Ensuring each chunk contributes proportionally to the final compressed sequence
    
    This approach offers several advantages:
    - Preserves local semantic coherence within chunks
    - Prevents over-concentration of selected tokens in specific regions
    - Maintains balanced representation across the entire sequence
    - Can improve performance compared to pure global selection
    
    The chunk-wise selection helps maintain the semantic structure of the input
    by ensuring that important tokens are distributed throughout the sequence
    rather than clustered in high-importance regions.
    """

    press: ScorerPress
    """
    The underlying scoring method used to compute global importance scores.
    
    This can be any ScorerPress subclass (e.g., SnapKVPress, KnormPress, etc.).
    The ChunkKV wrapper will use this method to compute importance scores for
    all tokens, then apply chunk-wise selection based on these scores.
    """
    
    chunk_length: int = 20
    """
    Length of each chunk for token selection.
    
    The sequence is divided into non-overlapping chunks of this size, and tokens
    are selected proportionally from each chunk based on their global importance
    scores. This parameter controls the granularity of the chunk-wise selection.
    
    Larger chunk lengths:
    - Allow more flexibility in token selection within each chunk
    - May better preserve local semantic coherence
    - Reduce the number of chunks to process
    
    Smaller chunk lengths:
    - Provide more fine-grained control over token distribution
    - Ensure more uniform selection across the sequence
    - May fragment semantic units if too small
    
    The default value of 20 provides a good balance between semantic preservation
    and uniform token distribution for most applications.
    """

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "ChunkKVPress requires a ScorerPress as input"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

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

        assert attentions is None, "ChunkPress does not support attentions."

        kv_len = keys.shape[2]

        # 1. Calculate global scores first
        global_scores = self.press.score(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
        )

        # 2. Calculate actual number of complete chunks and remaining tokens
        num_complete_chunks = kv_len // self.chunk_length
        remaining_tokens = kv_len % self.chunk_length

        # If we have no complete chunks, delegate to the underlying scorer press
        if num_complete_chunks == 0:
            return self.press.compress(module, hidden_states, keys, values, attentions, kwargs)

        # Reshape complete chunks for score calculation
        if num_complete_chunks > 0:
            main_scores = global_scores[..., : num_complete_chunks * self.chunk_length]
            main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, self.chunk_length)
            main_chunk_scores = main_chunk_scores.mean(dim=-1)
        else:
            main_chunk_scores = torch.empty((global_scores.shape[0], 0), device=global_scores.device)

        # Handle remaining tokens if any
        if remaining_tokens > 0:
            remaining_scores = global_scores[..., -remaining_tokens:]
            remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
            chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
        else:
            chunk_scores = main_chunk_scores

        # 3. Calculate number of chunks to keep
        n_chunks_kept = max(1, int((num_complete_chunks + (remaining_tokens > 0)) * (1 - self.press.compression_ratio)))
        top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

        # 4. Create indices for selected chunks
        indices = []
        for chunk_idx in top_chunks.indices[0]:
            if chunk_idx < num_complete_chunks:
                # For complete chunks
                start_idx = chunk_idx * self.chunk_length
                chunk_indices = torch.arange(start_idx, start_idx + self.chunk_length, device=keys.device)
            else:
                # For the remaining partial chunk
                chunk_indices = torch.arange(num_complete_chunks * self.chunk_length, kv_len, device=keys.device)
            indices.append(chunk_indices)

        indices = torch.cat(indices).sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)

        # 5. Use gather to collect selected keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
