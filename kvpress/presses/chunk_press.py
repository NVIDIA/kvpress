# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class ChunkPress(BasePress):
    """
    ChunkPress: Uniform compression through independent chunk processing.
    
    This wrapper enhances any ScorerPress by applying compression independently
    to fixed-size chunks of the sequence. Unlike global compression methods that
    may concentrate selection in high-importance regions, ChunkPress ensures
    uniform compression across the entire context by processing each chunk separately.
    
    The method works by:
    1. Dividing the sequence into non-overlapping chunks of fixed size
    2. Applying the underlying ScorerPress to each chunk independently
    3. Compressing each chunk according to the specified compression ratio
    4. Concatenating the compressed chunks to form the final result
    
    This approach was proposed in FINCH (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280)
    and offers several advantages:
    - Ensures uniform compression across the entire sequence
    - Prevents over-concentration of selected tokens in specific regions
    - Maintains balanced representation throughout the context
    - Can improve performance for tasks requiring distributed information
    
    The independent chunk processing helps maintain the structural integrity
    of the input by ensuring that each part of the sequence contributes
    proportionally to the compressed result.
    """

    press: ScorerPress
    """
    The underlying scoring method to apply to each chunk independently.
    
    This can be any ScorerPress subclass (e.g., SnapKVPress, KnormPress, etc.).
    The ChunkPress wrapper will apply this method to each chunk separately,
    ensuring uniform compression across the entire sequence.
    """
    
    chunk_length: int = 1024
    """
    Length of each chunk for independent compression.
    
    The sequence is divided into non-overlapping chunks of this size, and
    each chunk is compressed independently using the underlying ScorerPress.
    This parameter controls the granularity of the uniform compression.
    
    Larger chunk lengths:
    - Allow more context within each chunk for scoring decisions
    - May better preserve local semantic coherence
    - Reduce the number of chunks to process (more efficient)
    
    Smaller chunk lengths:
    - Provide more fine-grained uniform compression
    - Ensure more even distribution of selected tokens
    - May fragment semantic units if too small
    
    The default value of 1024 provides a good balance between semantic
    preservation and uniform compression for most applications. For very
    long sequences, larger chunk sizes may be more appropriate.
    """

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "ChunkPress requires a ScorerPress as input"

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
        indices = []
        for i in range(0, kv_len, self.chunk_length):
            chunk_scores = self.press.score(
                module,
                hidden_states[:, i : i + self.chunk_length],
                keys[:, :, i : i + self.chunk_length],
                values[:, :, i : i + self.chunk_length],
                attentions,
                kwargs,
            )
            chunk_length = keys[:, :, i : i + self.chunk_length].shape[2]
            n_kept = max(1, int(chunk_length * (1 - self.press.compression_ratio)))
            chunk_indices = i + chunk_scores.topk(n_kept, dim=-1).indices
            indices.append(chunk_indices)

        indices = torch.cat(indices, dim=-1)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
