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
    Wrapper class for any ScorerPress.
    First calculates global scores using the ScorerPress,
    then selects tokens chunk by chunk based on these global scores.
    This method was proposed in
    ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference
    https://arxiv.org/abs/2502.00299
    """

    press: ScorerPress
    chunk_length: int = 20

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
