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
    Wrapper class for any ScorerPress.
    Chunks keys and values into chunks of size chunk_length and compresses each chunk separately.
    This ensures that the context is compressed uniformly across the entire context.
    This method was proposed in FINCH: Prompt-guided Key-Value Cache Compression for Large Language Models
    https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280
    """

    press: ScorerPress
    chunk_length: int = 1024

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
