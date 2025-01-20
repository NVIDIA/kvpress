# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class ChunkPress(BasePress):
    """
    Wrapper class for any ScorerPress.
    Chunks keys and values into chunks of size chunk_length and compresses each chunk separately.
    This ensures that the context is compressed uniformly across the entire context.
    """

    press: ScorerPress
    chunk_length: int = 1024

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

        compressed_keys = []
        compressed_values = []

        for i in range(0, kv_len, self.chunk_length):
            keys_to_prune = keys[:, :, i : i + self.chunk_length]
            values_to_prune = values[:, :, i : i + self.chunk_length]
            hidden_states_chunk = hidden_states[:, i : i + self.chunk_length]
            scores = self.press.score(module, hidden_states_chunk, keys_to_prune, values_to_prune, attentions, kwargs)

            n_kept = max(1, int(keys_to_prune.shape[2] * (1 - self.press.compression_ratio)))
            indices = scores.topk(n_kept, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

            # Update cache
            keys_chunk = keys_to_prune.gather(2, indices).contiguous()
            values_chunk = values_to_prune.gather(2, indices).contiguous()

            compressed_keys.append(keys_chunk)
            compressed_values.append(values_chunk)

        keys = torch.cat(compressed_keys, dim=-2)
        values = torch.cat(compressed_values, dim=-2)

        return keys, values
