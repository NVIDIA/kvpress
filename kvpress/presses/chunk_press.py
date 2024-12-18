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
    Can be utilized to compress the key-value cache in chunks.
    Use kv-press-text-generation with chunk=chunk_size to enable this press.
    """

    press: ScorerPress

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

        q_len = hidden_states.shape[1]

        # keys from kev-value cache that have already been used
        keys_to_keep = keys[:, :, :-q_len]
        values_to_keep = values[:, :, :-q_len]
        # new keys and values from the current forward pass, still unpruned
        keys_to_prune = keys[:, :, -q_len:]
        values_to_prune = values[:, :, -q_len:]

        scores = self.press.score(module, hidden_states, keys_to_prune, values_to_prune, attentions, kwargs)

        n_kept = int(q_len * (1 - self.press.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Update cache
        keys = keys_to_prune.gather(2, indices).contiguous()
        values = values_to_prune.gather(2, indices).contiguous()
        if keys_to_keep.shape[-2] > 0:
            keys = torch.cat([keys_to_keep, keys], dim=-2)
            values = torch.cat([values_to_keep, values], dim=-2)
        return keys, values
