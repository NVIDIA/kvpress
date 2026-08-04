# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.decoding_press import DecodingPress


@dataclass
class UniformFilteringPress(DecodingPress):
    """
    A simpler variant of FilteringPress that makes uniform (all-or-nothing) keep/skip
    decisions per token instead of per-head decisions.

    Like FilteringPress, this press decides for each new decode token whether to keep
    it in the cache, making it compatible with append-only cache architectures
    (e.g. vLLM's paged KV cache). During prefill it is a no-op.

    The key difference from FilteringPress: each head independently checks whether
    the new token would survive retroactive compression, then a majority vote across
    heads produces a single keep/remove decision. If a majority of heads would evict
    the token, it is removed from the cache entirely. This avoids per-head ragged
    lengths and does not require PaddedTensor.

    Parameters
    ----------
    base_press : ScorerPress
        The scorer press used to compute importance scores for tokens.
    target_compression_ratio : float, default=0.5
        Target fraction of tokens to filter out during decoding.
    compression_interval : int, default=1
        Number of decoding steps between filtering decisions.
    hidden_states_buffer_size : int, default=256
        Maximum number of hidden states to keep before compression.
    """

    target_compression_ratio: float = 0.5
    compression_interval: int = 1
    target_size: int = field(default=1, init=False)

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.target_compression_ratio < 1, "target_compression_ratio must be between 0 and 1"

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_len = keys.shape[2]
        n_kept = max(1, int(k_len * (1 - self.target_compression_ratio)))

        if n_kept >= k_len:
            return keys, values

        scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)

        new_token_scores = scores[:, :, -1]
        threshold = scores.topk(n_kept, dim=-1).values[:, :, -1]
        survives_per_head = new_token_scores >= threshold

        keep = survives_per_head.float().mean(dim=-1) >= 0.5

        if not keep.any():
            keys = keys[:, :, :-1, :].contiguous()
            values = values[:, :, :-1, :].contiguous()

        return keys, values
