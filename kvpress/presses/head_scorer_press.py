# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class HeadScorerPress(BasePress):
    """
    Remove the KV pairs with a cumulative score below the score_loss threshold
    This is achieved per head and not per layer.
    Early results failed. RULER 4k, llama 8b and ExpectedAttentionPress:
    - adakv + CR 10%: 95.3
    - adakv + CR 25%: 93.6
    - headscorer + 0.002 : 91.3% for average CR of 7%
    - headscorer + 0.005 : 89.5% for average CR of 13%
    """

    press: ScorerPress
    score_loss: float = 0.0

    def __post_init__(self):
        assert 0.0 <= self.score_loss <= 1.0, "Threshold must be in [0, 1]"
        assert isinstance(self.press, ScorerPress), "Press must be a ScorerPress"

    @property
    def compression_ratio(self):
        return sum(self.compression_ratios) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError("compression ratio cannot be set")

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        assert module.config._attn_implementation != "eager", "eager mode not supported"

        # Compute scores
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        # Normalize scores
        assert scores.min() >= 0, "Scores must be non-negative"
        scores /= scores.sum(dim=-1, keepdim=True)

        # Get indices of the smallest scores
        sorted_scores, indices = torch.sort(scores, dim=-1)
        n_pruned = (sorted_scores.cumsum(dim=-1) < self.score_loss).sum(dim=-1)
        d_range = torch.arange(scores.size(-1), device=scores.device).view(1, 1, -1)
        mask = d_range < n_pruned.unsqueeze(-1)
        batch_indices, head_indices, seq_indices = torch.nonzero(mask, as_tuple=True)
        seq_indices = indices[batch_indices, head_indices, seq_indices]
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)

        # Compute compression ratio
        if module.layer_idx == 0:
            self.compression_ratios = []
        self.compression_ratios.append(n_pruned.sum().item() / scores.numel())

        return keys, values
