# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class AdaKVPress(BasePress):
    """
    AdaKV (https://arxiv.org/abs/2407.11550) selects the top-k keys and values among all heads in a layer
    based on the scores, achieving head-specific compression.
    A safeguard is applied to ensure a minimum fraction of KV pairs per head (alpha_safeguard parameter)
    """

    scorer: ScorerPress
    alpha_safeguard: float = 0.20

    @property
    def compression_ratio(self):
        return self.scorer.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.scorer.compression_ratio = value

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values

        assert module.config._attn_implementation != "eager", "eager mode not supported"
        assert isinstance(self.scorer, ScorerPress), "AdaKVPress requires a ScorerPress as input"

        # Compute scores
        scores = self.scorer.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, q_len = scores.shape

        # Make sure to keep at least alpha * (1 - compression_ratio) KV pairs per head
        n_kept = int(q_len * (1 - self.compression_ratio))  # ScorerPress definition
        n_safe = int(n_kept * self.alpha_safeguard)
        top_indices = torch.topk(scores, n_safe, dim=-1).indices
        scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)

        # Compute bottom-k across heads
        n_pruned = num_key_value_heads * (q_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        # Save indices for attention patching in the module
        module.masked_key_indices = (torch.arange(bsz).repeat_interleave(n_pruned), indices // q_len, indices % q_len)
        return keys, values
