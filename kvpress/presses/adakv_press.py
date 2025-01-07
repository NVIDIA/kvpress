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
    """

    scorer: ScorerPress

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

        # Compute scores
        scores = self.scorer.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, q_len = scores.shape

        # TODO: understand how floor_alpha works

        # Compute bottom-k across heads
        n_pruned = int(num_key_value_heads * q_len * self.compression_ratio)
        indices = torch.topk(-scores.view(bsz, -1), n_pruned, dim=1).indices.flatten()

        # Save indices for attention patching in the module
        module.indices = (torch.arange(bsz).repeat_interleave(n_pruned), indices // q_len, indices % q_len)

        # Return keys and values without compression (achieved with the attention patch)
        return keys, values
