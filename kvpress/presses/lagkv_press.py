# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class LagKVPress(ScorerPress):
    """
    Prune KV pairs with lag-relative information (https://arxiv.org/abs/2504.04704)

    Args:
        n_sink (`int`):
            The number of sink tokens.
        lag_size (`int`):
            The size of the partition. The subsequent partition will serve as a reference for the prior one.
        cross_scoring (`bool`):
            (experimental) if cross scoring is enabled, the score will not be limited to inside partion.
            Since the score is totally normalized, it's possible use it to allocating KV among heads.
            This switch will be useful for Press Wrapper like AdaKVPress.
    """
    n_sink: int = 4
    lag_size: int = 128
    cross_scoring: bool = False

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        bsz, num_key_value_heads, q_len, d = keys.shape
        if q_len < self.n_sink + 2 * self.lag_size:
            # no compression
            score = torch.ones((bsz, num_key_value_heads, q_len),
                               dtype=keys.dtype, device=keys.device)
            if q_len > self.n_sink:
                # make sure the sliding part will be selected.
                score[:, :, self.n_sink:] = (torch.arange(q_len - self.n_sink, device=keys.device)
                                             / (q_len - self.n_sink)
                                             ).to(keys.dtype)
            return score

        end_idx = self.n_sink + ((q_len - self.n_sink) // self.lag_size) * self.lag_size
        tail_len = self.lag_size + q_len - end_idx

        key_score = self._get_states_score(
            keys[:, :, self.n_sink:end_idx].view(bsz, num_key_value_heads, -1, self.lag_size, d))
        value_score = self._get_states_score(
            values[:, :, self.n_sink:end_idx].view(bsz, num_key_value_heads, -1, self.lag_size, d))
        # score is in range [0, 1]
        score = (key_score + value_score) / 2

        if not self.cross_scoring:
            score = score.argsort(dim=-1).argsort(dim=-1) / self.lag_size
            score = score.to(keys.dtype)
        # the parts should always keep
        sink_shape = (bsz, num_key_value_heads, self.n_sink)
        sink_score = torch.ones(sink_shape, dtype=score.dtype, device=score.device)
        tail_shape = (bsz, num_key_value_heads, tail_len)
        tail_score = torch.ones(tail_shape, dtype=score.dtype, device=score.device)
        score = torch.cat((sink_score, score.reshape(bsz, num_key_value_heads, -1), tail_score), dim=-1)
        return score

    def _get_states_score(self, target_v):
        """evaluate the scores of keys and values for each token"""
        ref = target_v[:, :, 1:, :, :]
        v = target_v[:, :, :-1, :, :]
        # lag-relative information
        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)

        score = ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)
        return score
