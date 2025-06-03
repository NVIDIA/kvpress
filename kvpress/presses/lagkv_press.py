# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KnormPress(ScorerPress):
    """Prune KV pairs with lag-relative information (https://arxiv.org/abs/2504.04704)"""
    n_sink: int = 4
    lag_size: int = 128
    # if cross scoring is enabled, the score will not be limited to inside partion
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
        return self._compress_algo(keys, values)
    
    def _get_states_score(self, base_len, in_size, end_idx, value):
        target_v = value[:, :, base_len:end_idx]
        # partition the key-value
        target_v = target_v.view(in_size[0], in_size[1], -1, self.lag_size, in_size[-1])
        ref = target_v[:, :, 1:, :, :]
        v = target_v[:, :, :-1, :, :]
        # lag-relative information
        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)

        score = ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)
        
        return score        
    
    def _compress_algo(self, key_states, value_states):
        base_len = self.n_sink
        
        in_size = key_states.shape
        end_idx = base_len + ((in_size[-2] - base_len) // self.lag_size) * self.lag_size
        tail_len = self.lag_size + in_size[-2] - end_idx
        
        key_score = self._get_states_score(base_len, in_size, end_idx, key_states)
        value_score = self._get_states_score(base_len, in_size, end_idx, value_states)
        # score is in range [0, 1]
        score = (key_score + value_score) / 2
        
        if not self.cross_scoring:
            score = score.argsort(dim=-1).argsort(dim=-1) / self.lag_size
            score = score.to(key_states.dtype)
        # the parts should always keep    
        sink_shape = list(in_size[:-1])
        sink_shape[-1] = self.n_sink
        sink_score = torch.ones(sink_shape, dtype=score.dtype, device=score.device)
        tail_shape = list(in_size[:-1])
        tail_shape[-1] = tail_len
        tail_score = torch.ones(tail_shape, dtype=score.dtype, device=score.device)
        score = torch.cat((sink_score, score.reshape(in_size[0], in_size[1], -1), tail_score), dim=-1)
        return score
