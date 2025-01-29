# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass
import math

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.snapkv_press import SnapKVPress
logger = logging.getLogger(__name__)


@dataclass
class CriticalScorerPress64(BasePress):
    """
    Default press method for using a score method.
    Any ScorerPress subclass must implement the `score` method that computes a tensor of scores for each key-value pair
    The KV pairs with the lowest scores will be pruned in the `compress` method.
    The cache is uniformly pruned across all heads and layers using the compression_ratio parameter.
    """

    press: ScorerPress
    epsilon: float = 1e-4
    first_stage_ratio: float = 0.5
    
    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value



    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "AdaKVPress requires a ScorerPress as input"
        assert 0 <= self.first_stage_ratio <= 1, "first_stage_ratio should be in [0, 1]"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        

        bsz, num_key_value_heads, q_len, _ = keys.shape

        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        # calculating projected value norm for critical cache selection
        output_w = module.o_proj.weight.transpose(0, 1)
        head_o_proj = output_w.view(module.config.num_attention_heads, module.config.head_dim, module.config.hidden_size)
        values_states = repeat_kv(values, module.num_key_value_groups)

        def vwl1norm(v, w):
            '''
            v.shape = (bs, head_num, seq_len, head_dim)
            w.shape = (head_num, head_dim, dim)
            vw_norm.shape = (bs, head_num, seq_len, 1)
            '''
            v = torch.abs(v).to(torch.float64)
            w = torch.abs(w).to(torch.float64)
            return torch.einsum("abcd,bde->abc", [v, w])

        projected_norm = vwl1norm(values_states, head_o_proj)

        projected_norm = projected_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, q_len).mean(dim=2)


        # calculate the budget for thresholding
        selection_budget = int((1 - self.compression_ratio) * q_len * self.first_stage_ratio)

        top_k_index = torch.topk(scores, selection_budget, sorted=True, dim=-1).indices  

        # stage 2 selection
        scores =  (scores + self.epsilon) * projected_norm

        # merge the two stages
        scores.scatter_(-1, top_k_index, torch.finfo(scores.dtype).max)

        """
        Compute a tensor of scores with shape (bsz, num_key_value_heads, q_len)
        The KV pairs with lowest scores will be pruned in the `compress` method.
        """
        return scores

    # [todo] this method is the same to the one in ScorerPress. Maybe tring to inherit from ScorerPress is a better idea
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = int(q_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
