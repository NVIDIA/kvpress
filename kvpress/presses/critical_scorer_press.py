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


# @dataclass
class CriticalScorerPress(ScorerPress):
    """
    Default press method for using a score method.
    Any ScorerPress subclass must implement the `score` method that computes a tensor of scores for each key-value pair
    The KV pairs with the lowest scores will be pruned in the `compress` method.
    The cache is uniformly pruned across all heads and layers using the compression_ratio parameter.
    """

    def __init__(self, press: ScorerPress, epsilon: float = 1e-4, first_stage_ratio: float = 0.5):
        self.press = press
        self.epsilon = epsilon
        self.first_stage_ratio = first_stage_ratio

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "CriticalScorerPress requires a ScorerPress as input"
        assert 0 <= self.first_stage_ratio <= 1, "first_stage_ratio should be in [0, 1]"
        if hasattr(self.press, "use_vnorm"):
            self.press.use_vnorm = False
            print("use_vnorm is set to False")

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

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
        def vwl1norm(values, module):
            bsz, num_key_value_heads, q_len, _ = values.shape
            output_w = module.o_proj.weight.transpose(0, 1)
            w = output_w.view(module.config.num_attention_heads, module.config.head_dim, module.config.hidden_size)
            v = repeat_kv(values, module.num_key_value_groups)
            head_vw_norm_weight_list = []
            for head in range(v.size(1)):
                head_o_proj_value_states = v[:,head, :,...].matmul(w[head,...].unsqueeze(0))

                head_vw_norm_weight = torch.norm(head_o_proj_value_states, p=1, dim=-1)
                head_vw_norm_weight_list.append(head_vw_norm_weight)
            # b_size, num_heads, q_len , k_len
            norm = torch.stack(head_vw_norm_weight_list, dim=1)
            norm = norm.view(bsz, num_key_value_heads,module.num_key_value_groups, q_len).mean(dim=2)
            return norm.to(v.dtype)

        # calculating projected value norm for critical cache selection
        projected_norm = vwl1norm(values, module)

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

