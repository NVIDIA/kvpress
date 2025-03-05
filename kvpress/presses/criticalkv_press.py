# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import torch
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.base_press import BasePress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


class CriticalKVPress(ScorerPress):
    """
    CriticalKV (https://arxiv.org/abs/2502.03805) rescales the scores of a ScorerPress by
    the L1 norm of Wo @ values
    """

    def __init__(self, press: ScorerPress, epsilon: float = 1e-4, first_stage_ratio: float = 0.5):
        self.press = press
        self.epsilon = epsilon
        self.first_stage_ratio = first_stage_ratio

        assert isinstance(self.press, ScorerPress), "CriticalAdaKVPress requires a ScorerPress as input"
        if isinstance(self.press, ExpectedAttentionPress) and self.press.use_vnorm:
            logger.warning("use_vnorm should be disabled for CriticalAdaKVPress")

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @staticmethod
    def vwl1norm(values, module):
        bsz, num_key_value_heads, q_len, _ = values.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        Wo = module.o_proj.weight.transpose(0, 1)
        Wo = Wo.view(module.config.num_attention_heads, module.config.head_dim, module.config.hidden_size)
        V = repeat_kv(values, num_key_value_groups)

        # We use head-wise computation instead of direct matmul to reduce the memory usage of WoV.
        # Future kernel fusion optimization could eliminate this intermediate variables to enhance performance.
        head_WoV_norm_list = []
        for head in range(V.size(1)):
            head_WoV = V[:, head, :, ...].matmul(Wo[head, ...].unsqueeze(0))
            head_WoV_norm = torch.norm(head_WoV, p=1, dim=-1)
            head_WoV_norm_list.append(head_WoV_norm)

        # b_size, num_heads, q_len , k_len
        WoV_norm = torch.stack(head_WoV_norm_list, dim=1)
        WoV_norm = WoV_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, q_len).mean(dim=2)
        return WoV_norm

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        # Stage 1
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        q_len = keys.shape[2]
        selection_budget = int((1 - self.compression_ratio) * q_len * self.first_stage_ratio)
        top_k_index = torch.topk(scores, selection_budget, sorted=True, dim=-1).indices

        # Stage 2
        projected_norm = self.vwl1norm(values, module)
        scores = (scores + self.epsilon) * projected_norm

        # Merge the two stages
        scores.scatter_(-1, top_k_index, torch.finfo(scores.dtype).max)

        return scores


@dataclass
class CriticalAdaKVPress(BasePress):
    """
    CriticalAdaKV (https://arxiv.org/abs/2502.03805) rescales the scores of a ScorerPress by
    the L1 norm of Wo @ values and combines it with AdaKV (https://arxiv.org/abs/2407.11550).
    """

    press: ScorerPress
    alpha_safeguard: float = 0.20
    epsilon: float = 1e-4
    first_stage_ratio: float = 0.5

    def __post_init__(self):
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in 0, 1]"
        assert isinstance(self.press, ScorerPress), "CriticalAdaKVPress requires a ScorerPress as input"
        if isinstance(self.press, ExpectedAttentionPress) and self.press.use_vnorm:
            logger.warning("use_vnorm should be disabled for CriticalAdaKVPress")

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):

        if self.compression_ratio == 0:
            return keys, values

        assert module.config._attn_implementation != "eager", "eager mode not supported"

        # Compute scores
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, q_len = scores.shape

        # Make sure to keep at least alpha * (1 - compression_ratio) KV pairs per head
        n_kept = int(q_len * (1 - self.compression_ratio))  # ScorerPress definition
        n_safe = int(n_kept * self.alpha_safeguard)
        top_indices = torch.topk(scores, n_safe, dim=-1).indices
        scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)

        ############################
        # Start of CriticalKV code #
        ############################

        # Budget allocation
        budget_scores = scores.scatter(-1, top_indices, torch.finfo(scores.dtype).max)
        budget_scores = budget_scores.reshape(bsz, -1)
        top_indices = torch.topk(budget_scores, n_kept * num_key_value_heads, dim=-1).indices
        top_indices_head_idx = top_indices // q_len
        head_budgets = torch.zeros(num_key_value_heads, device=keys.device, dtype=torch.int64)
        head_budgets.scatter_add_(0, top_indices_head_idx.flatten(), torch.ones_like(top_indices_head_idx.flatten()))

        # Stage 1
        head_selection_budget_1st = (head_budgets * self.first_stage_ratio).to(torch.int64).tolist()
        top_k_index = torch.topk(scores, max(head_selection_budget_1st), sorted=True, dim=-1).indices
        for head_idx in range(num_key_value_heads):
            phase1_budget = head_selection_budget_1st[head_idx]
            scores[:, head_idx, :].scatter_(-1, top_k_index[:, head_idx, :phase1_budget], torch.finfo(scores.dtype).max)

        # Stage 2
        projected_norm = CriticalKVPress.vwl1norm(values, module)
        scores = (scores + self.epsilon) * projected_norm
        top_k_index = torch.topk(scores, max(head_budgets), sorted=True, dim=-1).indices
        for head_idx in range(num_key_value_heads):
            budget = head_budgets[head_idx]
            scores[:, head_idx, :].scatter_(-1, top_k_index[:, head_idx, :budget], torch.finfo(scores.dtype).max)

        ##########################
        # End of CriticalKV code #
        ##########################

        # Compute bottom-k across heads
        n_pruned = num_key_value_heads * (q_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for more details
        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // q_len
        seq_indices = indices % q_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values
