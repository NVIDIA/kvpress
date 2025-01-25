# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class CriticalAdaKVPress(BasePress):
    """
    AdaKV (https://arxiv.org/abs/2407.11550) selects the top-k keys and values among all heads in a layer
    based on the scores, achieving head-specific compression.
    A safeguard is applied to ensure a minimum fraction of KV pairs per head (alpha_safeguard parameter)
    This press has been reviewed by Yuan Feng, first author of AdaKV.
    """

    press: ScorerPress
    alpha_safeguard: float = 0.20

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "AdaKVPress requires a ScorerPress as input"
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in [0, 1]"

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
        # Compute bottom-k across heads
        n_pruned = num_key_value_heads * (q_len - n_kept)

        # budget allocation
        top_indices = torch.topk(scores, n_safe, dim=-1).indices
        budget_scores = scores.scatter(-1, top_indices, torch.finfo(scores.dtype).max)
        budget_scores = budget_scores.reshape(bsz, -1)
        top_indices = torch.topk(budget_scores, n_kept * num_key_value_heads, dim=-1).indices
        top_indices_head_idx = top_indices // q_len
        head_budgts = torch.zeros(num_key_value_heads, device=module.device)
        head_budgts.scatter_add_(0, top_indices_head_idx.flatten(), torch.ones_like(top_indices_head_idx.flatten()))
        print('head_budgts:', head_budgts)
        exit()


        def vwl1norm(v, w):
            '''
            v.shape = (bs, head_num, seq_len, head_dim)
            w.shape = (head_num, head_dim, dim)
            vw_norm.shape = (bs, head_num, seq_len, 1)
            '''
            v = torch.abs(v)
            w = torch.abs(w)
            return torch.einsum("abcd,bde->abc", [v, w])
        # Critical Score refinement


        # pruned indices
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for more details
        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // q_len
        seq_indices = indices % q_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values
