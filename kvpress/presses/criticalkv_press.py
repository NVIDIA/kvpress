# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    CriticalKV: Two-stage compression with output projection weighting.
    
    Based on CriticalKV (https://arxiv.org/abs/2502.03805), this method enhances
    existing scoring methods by rescaling their scores using the L1 norm of the
    output projection applied to values (Wo @ values). This provides a more
    accurate estimate of each token's contribution to the final output.
    
    The method works in two stages:
    1. First stage: Select a subset of tokens using the base scoring method
    2. Second stage: Rescale all scores by their output projection magnitude
    3. Final selection: Combine both stages to make final compression decisions
    
    This approach is particularly effective because:
    - It considers not just attention patterns but actual output contributions
    - The two-stage process balances different importance signals
    - It can improve any existing scoring method as a wrapper
    """

    def __init__(self, press: ScorerPress, epsilon: float = 1e-4, first_stage_ratio: float = 0.5):
        """
        Initialize CriticalKV compression method.
        
        Parameters
        ----------
        press : ScorerPress
            The base scoring method to enhance with output projection weighting.
        epsilon : float, default=1e-4
            Small value added for numerical stability when computing score rescaling.
            Prevents division by zero and stabilizes gradients during computation.
        first_stage_ratio : float, default=0.5
            Fraction of the compression budget allocated to the first stage selection.
            
            The first stage selects `compression_ratio * first_stage_ratio * seq_len`
            tokens using the base scoring method. The remaining budget is used in
            the second stage with output projection weighting.
            
            Values should be between 0.0 and 1.0:
            - 0.0: Only use output projection weighting (skip first stage)
            - 0.5: Balance both stages equally (default)
            - 1.0: Only use base scoring method (skip output weighting)
        """
        self.press = press
        self.epsilon = epsilon
        self.first_stage_ratio = first_stage_ratio

        assert isinstance(self.press, ScorerPress), "CriticalKVPress requires a ScorerPress as input"
        if isinstance(self.press, ExpectedAttentionPress) and self.press.use_vnorm:
            logger.warning("use_vnorm should be disabled for CriticalKVPress")

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
    CriticalAdaKV: Combined two-stage compression with adaptive head-wise selection.
    
    Based on CriticalAdaKV (https://arxiv.org/abs/2502.03805), this method combines
    the output projection weighting from CriticalKV with the adaptive head-wise
    compression from AdaKV. This provides both accurate importance estimation
    and head-specific compression adaptation.
    
    The method works by:
    1. Computing base importance scores using the underlying scorer
    2. Applying two-stage selection with output projection weighting
    3. Performing adaptive head-wise compression with safeguards
    4. Masking less important tokens during attention computation
    
    This combines the best of both approaches:
    - CriticalKV's accurate output-based importance estimation
    - AdaKV's adaptive head-wise compression strategy
    - Safeguards to protect individual attention heads
    """

    press: ScorerPress
    """The underlying scoring method used to evaluate token importance."""
    
    alpha_safeguard: float = 0.20
    """
    Minimum fraction of KV pairs that each head must retain.
    
    This safeguard parameter ensures that no attention head is compressed too
    aggressively, which could severely impact its functionality. Even if a head's
    tokens receive low global importance scores, it will still retain at least
    `alpha_safeguard` fraction of its original tokens.
    
    See AdaKVPress.alpha_safeguard for detailed description.
    """
    
    epsilon: float = 1e-4
    """
    Small value added for numerical stability when computing score rescaling.
    
    This prevents division by zero and stabilizes gradients during the computation
    of output projection magnitudes. Should be a small positive value.
    """
    
    first_stage_ratio: float = 0.5
    """
    Fraction of the compression budget allocated to the first stage selection.
    
    The first stage selects tokens using the base scoring method, while the
    second stage applies output projection weighting. This parameter controls
    the balance between these two selection criteria.
    
    See CriticalKVPress.__init__ for detailed description of this parameter.
    """
    
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
