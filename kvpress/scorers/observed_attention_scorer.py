# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from transformers.utils import logging

from kvpress.scorers.base_scorer import BaseScorer

logger = logging.get_logger(__name__)


@dataclass
class ObservedAttentionScorer(BaseScorer):
    """The observed attention score is defined as the average attention weight over all prompt tokens
    Requires output_attentions=True and attn_implementation="eager" to have access to attentions
    This approach is related to H2O (https://arxiv.org/abs/2306.14048).
    """

    compression_ratio: float = 0.0

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        assert attentions is not None, 'Set output_attentions=True and attn_implementation="eager" to use this hook'
        scores = attentions.sum(2)
        bsz, num_key_value_heads, n_tokens, _ = keys.shape
        n_tokens_in_sum = torch.arange(n_tokens, 0, -1).to(attentions.device, attentions.dtype)
        scores = scores / n_tokens_in_sum
        scores = scores.view(bsz, num_key_value_heads, -1, n_tokens).mean(2)
        return scores
