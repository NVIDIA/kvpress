# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class ObservedAttentionPress(ScorerPress):
    """
    Observed attention-based KV cache compression.

    Computes importance scores based on actual attention weights observed during
    forward pass. Score for each key-value pair is the average attention weight
    it receives from all query tokens.

    Requires: output_attentions=True and attn_implementation="eager".

    Related to H2O (https://arxiv.org/abs/2306.14048).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    output_attentions : bool, default=True
        Whether to output the attention weights. Must be set True but we keep it for backward compatibility.
    """

    compression_ratio: float = 0.0
    output_attentions: bool = True

    def __post_init__(self):
        if not self.output_attentions:
            # keep for backward compatibility, remove in version 1.0
            raise ValueError(
                "With transformers >= 4.54, " "ObservedAttentionPress will only work with output_attentions=True"
            )

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
