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
    output_attentions : bool, default=False
        Whether to return attention weights in model output.
        Controls whether attention weights are included in output after compression.
        Attention weights are always needed internally for scoring but can be removed
        from output to save memory.
    """

    compression_ratio: float = 0.0
    output_attentions: bool = False

    def __post_init__(self):
        if not self.output_attentions:
            logger.warning(
                "Model will not return attentions in its output to save memory. "
                "Set output_attentions=True if attentions are needed in the output."
            )
        super().__post_init__()

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

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        output = super().forward_hook(module, input, kwargs, output)
        # attentions are needed as input for the hook, but unless the user wants to return them in the output,
        # we can remove them to save memory
        if not self.output_attentions:
            output = (output[0], None)

        return output
