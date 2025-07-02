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
    
    This method computes importance scores based on the actual attention weights
    observed during the forward pass. The score for each key-value pair is defined
    as the average attention weight it receives from all query tokens in the sequence.
    
    This approach is related to H2O (https://arxiv.org/abs/2306.14048) and provides
    a direct measure of how much attention each token actually receives during processing.
    
    Requirements:
    - Model must be configured with output_attentions=True
    - Model must use attn_implementation="eager" (not flash attention)
    - Attention weights must be available in the forward pass output
    
    The method works by:
    1. Extracting attention weights from the model's forward pass
    2. Computing average attention received by each key position
    3. Using these averages as importance scores for compression
    """

    compression_ratio: float = 0.0
    """
    Fraction of key-value pairs to remove during compression.
    See ScorerPress.compression_ratio for detailed description.
    """
    
    output_attentions: bool = False
    """
    Whether to return attention weights in the model output.
    
    This parameter controls whether attention weights are included in the
    model's output after compression is applied. The attention weights are
    always needed internally for computing importance scores, but they can
    be removed from the output to save memory.
    
    - True: Include attention weights in model output (uses more memory)
    - False: Remove attention weights from output after scoring (saves memory)
    
    Note: Regardless of this setting, attention weights must be computed
    during the forward pass (requires output_attentions=True in model config).
    """

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
