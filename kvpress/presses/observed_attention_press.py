# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.scorers.observed_attention_scorer import ObservedAttentionScorer

logger = logging.getLogger(__name__)


@dataclass
class ObservedAttentionPress(ScorerPress):
    """
    This pruner can be used when eager attention is used in the model (i.e. the attention is materialized).
    It will not return attentions in its output to save memory.
    """

    scorer: ObservedAttentionScorer = field(default_factory=ObservedAttentionScorer, init=False)
    compression_ratio: float = 0.0
    output_attentions: bool = False

    def __post_init__(self):
        self.scorer = ObservedAttentionScorer()
        if not self.output_attentions:
            logger.warning(
                "Model will not return attentions in its output to save memory. Please use DefaultPruner if"
                " attentions are needed in the output."
            )
        super().__post_init__()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        output = super().forward_hook(module, input, kwargs, output)
        # attentions are needed as input for the hook, but unless the user wants to return them in the output,
        # we can remove them to save memory
        if not self.output_attentions:
            output = list(output)
            output[-2] = None
            output = tuple(output)

        return output
