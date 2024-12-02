# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from transformers.utils import logging

from kvpress.prunners.default_pruner import DefaultPruner

logger = logging.get_logger(__name__)


@dataclass
class EagerAttentionPruner(DefaultPruner):
    """
    This press is used when eager attention is used in the model (i.e. the attention is materialized).
    It is used to calculate the observed attention score.
    """

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        # attentions are needed as input for the hook, but unless the user wants to return them in the output,
        # we can remove them to save memory
        output = super().forward_hook(module, input, kwargs, output)
        logger.warning_once("Model will not return attentions in its output to save memory. ")
        output = list(output)
        output[-2] = None
        output = tuple(output)

        return output
