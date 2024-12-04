# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from kvpress.prunners.base_pruner import BasePruner
from kvpress.prunners.default_pruner import DefaultPruner
from kvpress.scorers.base_scorer import BaseScorer

logger = logging.getLogger(__name__)


@dataclass
class PerLayerCompressionPruner(BasePruner):
    scorer: BaseScorer
    compression_ratios: List[float]

    def __post_init__(self):
        logger.warning(
            "Per layer compression wrapper is an experimental feature and only works with flash attention. "
            "Please make sure that the model uses flash attention."
        )
        self.pruner = DefaultPruner(scorer=self.scorer)

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        self.pruner.compression_ratio = self.compression_ratios[module.layer_idx]
        return self.pruner.forward_hook(module, input, kwargs, output)

    @property
    def compression_ratio(self):
        return sum(self.compression_ratios) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        # While we could set a uniform compression ratio, raise an error to indicate that this may rather be a mistake
        raise NotImplementedError(
            "Setting compression ratio is not supported for PerLayerCompressionPruner. "
            "Please use DefaultPruner for a uniform compression ratio."
        )
