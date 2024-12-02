# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from kvpress.prunners.default_pruner import DefaultPruner

logger = logging.getLogger(__name__)


@dataclass
class PerLayerCompressionPruner(DefaultPruner):
    compression_ratios: List[float] = None

    def __post_init__(self):
        super().__post_init__()
        logger.warning(
            "Per layer compression wrapper is an experimental feature and only works with flash attention. "
            "Please make sure that the model uses flash attention."
        )
        assert self.compression_ratios is not None, "Please provide a list of compression ratios for each layer."

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        self.compression_ratio = self.compression_ratios[module.layer_idx]
        return super().forward_hook(module, input, kwargs, output)
