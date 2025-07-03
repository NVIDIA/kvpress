# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import inspect
import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class PerLayerCompressionPress(BasePress):
    """
    Per-layer compression: Apply different compression ratios to different layers.

    Wrapper that applies layer-specific compression ratios using any underlying
    ScorerPress method. Different layers may have different importance patterns,
    so layer-specific compression can improve quality-efficiency trade-offs.

    **Important**: Experimental feature that only works with flash attention.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method to apply with layer-specific compression ratios.
    compression_ratios : List[float]
        List of compression ratios to apply to each layer.
        Length should match number of model layers. Each value between 0.0-1.0
        represents fraction of tokens to remove for that layer.
    """

    press: ScorerPress
    compression_ratios: List[float]

    def __post_init__(self):
        logger.warning(
            "Per layer compression wrapper is an experimental feature and only works with flash attention. "
            "Please make sure that the model uses flash attention."
        )
        assert (
            "compression_ratio"
            in inspect.signature(
                self.press.__init__  # type:ignore[misc]
            ).parameters
        ), f"compression_ratio can't be set in the provided press: {self.press.__class__}"
        assert isinstance(self.press, ScorerPress), "PerLayerCompressionPress requires a ScorerPress as input"

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        original_compression_ratio = self.press.compression_ratio  # type:ignore[index]
        self.press.compression_ratio = self.compression_ratios[module.layer_idx]  # type:ignore[index]
        output = self.press.forward_hook(module, input, kwargs, output)
        self.press.compression_ratio = original_compression_ratio  # type:ignore[attr-defined]
        return output

    @property
    def compression_ratio(self):
        return sum(self.compression_ratios) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")
