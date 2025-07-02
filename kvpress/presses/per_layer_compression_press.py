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
    
    This wrapper allows applying layer-specific compression ratios using any
    underlying ScorerPress method. Different layers in transformer models may
    have different importance and attention patterns, so layer-specific compression
    can potentially improve the quality-efficiency trade-off.
    
    The method works by:
    1. Taking a base ScorerPress method and a list of compression ratios
    2. Dynamically setting the compression ratio based on the current layer
    3. Applying the underlying scoring method with the layer-specific ratio
    4. Restoring the original compression ratio after processing
    
    Example usage:
    ```python
    # Apply different compression to different layers
    press = PerLayerCompressionPress(
        press=SnapKVPress(),
        compression_ratios=[0.1, 0.2, 0.3, 0.4]  # Increasing compression by layer
    )
    ```
    
    **Important**: This is an experimental feature that only works with flash attention.
    Make sure your model is configured to use flash attention before using this wrapper.
    
    The average compression ratio across all layers is reported as the overall
    compression ratio for this press.
    """

    press: ScorerPress
    """
    The underlying scoring method to apply with layer-specific compression ratios.
    
    This should be any ScorerPress subclass that supports dynamic compression_ratio
    setting. The wrapper will temporarily modify the compression_ratio of this
    press for each layer.
    """
    
    compression_ratios: List[float]
    """
    List of compression ratios to apply to each layer.
    
    The length of this list should match the number of layers in the model.
    Each value should be between 0.0 and 1.0, representing the fraction of
    tokens to remove for that specific layer.
    
    Example patterns:
    - [0.1, 0.2, 0.3, 0.4]: Increasing compression in later layers
    - [0.3, 0.1, 0.1, 0.3]: Higher compression in first and last layers
    - [0.2] * 32: Uniform compression across all layers (equivalent to single ratio)
    
    The ratios allow fine-tuning compression based on layer-specific importance
    or attention patterns observed in the model.
    """

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
