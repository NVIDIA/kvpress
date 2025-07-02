# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.observed_attention_press import ObservedAttentionPress


@dataclass
class ComposedPress(BasePress):
    """
    Composed compression: Chain multiple compression methods sequentially.
    
    This class allows combining multiple compression methods to achieve more
    sophisticated compression strategies. The methods are applied sequentially,
    with each method operating on the output of the previous one.
    
    The composition is particularly useful for:
    - Combining complementary compression approaches (e.g., sequence + dimension compression)
    - Creating multi-stage compression pipelines
    - Experimenting with hybrid compression strategies
    
    Example usage:
    ```python
    # Combine sequence compression with dimensional compression
    press = ComposedPress([
        SnapKVPress(compression_ratio=0.3),  # First reduce sequence length
        ThinKPress(key_channel_compression_ratio=0.2)  # Then reduce key dimensions
    ])
    
    # Multi-stage sequence compression
    press = ComposedPress([
        StreamingLLMPress(compression_ratio=0.4, n_sink=4),  # Window-based pruning
        KnormPress(compression_ratio=0.2)  # Further refinement with norm-based pruning
    ])
    ```
    
    The effective compression ratio is the product of all individual ratios.
    For example, two methods with 0.5 compression ratio each result in 0.75
    overall compression (keeping 25% of original tokens).
    
    Limitations:
    - Cannot include ObservedAttentionPress or AdaKVPress due to implementation constraints
    - Order of composition matters and affects final results
    - Computational overhead increases with number of composed methods
    """

    presses: list[BasePress]
    """
    List of compression methods to apply sequentially.
    
    The methods are applied in the order specified, with each method operating
    on the compressed output of the previous method. The list should contain
    BasePress instances that are compatible with sequential composition.
    
    Restrictions:
    - Cannot include ObservedAttentionPress (requires specific attention handling)
    - Cannot include AdaKVPress (requires head-wise masking incompatible with composition)
    
    The final compression ratio will be the product of all individual compression ratios.
    """

    def __post_init__(self):
        self.compression_ratio = None
        assert not any(
            isinstance(press, (ObservedAttentionPress, AdaKVPress)) for press in self.presses
        ), "ComposedPress cannot contains ObservedAttentionPress or AdaKVPress"

    def forward_hook(self, module, input, kwargs, output):
        self.compression_ratio = 1.0
        for press in self.presses:
            output = press.forward_hook(module, input, kwargs, output)
            self.compression_ratio *= press.compression_ratio  # type: ignore
        return output
