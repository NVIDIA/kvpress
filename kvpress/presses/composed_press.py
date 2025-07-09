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

    Applies multiple compression methods in sequence, with each method operating
    on the output of the previous one. Useful for combining complementary approaches
    like sequence + dimension compression.

    Example:
    ```python
    press = ComposedPress([
        SnapKVPress(compression_ratio=0.3),
        ThinKPress(key_channel_compression_ratio=0.2)
    ])
    ```

    ObservedAttentionPress or AdaKVPress are currently not supported.

    Parameters
    ----------
    presses : list[BasePress]
        List of compression methods to apply sequentially.
        Methods are applied in order, with each operating on the compressed output
        of the previous method. Final compression ratio is the product of all ratios.
    """

    presses: list[BasePress]

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
