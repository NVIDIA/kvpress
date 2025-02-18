# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    RandomPress,
    SimLayerKVPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    DuoAttentionPress,
)


class TestDuoAttentionPress(DuoAttentionPress):
    @staticmethod
    def load_attention_pattern(model):
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_key_value_heads
        return 2, 2, np.random.rand(n_layers, n_heads)


# contains all presses to be tested
# kwargs should be ordered easy to hard compression
default_presses = [
    {"cls": TestDuoAttentionPress, "kwargs": [{"head_compression_ratio": 0.2}, {"head_compression_ratio": 0.8}]},
    {"cls": KnormPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {"cls": ExpectedAttentionPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {"cls": RandomPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {"cls": StreamingLLMPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {
        "cls": SnapKVPress,
        "kwargs": [{"compression_ratio": 0.2, "window_size": 2}, {"compression_ratio": 0.8, "window_size": 2}],
    },
    {"cls": TOVAPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {
        "cls": ThinKPress,
        "kwargs": [
            {"key_channel_compression_ratio": 0.2, "window_size": 2},
            {"key_channel_compression_ratio": 0.8, "window_size": 2},
        ],
    },
    {
        "cls": SimLayerKVPress,
        "kwargs": [
            {"lazy_threshold": 0.8, "n_initial": 1, "n_recent": 1, "n_last": 1},
            {"lazy_threshold": 0.2, "n_initial": 1, "n_recent": 1, "n_last": 1},
        ],
    },
]
