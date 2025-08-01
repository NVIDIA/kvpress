# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from kvpress import (
    DuoAttentionPress,
    ExpectedAttentionPress,
    KeyDiffPress,
    KnormPress,
    KVzipPress,
    LagKVPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SimLayerKVPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
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
    {"cls": QFilterPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
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
    {
        "cls": PyramidKVPress,
        "kwargs": [{"compression_ratio": 0.2, "window_size": 2}, {"compression_ratio": 0.8, "window_size": 2}],
    },
    {
        "cls": LagKVPress,
        "kwargs": [
            {"compression_ratio": 0.5, "n_sink": 16, "lag_size": 128},
            {"compression_ratio": 0.8, "n_sink": 16, "lag_size": 128},
        ],
    },
    {"cls": KeyDiffPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {
        "cls": KVzipPress,
        "kwargs": [{"compression_ratio": 0.5, "layerwise": False}, {"compression_ratio": 0.8, "layerwise": True}],
    },
]
