# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from kvpress import (
    CompactorPress,
    CURPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    ExpectedAttentionStatsPress,
    KeyDiffPress,
    KnormPress,
    KVzapPress,
    KVzipPress,
    LagKVPress,
    LeverageScorePress,
    NonCausalAttnPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SimLayerKVPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)
from kvpress.presses.kvzap_press import KVzapConfig, KVzapModel


class TestDuoAttentionPress(DuoAttentionPress):
    @staticmethod
    def load_attention_pattern(model):
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_key_value_heads
        return 2, 2, np.random.rand(n_layers, n_heads)


class TestKVzapPress(KVzapPress):
    """Test version of KVzapPress that creates a mock model instead of loading from HuggingFace."""

    def post_init_from_model(self, model):
        config = KVzapConfig(
            input_dim=model.config.hidden_size,
            output_dim=model.config.num_key_value_heads,
            hidden_dim=None,  # Use linear model for testing
            n_modules=model.config.num_hidden_layers,
        )
        self.kvzap_model = KVzapModel(config)


# contains all presses to be tested
# kwargs should be ordered easy to hard compression
default_presses = [
    {"cls": TestDuoAttentionPress, "kwargs": [{"head_compression_ratio": 0.2}, {"head_compression_ratio": 0.8}]},
    {"cls": KnormPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {"cls": ExpectedAttentionPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {"cls": ExpectedAttentionStatsPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
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
    {"cls": CURPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {"cls": TestKVzapPress, "kwargs": [{"compression_ratio": 0.2}, {"compression_ratio": 0.8}]},
    {
        "cls": CompactorPress,
        "kwargs": [
            {
                "compression_ratio": 0.5,
                "sink_size_start": 1,
                "sink_size_end": 1,
                "chunk_size": 256,
            },
            {"compression_ratio": 0.8, "sink_size_start": 0, "sink_size_end": 0, "chunk_size": 256},
        ],
    },
    {
        "cls": LeverageScorePress,
        "kwargs": [
            {"compression_ratio": 0.8, "sketch_dimension": 48},
        ],
    },
    {
        "cls": NonCausalAttnPress,
        "kwargs": [
            {
                "compression_ratio": 0.5,
                "chunk_size": 256,
            },
        ],
    },
]
