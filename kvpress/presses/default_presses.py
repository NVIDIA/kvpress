# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This file provides the default presses for the KVPress library that are composed of pruner+scorers.
Each press is a shortcut to a specific pruner and scorer combination.
Function names are uppercase for backwards compatibility.
"""
from kvpress.presses.default_press import DefaultPress
from kvpress.presses.eager_attention_press import EagerAttentionPruner
from kvpress.scorers.knorm_scorer import KnormScorer
from kvpress.scorers.observed_attention_scorer import ObservedAttentionScorer
from kvpress.scorers.random_scorer import RandomScorer
from kvpress.scorers.snapkv_scorer import SnapKVScorer
from kvpress.scorers.streaming_llm_scorer import StreamingLLMScorer
from kvpress.scorers.tova_scorer import TOVAScorer


def KnormPress(
    compression_ratio: float = 0.0,
):
    return DefaultPress(compression_ratio=compression_ratio, scorer=KnormScorer())


def ObservedAttentionPress(
    compression_ratio: float = 0.0,
    output_attentions: bool = False,
):
    if output_attentions:
        return DefaultPress(compression_ratio=compression_ratio, scorer=ObservedAttentionScorer())
    return EagerAttentionPruner(compression_ratio=compression_ratio, scorer=ObservedAttentionScorer())


def RandomPress(compression_ratio: float = 0.0):
    return DefaultPress(compression_ratio=compression_ratio, scorer=RandomScorer())


def SnapKVPress(
    compression_ratio: float = 0.0,
    window_size: int = 64,
    kernel_size: int = 5,
):
    return DefaultPress(
        compression_ratio=compression_ratio, scorer=SnapKVScorer(window_size=window_size, kernel_size=kernel_size)
    )


def StreamingLLMPress(compression_ratio: float = 0.0, n_sink: int = 4):
    return DefaultPress(compression_ratio=compression_ratio, scorer=StreamingLLMScorer(n_sink=n_sink))


def TOVAPress(compression_ratio: float = 0.0, window_size: int = 1):
    return DefaultPress(compression_ratio=compression_ratio, scorer=TOVAScorer(window_size=window_size))
