# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This file provides the default presses for the KVPress library.
Each press is a shortcut to a specific pruner and scorer combination.
Function names are uppercase for backwards compatibility.
"""
from kvpress.prunners.default_pruner import DefaultPruner
from kvpress.prunners.eager_attention_pruner import EagerAttentionPruner
from kvpress.scorers.expected_attention_scorer import ExpectedAttentionScorer
from kvpress.scorers.knorm_scorer import KnormScorer
from kvpress.scorers.observed_attention_scorer import ObservedAttentionScorer
from kvpress.scorers.random_scorer import RandomScorer
from kvpress.scorers.snapkv_scorer import SnapKVScorer
from kvpress.scorers.streaming_llm_scorer import StreamingLLMScorer
from kvpress.scorers.tova_scorer import TOVAScorer


def ExpectedAttentionPress(
    compression_ratio: float = 0.0,
    n_future_positions: int = 512,
    n_sink: int = 4,
    use_covariance: bool = True,
    use_vnorm: bool = True,
):
    return DefaultPruner(
        compression_ratio=compression_ratio,
        scorer=ExpectedAttentionScorer(
            n_future_positions=n_future_positions, n_sink=n_sink, use_covariance=use_covariance, use_vnorm=use_vnorm
        ),
    )


def KnormPress(
    compression_ratio: float = 0.0,
):
    return DefaultPruner(compression_ratio=compression_ratio, scorer=KnormScorer())


def ObservedAttentionPress(
    compression_ratio: float = 0.0,
    output_attentions: bool = False,
):
    if output_attentions:
        return DefaultPruner(compression_ratio=compression_ratio, scorer=ObservedAttentionScorer())
    return EagerAttentionPruner(compression_ratio=compression_ratio, scorer=ObservedAttentionScorer())


def RandomPress(compression_ratio: float = 0.0):
    return DefaultPruner(compression_ratio=compression_ratio, scorer=RandomScorer())


def SnapKVPress(
    compression_ratio: float = 0.0,
    window_size: int = 64,
    kernel_size: int = 5,
):
    return DefaultPruner(
        compression_ratio=compression_ratio, scorer=SnapKVScorer(window_size=window_size, kernel_size=kernel_size)
    )


def StreamingLLMPress(compression_ratio: float = 0.0, n_sink: int = 4):
    return DefaultPruner(compression_ratio=compression_ratio, scorer=StreamingLLMScorer(n_sink=n_sink))


def TOVAPress(compression_ratio: float = 0.0, window_size: int = 1):
    return DefaultPruner(compression_ratio=compression_ratio, scorer=TOVAScorer(window_size=window_size))
