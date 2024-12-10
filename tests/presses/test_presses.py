# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch
from torch import nn
from transformers import DynamicCache

from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    TOVAPress,
)
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.think_press import ThinKPress
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


def test_think_inner_press(unit_test_model):  # noqa: F811
    press = ThinKPress(key_channel_compression_ratio=0.5, window_size=2, inner_press=KnormPress(0.5))
    with press(unit_test_model):
        input_ids = unit_test_model.dummy_inputs["input_ids"]
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_presses_run(unit_test_model):  # noqa: F811
    for cls in [KnormPress, ExpectedAttentionPress, RandomPress, StreamingLLMPress, SnapKVPress, TOVAPress, ThinKPress]:
        for compression_ratio in [0.2, 0.4, 0.6, 0.8]:
            if cls == ThinKPress:
                press = cls(key_channel_compression_ratio=compression_ratio, window_size=2)
            else:
                press = cls(compression_ratio=compression_ratio)
            if cls in [SnapKVPress]:
                press.window_size = 2
            with press(unit_test_model):
                input_ids = unit_test_model.dummy_inputs["input_ids"]
                unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_presses_run_observed_attention(unit_test_model_output_attention):  # noqa: F811
    for cls in [ObservedAttentionPress]:
        for compresion_ratio in [0.2, 0.4, 0.6, 0.8]:
            press = cls(compression_ratio=compresion_ratio)
            with press(unit_test_model_output_attention):
                input_ids = unit_test_model_output_attention.dummy_inputs["input_ids"]
                unit_test_model_output_attention(input_ids, past_key_values=DynamicCache()).past_key_values


@dataclass
class StoreKnormPress(ScorerPress):

    def __post_init__(self):
        self.scores = []

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        scores = -keys.norm(dim=-1)
        self.scores.append(scores)
        return scores


@torch.no_grad()
def test_presses_keep_highest_score(unit_test_model):  # noqa: F811
    """
    Test that kept keys are those with the highest score
    """
    for compresion_ratio in [0.0, 0.2, 0.4, 0.6, 0.8]:
        press = StoreKnormPress(compression_ratio=compresion_ratio)
        with press(unit_test_model):
            input_ids = torch.randint(0, 3_000, (5, 256))
            past_key_values = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

        for scores, key in zip(press.scores, past_key_values.key_cache):
            max_scores = -key.norm(dim=-1)
            for batch_idx in range(scores.shape[0]):
                for head_idx in range(scores.shape[1]):
                    assert torch.allclose(
                        scores[batch_idx, head_idx].sort().values[-max_scores.shape[-1] :],
                        max_scores[batch_idx, head_idx].sort().values,
                    )
