# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from transformers import DynamicCache

from kvpress import (
    AdaKVPress,
    ChunkKVPress,
    ChunkPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    KeyRerotationPress,
    KnormPress,
    ObservedAttentionPress,
    ScorerPress,
    SnapKVPress,
    ThinKPress,
)
from tests.default_presses import default_presses
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


def compute_masked_percentage(module, batch_size, num_key_value_heads, seq_len):
    """
    Compute the percentage of masked indices from module.masked_key_indices.
    """
    if module.masked_key_indices is None:
        return 0.0

    batch_indices, head_indices, seq_indices = module.masked_key_indices
    num_masked = len(batch_indices)
    total_positions = batch_size * num_key_value_heads * seq_len
    masked_percentage = num_masked / total_positions
    return masked_percentage


def test_composed_press(unit_test_model):  # noqa: F811
    press1 = KnormPress(compression_ratio=0.5)
    press2 = ThinKPress(key_channel_compression_ratio=0.5, window_size=2)
    composed_press = ComposedPress([press1, press2])
    with composed_press(unit_test_model):
        input_ids = unit_test_model.dummy_inputs["input_ids"]
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_chunk_press(unit_test_model):  # noqa: F811
    press = KnormPress(compression_ratio=0.5)
    for chunk_length in [2, 4, 8, 128]:
        composed_press = ChunkPress(press=press, chunk_length=chunk_length)
        with composed_press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 256))
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache).past_key_values
            assert cache.get_seq_length() == 128


def test_chunkkv_press(unit_test_model):  # noqa: F811
    press = SnapKVPress(compression_ratio=0.5)
    for chunk_length in [2, 4, 8, 128]:
        composed_press = ChunkKVPress(press=press, chunk_length=chunk_length)
        with composed_press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 256))
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache).past_key_values
            assert cache.get_seq_length() == 128


@pytest.mark.parametrize("press_dict", default_presses)
@pytest.mark.parametrize(
    "wrapper_press",
    [None, ComposedPress, KeyRerotationPress, AdaKVPress, ChunkPress, CriticalKVPress, CriticalAdaKVPress],
)
def test_presses_run(unit_test_model, press_dict, wrapper_press):  # noqa: F811
    cls = press_dict["cls"]
    for kwargs in press_dict["kwargs"]:
        press = cls(**kwargs)
        if wrapper_press is not None:
            if hasattr(press, "__post_init_from_model__"):
                press.__post_init_from_model__(unit_test_model)
            if issubclass(wrapper_press, ComposedPress):
                press = ComposedPress(presses=[press])
            elif not isinstance(press, ScorerPress):  # remaining wrapper presses only support ScorerPress
                return
            elif issubclass(wrapper_press, (KeyRerotationPress, AdaKVPress, CriticalKVPress, CriticalAdaKVPress)):
                press = wrapper_press(press=press)
            elif issubclass(wrapper_press, ChunkPress):
                press = ChunkPress(press=press, chunk_length=24)

        # TODO: Handle __post_init_from_model__ differently
        if hasattr(press, "__post_init_from_model__"):
            press.__post_init_from_model__(unit_test_model)
        with press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 128))
            unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values
        # Check that the press has a compression_ratio attribute
        assert hasattr(press, "compression_ratio")

        # Make sure that head compression is working properly
        # Only for those presses that use masked_key_indices
        if unit_test_model.model.layers[0].self_attn.masked_key_indices is not None:
            headwise_compression_ratio = 0.0
            for layer in unit_test_model.model.layers:
                cr = compute_masked_percentage(layer.self_attn, 1, unit_test_model.config.num_key_value_heads, 128)
                headwise_compression_ratio += cr
            print(headwise_compression_ratio / len(unit_test_model.model.layers))
            cumulative_compression_ratio = headwise_compression_ratio / len(unit_test_model.model.layers)
            assert abs(cumulative_compression_ratio - press.compression_ratio) < 1e-2  # tolerate small differences


def test_presses_run_observed_attention(unit_test_model_output_attention):  # noqa: F811
    for cls in [ObservedAttentionPress]:
        for compresion_ratio in [0.2, 0.8]:
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
