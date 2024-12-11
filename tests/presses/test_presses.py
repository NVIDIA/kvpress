# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch
from torch import nn
from transformers import DynamicCache

from kvpress import (
    ComposedPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    TOVAPress,
    AdaSnapKVPress
)
from kvpress.ada_cache import DynamicCacheSplitHeadFlatten
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.think_press import ThinKPress
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


def test_composed_press(unit_test_model):  # noqa: F811
    press1 = KnormPress(compression_ratio=0.5)
    press2 = ThinKPress(key_channel_compression_ratio=0.5, window_size=2)
    composed_press = ComposedPress([press1, press2])
    with composed_press(unit_test_model):
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
            # Check that the press has a compression_ratio attribute
            assert hasattr(press, "compression_ratio")

def test_ada_press():

    from transformers import AutoModelForCausalLM, AutoConfig
    from kvpress.ada_attn import replace_var_flash_attn

    replace_var_flash_attn("llama")
    replace_var_flash_attn("mistral")

    model_kwargs = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16}
    # Flash Attention only supports fp16 or fp16, thus we use fp16 for unit tests
    model = AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test", 
                                                 **model_kwargs).eval().to("cuda:0")
    for cls in [AdaSnapKVPress, ]:
        for compression_ratio in [0.2, 0.4, 0.6, 0.8]:
            press = cls(compression_ratio=compression_ratio, window_size=2)
            with press(model):
                input_ids = model.dummy_inputs["input_ids"].to("cuda:0")
                # run the model with batch size 1
                for i in range(input_ids.size(0)):
                    model(input_ids[i].unsqueeze(0), past_key_values=DynamicCacheSplitHeadFlatten()).past_key_values

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
