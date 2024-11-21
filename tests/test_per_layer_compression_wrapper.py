# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import DynamicCache

from kvpress import KnormPress, apply_per_layer_compression
from tests.fixtures import kv_press_pipeline, unit_test_model  # noqa: F401


@dataclass
class RecordCompressionKnormPress(KnormPress):
    compression_ratio: float = 0
    recorded_compression_ratios: list[float] = field(default_factory=list)

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        self.recorded_compression_ratios.append(self.compression_ratio)
        return super().forward_hook(module, input, kwargs, output)


def test_presses_run(kv_press_pipeline):  # noqa: F811
    press = RecordCompressionKnormPress(compression_ratio=0)
    compression_ratios = [0.1, 0.2]
    wrapped_press = apply_per_layer_compression(press, compression_ratios)
    assert press.compression_ratio is None

    context = "This is a test article. It was written on 2022-01-01."
    questions = ["When was this article written?"]
    answers = kv_press_pipeline(context, questions=questions, press=wrapped_press)["answers"]

    assert len(answers) == 1
    assert isinstance(answers[0], str)

    assert press.recorded_compression_ratios == compression_ratios


def test_compression_rate_1(unit_test_model):  # noqa: F811
    press = RecordCompressionKnormPress(compression_ratio=0)
    compression_ratios = [0.1, 1]
    wrapped_press = apply_per_layer_compression(press, compression_ratios)
    with wrapped_press(unit_test_model):
        input_ids = torch.randint(0, 3_000, (5, 256))
        past_key_values = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    assert past_key_values.key_cache[0].shape == torch.Size([5, 2, 230, 6])
    assert past_key_values.key_cache[1].shape == torch.Size([5, 2, 0, 6])
