# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from kvpress import CriticalAdaKVPress, CriticalKVPress, KnormPress


@dataclass
class RecordingKnormPress(KnormPress):
    models: list = field(default_factory=list)

    def post_init_from_model(self, model):
        self.models.append(model)


@pytest.fixture(scope="module")
def qwen2_model():
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    assert getattr(config, "head_dim", None) is None
    return Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize("press_cls", [CriticalKVPress, CriticalAdaKVPress])
@torch.no_grad()
def test_critical_presses_run_on_qwen2(qwen2_model, press_cls):
    cache = DynamicCache()
    with press_cls(KnormPress(compression_ratio=0.5))(qwen2_model):
        qwen2_model(torch.randint(0, 128, (1, 32)), past_key_values=cache)
    expected_length = 16 if press_cls is CriticalKVPress else 32
    assert cache.get_seq_length() == expected_length


@pytest.mark.parametrize("press_cls", [CriticalKVPress, CriticalAdaKVPress])
def test_post_init_from_model_is_forwarded(qwen2_model, press_cls):
    inner_press = RecordingKnormPress(compression_ratio=0.5)
    with press_cls(inner_press)(qwen2_model):
        pass
    assert inner_press.models == [qwen2_model]
