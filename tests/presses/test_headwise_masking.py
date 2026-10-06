# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from kvpress import AdaKVPress, CriticalAdaKVPress, DMSPress, KnormPress, RandomPress
from tests import default_presses
from tests.fixtures import unit_test_model_output_attention  # noqa: F401


def make_tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    return LlamaForCausalLM(config).eval()


@pytest.mark.parametrize(
    "make_press",
    [
        lambda: AdaKVPress(KnormPress(compression_ratio=0.5)),
        lambda: CriticalAdaKVPress(KnormPress(compression_ratio=0.5)),
        lambda: default_presses.TestLUKVPress(KnormPress(), compression_ratio=0.5),
        lambda: DMSPress(press=RandomPress(), threshold=0.5, sliding_window_size=0),
    ],
    ids=["adakv", "critical_adakv", "lukv", "dms"],
)
def test_head_wise_presses_reject_eager_attention(unit_test_model_output_attention, make_press):  # noqa: F811
    model = unit_test_model_output_attention
    input_ids = torch.randint(0, 1024, (1, 64), device=model.device)
    with pytest.raises(ValueError, match="eager"):
        with make_press()(model):
            model(input_ids, past_key_values=DynamicCache())


def test_press_initializes_masked_key_indices():
    model = make_tiny_llama()
    assert not hasattr(model.model.layers[0].self_attn, "masked_key_indices")
    with KnormPress(compression_ratio=0.5)(model):
        assert all(layer.self_attn.masked_key_indices is None for layer in model.model.layers)


@torch.no_grad()
def test_press_keeps_existing_masked_key_indices():
    model = make_tiny_llama()
    with AdaKVPress(KnormPress(compression_ratio=0.5))(model):
        model(torch.randint(0, 128, (1, 32)), past_key_values=DynamicCache())
    masked_key_indices = [layer.self_attn.masked_key_indices for layer in model.model.layers]
    assert all(indices is not None for indices in masked_key_indices)

    with KnormPress()(model):
        assert all(
            layer.self_attn.masked_key_indices is indices
            for layer, indices in zip(model.model.layers, masked_key_indices)
        )
