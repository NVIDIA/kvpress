# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import CompactorPress, LeverageScorePress, NonCausalAttnPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_compactor_press(unit_test_model):  # noqa: F811
    for press in [
        CompactorPress(0.5, sink_size_start=0, sink_size_end=0),
        CompactorPress(0.2, sink_size_start=8, sink_size_end=4),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


@pytest.mark.parametrize("seq_len,expected_cache_len", [(1, 1), (8, 4), (12, 6)])
def test_compactor_short_prefill(seq_len, expected_cache_len):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    ).eval()
    press = CompactorPress(compression_ratio=0.5)
    input_ids = torch.arange(seq_len).unsqueeze(0)

    with torch.no_grad(), press(model):
        output = model(input_ids, use_cache=True)

    assert output.past_key_values.get_seq_length() == expected_cache_len
    assert torch.isfinite(output.logits).all()


def test_leverage_press(unit_test_model):  # noqa: F811
    for press in [
        LeverageScorePress(0.5, sketch_dimension=48),
        LeverageScorePress(0.5, sketch_dimension=64),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


def test_non_causal_attn_press(unit_test_model):  # noqa: F811
    for press in [
        NonCausalAttnPress(0.5, chunk_size=128),
        NonCausalAttnPress(0.5, chunk_size=256),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)
