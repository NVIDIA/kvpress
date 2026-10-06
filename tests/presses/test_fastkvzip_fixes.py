# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

from kvpress import FastKVzipPress
from kvpress.presses import fastkvzip_press
from kvpress.presses.fastkvzip_press import FastKVzipGate
from tests.fixtures import get_device

MODEL_NAME = "MaxJeblick/llama2-0b-unit-test"


def use_random_gates(monkeypatch, gate_dtype=torch.bfloat16):
    """Replace the gate weights downloaded from the Hub by random ones, stored in `gate_dtype` as published."""
    loaded = []

    def get_gate_weight(model_name):
        loaded.append(model_name)
        config = AutoConfig.from_pretrained(model_name)
        ngroup = config.num_attention_heads // config.num_key_value_heads
        gates = [
            FastKVzipGate(idx, config.hidden_size, config.num_key_value_heads, ngroup, gate_dtype, sink=16)
            for idx in range(config.num_hidden_layers)
        ]
        return [gate.state_dict() for gate in gates], f"{model_name.split('/')[-1]}/q{ngroup}_dim16_sink16.pt"

    monkeypatch.setattr(fastkvzip_press, "get_gate_weight", get_gate_weight)
    return loaded


def run_press(model, press, context_len=64):
    with press(model):
        model(torch.randint(0, 1024, (1, context_len), device=model.device), past_key_values=DynamicCache())


def n_masked(model):
    return sum(len(layer.self_attn.masked_key_indices[0]) for layer in model.model.layers)


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float16])
def test_fastkvzip_bfloat16_gates_on_other_model_dtypes(monkeypatch, model_dtype):
    use_random_gates(monkeypatch, gate_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=model_dtype).eval().to(get_device())
    context_len, press = 64, FastKVzipPress(compression_ratio=0.5)
    run_press(model, press, context_len)

    assert press.gates[0].q_proj.weight.dtype == torch.bfloat16
    config = model.config
    assert n_masked(model) == int(config.num_hidden_layers * config.num_key_value_heads * context_len * 0.5)

