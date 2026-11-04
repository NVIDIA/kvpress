# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from kvpress import FastKVzipPress
from kvpress.presses import fastkvzip_press
from kvpress.presses.fastkvzip_press import FastKVzipGate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a second device")
def test_fastkvzip_gathers_the_scores_of_layers_on_other_devices(monkeypatch):
    model = AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test").eval()
    config = model.config
    ngroup = config.num_attention_heads // config.num_key_value_heads
    gates = [
        FastKVzipGate(idx, config.hidden_size, config.num_key_value_heads, ngroup, model.dtype, sink=16)
        for idx in range(config.num_hidden_layers)
    ]
    monkeypatch.setattr(
        fastkvzip_press, "get_gate_weight", lambda name: ([gate.state_dict() for gate in gates], "q_sink16.pt")
    )

    context_len, press = 64, FastKVzipPress(compression_ratio=0.5)
    with press(model):
        model(torch.randint(0, 1024, (1, context_len)), past_key_values=DynamicCache())
        # Scores are computed on the device of each layer
        press.score_val[-1] = press.score_val[-1].cuda()

    assert press.score_val.device == model.device
    n_masked = sum(len(layer.self_attn.masked_key_indices[0]) for layer in model.model.layers)
    assert n_masked == int(config.num_hidden_layers * config.num_key_value_heads * context_len * 0.5)
