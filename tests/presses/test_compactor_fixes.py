# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import CompactorPress


@pytest.fixture(scope="module")
def attention_inputs():
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    torch.manual_seed(0)
    module = LlamaForCausalLM(config).eval().model.layers[0].self_attn
    hidden_states = torch.randn(1, 64, config.hidden_size)
    keys, values = torch.randn(2, 1, config.num_key_value_heads, 64, module.head_dim).unbind(0)
    cos, sin = torch.randn(2, 1, 64, module.head_dim).unbind(0)
    return module, hidden_states, keys, values, {"position_embeddings": (cos, sin)}


@torch.no_grad()
def test_compactor_sinks_score_above_every_other_token(attention_inputs):
    module, hidden_states, keys, values, kwargs = attention_inputs
    press = CompactorPress(compression_ratio=0.5, sink_size_start=8, sink_size_end=4)
    scores = press.score(module, hidden_states, keys, values, None, kwargs)
    sinks = torch.cat([scores[..., :8], scores[..., -4:]], dim=-1)
    assert sinks.min() > scores[..., 8:-4].max()
