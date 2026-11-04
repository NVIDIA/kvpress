# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from kvpress import AdaKVPress, CAMPress, KnormPress, SnapKVPress


def test_cam_rejects_adakv_base_press():
    with pytest.raises(AssertionError, match="requires a ScorerPress"):
        CAMPress(base_press=AdaKVPress(SnapKVPress()))


@torch.no_grad()
def test_cam_accumulates_attention_in_float32():
    press = CAMPress(base_press=KnormPress(), compression_interval=10**6, target_size=4)
    module = types.SimpleNamespace(layer_idx=0)
    cache = DynamicCache()
    keys = torch.zeros(1, 1, 8, 4, dtype=torch.bfloat16)
    cache.update(keys, keys.clone(), layer_idx=0)
    kwargs = {
        "hidden_states": torch.zeros(1, 1, 4, dtype=torch.bfloat16),
        "past_key_values": cache,
        "cache_position": torch.tensor([8]),
        "position_embeddings": (torch.ones(1, 1, 4), torch.zeros(1, 1, 4)),
    }
    attentions = torch.full((1, 1, 1, 8), 2e-3, dtype=torch.bfloat16)
    for _ in range(2000):
        press.forward_hook(module, [], kwargs, (None, attentions))
    torch.testing.assert_close(press._running_attn_sum[0], torch.full((1, 1, 8), 4.0), rtol=1e-2, atol=0)


@torch.no_grad()
def test_cam_compresses_half_precision_caches():
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    cache = DynamicCache()
    with CAMPress(base_press=KnormPress(), compression_interval=2, target_size=12)(model):
        model.generate(
            torch.randint(0, 128, (1, 16)),
            past_key_values=cache,
            max_new_tokens=5,
            min_new_tokens=5,
            do_sample=False,
        )
    assert cache.get_seq_length() == 12
