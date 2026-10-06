# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache, Gemma3ForCausalLM, Gemma3TextConfig

from kvpress import DropKVPress


@torch.no_grad()
def test_dropkv_recomputed_attention_uses_the_attention_scaling():
    """Gemma3 scales the attention logits by query_pre_attn_scalar**-0.5 rather than head_dim**-0.5."""
    config = Gemma3TextConfig(
        vocab_size=1024,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        query_pre_attn_scalar=64,
        sliding_window=64,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Gemma3ForCausalLM(config).eval()
    recorded: dict[str, list] = {"reused": [], "recomputed": []}

    class RecordingDropKVPress(DropKVPress):
        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            recorded["reused"].append(super().score(module, hidden_states, keys, values, attentions, kwargs))
            recorded["recomputed"].append(super().score(module, hidden_states, keys, values, None, kwargs))
            return recorded["reused"][-1]

    with RecordingDropKVPress(compression_ratio=0.5, window_size=8, kernel_size=3)(model):
        model(torch.randint(0, 1024, (1, 40)), past_key_values=DynamicCache(), output_attentions=True)

    assert len(recorded["reused"]) == config.num_hidden_layers
    for reused, recomputed in zip(recorded["reused"], recorded["recomputed"]):
        torch.testing.assert_close(reused, recomputed, rtol=1e-4, atol=1e-4)
