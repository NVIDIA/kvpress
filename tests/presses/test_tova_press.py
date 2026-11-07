# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import TOVAPress


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("seq_len,expected_cache_len", [(1, 1), (2, 1), (8, 4)])
def test_tova_prefill_preserves_last_token(seq_len, expected_cache_len, attn_implementation):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            _attn_implementation=attn_implementation,
        )
    ).eval()
    input_ids = torch.arange(seq_len).unsqueeze(0)
    press = TOVAPress(compression_ratio=0.5)

    with torch.no_grad():
        reference = model(input_ids, use_cache=True)
        with press(model):
            output = model(input_ids, use_cache=True)

    assert output.past_key_values.get_seq_length() == expected_cache_len
    assert torch.isfinite(output.logits).all()
    torch.testing.assert_close(
        output.past_key_values.layers[0].keys[:, :, 0],
        reference.past_key_values.layers[0].keys[:, :, -1],
    )
