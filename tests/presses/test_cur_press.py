# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import CURPress


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_local_approximation", [False, True])
@pytest.mark.parametrize("zero_keys, zero_values", [(True, False), (False, True), (True, True)])
def test_cur_scores_remain_finite_with_zero_energy(dtype, use_local_approximation, zero_keys, zero_values):
    keys = torch.zeros(1, 1, 4, 2, dtype=dtype) if zero_keys else torch.ones(1, 1, 4, 2, dtype=dtype)
    values = torch.zeros(1, 1, 4, 2, dtype=dtype) if zero_values else torch.ones(1, 1, 4, 2, dtype=dtype)
    press = CURPress(num_sinks=1, local_window_size=2, use_local_approximation=use_local_approximation)

    scores = press.score(None, None, keys, values, None, {})

    torch.testing.assert_close(scores, torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=dtype))


@pytest.mark.parametrize("use_local_approximation", [False, True])
def test_cur_zero_energy_window_does_not_contaminate_valid_tokens(use_local_approximation):
    keys = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]])
    press = CURPress(num_sinks=1, local_window_size=2, use_local_approximation=use_local_approximation)

    scores = press.score(None, None, keys, keys, None, {})

    torch.testing.assert_close(scores, torch.tensor([[[1.0, 0.0, 0.5, 0.5]]]))


@pytest.mark.parametrize("use_local_approximation", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_cur_preserves_normalization_for_nonzero_energy(use_local_approximation, scale):
    keys = torch.full((1, 1, 4, 2), scale)
    press = CURPress(num_sinks=1, local_window_size=2, use_local_approximation=use_local_approximation)

    scores = press.score(None, None, keys, keys, None, {})

    torch.testing.assert_close(scores, torch.tensor([[[1.0, 0.25, 0.25, 0.25]]]))


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_cur_keeps_valid_llama_cache_instead_of_zero_padding(attn_implementation):
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            pad_token_id=0,
            _attn_implementation=attn_implementation,
        )
    ).eval()
    input_ids = torch.cat([torch.arange(1, 17), torch.zeros(16, dtype=torch.long)]).unsqueeze(0)
    attention_mask = input_ids.ne(0)
    press = CURPress(compression_ratio=0.5)

    with torch.no_grad():
        reference = model(input_ids, attention_mask=attention_mask, use_cache=True).past_key_values.layers[0]
        with press(model):
            compressed = model(input_ids, attention_mask=attention_mask, use_cache=True).past_key_values.layers[0]

    # The 16 padding entries have zero energy; all 16 real entries should survive.
    expected_keys = reference.keys[:, :, :16]
    expected_values = reference.values[:, :, :16]
    actual_order = compressed.keys[..., 0].argsort(dim=2).unsqueeze(-1).expand_as(compressed.keys)
    expected_order = expected_keys[..., 0].argsort(dim=2).unsqueeze(-1).expand_as(expected_keys)
    torch.testing.assert_close(compressed.keys.gather(2, actual_order), expected_keys.gather(2, expected_order))
    torch.testing.assert_close(compressed.values.gather(2, actual_order), expected_values.gather(2, expected_order))
