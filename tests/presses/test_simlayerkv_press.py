# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvpress import SimLayerKVPress


@pytest.mark.parametrize(
    "seq_len, n_last, lazy_threshold, expected_kept, expected_ratio",
    [
        (16, 1, 0.0, 7, 0.5625),
        (16, 2, 0.0, 6, 0.625),
        (16, 4, 0.0, 4, 0.75),
        (8, 1, 0.0, 8, 0.0),
        (8, 2, 0.0, 8, 0.0),
        (8, 4, 0.0, 8, 0.0),
        (16, 2, 1.0, 16, 0.0),
    ],
)
def test_simlayer_compression_ratio_matches_retained_cache(
    seq_len, n_last, lazy_threshold, expected_kept, expected_ratio
):
    """Nondefault observation windows must report the actual fraction evicted."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    ).eval()
    press = SimLayerKVPress(lazy_threshold=lazy_threshold, n_last=n_last, n_initial=2, n_recent=6)

    with torch.no_grad(), press(model):
        output = model(torch.arange(seq_len).unsqueeze(0), use_cache=True)

    for layer in output.past_key_values.layers:
        assert layer.keys.shape[-2] == expected_kept
        assert layer.values.shape[-2] == expected_kept
    assert press.compression_ratio == pytest.approx(expected_ratio)
