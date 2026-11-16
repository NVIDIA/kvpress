# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from kvpress import ChunkPress, KnormPress, SnapKVPress, TOVAPress


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
@pytest.mark.parametrize("seq_len,chunk_length", [(1, 4), (5, 4), (9, 4), (5, 1)])
@pytest.mark.parametrize("method", ["tova", "snapkv", "knorm"])
def test_singleton_chunks_are_kept(device, seq_len, chunk_length, method):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        _attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).eval().to(device)
    inputs = torch.arange(2 * seq_len, device=device).reshape(2, seq_len) % config.vocab_size
    presses = {
        "tova": TOVAPress(compression_ratio=0.5),
        "snapkv": SnapKVPress(compression_ratio=0.5, window_size=1, kernel_size=3),
        "knorm": KnormPress(compression_ratio=0.5),
    }

    with torch.inference_mode():
        reference = model(inputs, past_key_values=DynamicCache()).past_key_values
        with ChunkPress(press=presses[method], chunk_length=chunk_length)(model):
            compressed = model(inputs, past_key_values=DynamicCache()).past_key_values

    expected_length = sum(max(1, int(min(chunk_length, seq_len - i) * 0.5)) for i in range(0, seq_len, chunk_length))
    assert compressed.layers[0].keys.shape[2] == expected_length
    assert compressed.layers[0].values.shape[2] == expected_length
    torch.testing.assert_close(compressed.layers[0].keys[:, :, -1], reference.layers[0].keys[:, :, -1])
    torch.testing.assert_close(compressed.layers[0].values[:, :, -1], reference.layers[0].values[:, :, -1])
