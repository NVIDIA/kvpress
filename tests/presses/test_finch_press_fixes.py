# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch

from kvpress import FinchPress
from kvpress.utils import compute_n_kept


def make_module(num_heads):
    return types.SimpleNamespace(head_dim=1, config=types.SimpleNamespace(num_attention_heads=num_heads))


@pytest.mark.parametrize("context_length, window_size, chunk_length", [(10, 3, 5), (12, 4, 5), (7, 5, 3)])
def test_chunked_compression_keeps_the_window(context_length, window_size, chunk_length):
    torch.manual_seed(0)
    k_len = context_length + window_size
    press = FinchPress(compression_ratio=0.5, chunk_length=chunk_length, rerotate_keys=False)
    press.window_size = window_size
    keys = torch.arange(k_len, dtype=torch.float32).view(1, 1, k_len, 1)
    attentions = torch.rand(1, 1, k_len, k_len).softmax(dim=-1)

    new_keys, _ = press.compress(make_module(1), None, keys, keys.clone(), attentions, {})

    kept = new_keys[0, 0, :, 0].long().tolist()
    assert set(range(context_length, k_len)) <= set(kept)
    chunk_lengths = [min(chunk_length, context_length - i) for i in range(0, context_length, chunk_length)]
    assert len(kept) == sum(compute_n_kept(length, 0.5) for length in chunk_lengths) + window_size
