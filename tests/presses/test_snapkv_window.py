# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from kvpress.presses.snapkv_press import SnapKVPress


def make_inputs(device, dtype):
    # Constant attention per query head makes the expected GQA averages exact.
    batch, heads, kv_heads, length, window, head_dim = 2, 4, 2, 16, 4, 8
    head_values = torch.tensor(
        [[1 / 8, 2 / 8, 3 / 8, 4 / 8], [2 / 8, 4 / 8, 6 / 8, 8 / 8]],
        device=device,
        dtype=dtype,
    )
    attention = head_values[:, :, None, None].expand(batch, heads, window, length).contiguous()
    module = SimpleNamespace(config=SimpleNamespace(num_attention_heads=heads), head_dim=head_dim)
    hidden = torch.zeros(batch, length, head_dim, device=device, dtype=dtype)
    keys = torch.zeros(batch, kv_heads, length, head_dim, device=device, dtype=dtype)
    return module, hidden, keys, attention


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_window_scores_preserve_global_sentinel_and_head_averages(device, dtype):
    module, hidden, keys, attention = make_inputs(device, dtype)
    press = SnapKVPress(compression_ratio=0.5, window_size=4, kernel_size=1)

    scores = press.score(module, hidden, keys, keys, attention, {})

    group_values = torch.tensor([[3 / 16, 7 / 16], [3 / 8, 7 / 8]], device=device, dtype=dtype)
    expected = group_values[:, :, None].expand(2, 2, 16).clone()
    expected[:, :, -4:] = 15 / 8
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_scoring_does_not_synchronize_cuda_to_read_window_sentinel():
    module, hidden, keys, attention = make_inputs("cuda", torch.float32)
    press = SnapKVPress(compression_ratio=0.5, window_size=4, kernel_size=1)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        scores = press.score(module, hidden, keys, keys, attention, {})
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert scores.shape == (2, 2, 16)
