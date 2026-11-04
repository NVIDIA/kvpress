# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress import CompactorPress, LeverageScorePress, NonCausalAttnPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_compactor_press(unit_test_model):  # noqa: F811
    for press in [
        CompactorPress(0.5, sink_size_start=0, sink_size_end=0),
        CompactorPress(0.2, sink_size_start=8, sink_size_end=4),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


def test_leverage_press(unit_test_model):  # noqa: F811
    for press in [
        LeverageScorePress(0.5, sketch_dimension=48),
        LeverageScorePress(0.5, sketch_dimension=64),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


def test_non_causal_attn_press(unit_test_model):  # noqa: F811
    for press in [
        NonCausalAttnPress(0.5, chunk_size=128),
        NonCausalAttnPress(0.5, chunk_size=256),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


@pytest.mark.parametrize("seq_len", [7, 8, 10, 13])
def test_non_causal_chunked_attn_ignores_padding(seq_len):
    torch.manual_seed(0)
    q, k = torch.randn(2, 2, 3, seq_len, 8).unbind(0)
    # Column sums of the softmax attention computed independently on each chunk, without padding
    expected = []
    for start in range(0, seq_len, 4):
        q_chunk, k_chunk = q[..., start : start + 4, :], k[..., start : start + 4, :]
        expected.append(torch.softmax(q_chunk @ k_chunk.transpose(-2, -1), dim=-1).sum(dim=-2))
    scores = NonCausalAttnPress.non_causal_chunked_attn(q, k, chunk_size=4)
    torch.testing.assert_close(scores, torch.cat(expected, dim=-1))
