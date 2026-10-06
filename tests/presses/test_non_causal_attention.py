# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress import NonCausalAttnPress


def reference_chunked_attn(q, k, chunk_size):
    """Column sums of the softmax attention computed independently on each chunk, without padding."""
    column_sums = []
    for start in range(0, q.shape[-2], chunk_size):
        q_chunk, k_chunk = q[..., start : start + chunk_size, :], k[..., start : start + chunk_size, :]
        column_sums.append(torch.softmax(q_chunk @ k_chunk.transpose(-2, -1), dim=-1).sum(dim=-2))
    return torch.cat(column_sums, dim=-1)


def test_padding_adds_no_attention_mass():
    torch.manual_seed(0)
    q, k = torch.randn(2, 1, 3, 2, 8).unbind(0)
    scores = NonCausalAttnPress.non_causal_chunked_attn(q, k, chunk_size=4)
    torch.testing.assert_close(scores.sum(dim=-1), torch.full((1, 3), 2.0))


@pytest.mark.parametrize("seq_len", [7, 8, 10, 13])
def test_matches_unpadded_chunked_attention(seq_len):
    torch.manual_seed(0)
    q, k = torch.randn(2, 2, 3, seq_len, 8).unbind(0)
    scores = NonCausalAttnPress.non_causal_chunked_attn(q, k, chunk_size=4)
    torch.testing.assert_close(scores, reference_chunked_attn(q, k, chunk_size=4))
