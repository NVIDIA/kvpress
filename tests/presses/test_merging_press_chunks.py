# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress import KnormPress
from kvpress.presses.merging_press import MergingPress


@pytest.mark.parametrize("similarity_threshold, merge_fraction", [(0.0, 1.0), (0.1, 0.75)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_merge_matches_unchunked_merge(similarity_threshold, merge_fraction, dtype):
    torch.manual_seed(0)
    bsz, num_heads, seq_len, head_dim, n_kept = 2, 3, 50, 8, 20
    keys, values = torch.randn(2, bsz, num_heads, seq_len, head_dim, dtype=dtype).unbind(0)
    indices = torch.rand(bsz, num_heads, seq_len).topk(n_kept, dim=-1).indices

    def merge(chunk_size):
        press = MergingPress(
            KnormPress(compression_ratio=0.6),
            similarity_threshold=similarity_threshold,
            merge_fraction=merge_fraction,
            chunk_size=chunk_size,
        )
        return press.merge(keys, values, indices)

    # Matrix products of different shapes may round differently, hence assert_close
    reference_keys, reference_values = merge(chunk_size=seq_len)
    for chunk_size in [1, 3, 7]:
        new_keys, new_values = merge(chunk_size)
        assert torch.equal(new_keys, reference_keys)
        torch.testing.assert_close(new_values, reference_values)
    assert torch.equal(merge(chunk_size=seq_len - n_kept)[1], reference_values)
