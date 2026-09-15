# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch

from kvpress.presses.knorm_press import KnormPress

# Two keys whose exact L2 norms are 1.0 and sqrt(1 + 2^-8) = 1.00195...: every component is
# exactly representable in bfloat16, but the two norms round onto the same bfloat16 value, so a
# bfloat16 score ties them and top-k tie-breaking, not the key norm, decides which one is pruned.
TIED_KEYS = torch.tensor([[[[1.0, 0.0], [1.0, 0.0625]]]])
DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]


def _module(head_dim):
    return types.SimpleNamespace(head_dim=head_dim)


@pytest.mark.parametrize("dtype", DTYPES)
def test_knorm_score_does_not_tie_distinct_norms(dtype):
    keys = TIED_KEYS.to(dtype)
    scores = KnormPress(compression_ratio=0.5).score(_module(2), None, keys, keys, None, {})

    assert scores.dtype == torch.promote_types(dtype, torch.float32)
    assert scores.shape == keys.shape[:-1]
    # Lower norm means a higher score, and the two positions must stay distinguishable.
    assert scores[0, 0, 0] > scores[0, 0, 1]


@pytest.mark.parametrize("dtype", DTYPES)
def test_knorm_compress_keeps_the_lowest_norm_key(dtype):
    keys = TIED_KEYS.to(dtype)
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).to(dtype)
    pressed_keys, pressed_values = KnormPress(compression_ratio=0.5).compress(_module(2), None, keys, values, None, {})

    assert pressed_keys.shape == (1, 1, 1, 2)
    assert torch.equal(pressed_keys, keys[:, :, :1])
    assert torch.equal(pressed_values, values[:, :, :1])


def test_knorm_score_of_bfloat16_keys_is_almost_tie_free():
    torch.manual_seed(0)
    keys = torch.randn(1, 8, 512, 128, dtype=torch.bfloat16)
    scores = KnormPress(compression_ratio=0.5).score(_module(128), None, keys, keys, None, {})

    tie_free = min(row.unique().numel() / row.numel() for row in scores[0])
    assert tie_free > 0.99
