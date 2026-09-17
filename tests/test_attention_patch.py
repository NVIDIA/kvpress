# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress.attention_patch import search_hyperplane


def attention_weights(X, K, attention_scaling):
    """Unnormalized attention weights of the fake keys K, computed in float32 like attention implementations do."""
    return torch.exp(attention_scaling * torch.bmm(X.float(), K.float().unsqueeze(-1)))


def test_search_hyperplane():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bsz, seq_len, head_dim = 50, 500, 128
    X = torch.rand(bsz, seq_len, head_dim, device=device)
    K = search_hyperplane(X)
    assert attention_weights(X, K, head_dim**-0.5).max() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_search_hyperplane_returns_finite_keys_in_cache_dtype(dtype):
    torch.manual_seed(0)
    X = torch.randn(4, 128, 64, dtype=dtype) + 1.0
    attention_scaling = X.shape[-1] ** -0.5

    K = search_hyperplane(X)

    assert K.dtype == dtype
    assert torch.isfinite(K).all()
    assert attention_weights(X, K, attention_scaling).max() == 0


def test_search_hyperplane_scales_fake_keys_to_float16_range():
    torch.manual_seed(2)
    X = 3e-3 * torch.randn(1, 1, 128, dtype=torch.float16)
    attention_scaling = X.shape[-1] ** -0.5

    K = search_hyperplane(X)

    assert K.abs().max() == torch.finfo(torch.float16).max
    assert torch.isfinite(K).all()
    assert attention_weights(X, K, attention_scaling).max() == 0


def test_search_hyperplane_raises_for_non_separable_queries():
    X = torch.randn(1, 1, 8)
    X = torch.cat([X, -X], dim=1)  # no hyperplane has both q and -q on its positive side
    with pytest.raises(ValueError, match="Could not find fake keys"):
        search_hyperplane(X, max_iter=10)
