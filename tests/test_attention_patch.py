# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress.attention_patch import MIN_COSINE, search_hyperplane


def attention_weights(X, K, attention_scaling=1.0):
    """Unnormalized attention weights of the fake keys K, computed in float32 like attention implementations do."""
    return torch.exp(attention_scaling * torch.bmm(X.float(), K.float().unsqueeze(-1)))


def barely_separable_queries(bsz, seq_len, head_dim, margin, dtype):
    """
    Queries q = r + margin * w with r orthogonal to the unit vector w. They are separable, but only by hyperplanes
    close to w and only with a small margin, so the first separating hyperplane found is not safe against rounding.
    """
    w = torch.randn(bsz, head_dim)
    w = w / w.norm(dim=-1, keepdim=True)
    r = torch.randn(bsz, seq_len, head_dim)
    r = r - torch.bmm(r, w.unsqueeze(-1)) * w.unsqueeze(1)
    return (r + margin * w.unsqueeze(1)).to(dtype)


def test_search_hyperplane():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bsz, seq_len, head_dim = 50, 500, 128
    X = torch.rand(bsz, seq_len, head_dim, device=device)
    K = search_hyperplane(X)
    assert attention_weights(X, K).max() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_search_hyperplane_returns_finite_keys_in_cache_dtype(dtype):
    torch.manual_seed(0)
    X = torch.randn(4, 128, 64, dtype=dtype) + 1.0
    attention_scaling = X.shape[-1] ** -0.5

    K = search_hyperplane(X, attention_scaling=attention_scaling)

    assert K.dtype == dtype
    assert torch.isfinite(K).all()
    assert attention_weights(X, K, attention_scaling).max() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_search_hyperplane_barely_separable_queries(dtype):
    """
    The search must keep improving the margin until it is safe against rounding in the cache dtype,
    instead of stopping at the first separating hyperplane and rejecting it afterwards.
    """
    torch.manual_seed(1)
    bsz, seq_len, head_dim = 4, 200, 128
    attention_scaling = head_dim**-0.5

    for _ in range(20):
        X = barely_separable_queries(bsz, seq_len, head_dim, margin=0.5, dtype=dtype)
        K = search_hyperplane(X, attention_scaling=attention_scaling)

        assert torch.isfinite(K).all()
        assert attention_weights(X, K, attention_scaling).max() == 0
        # Every query keeps a safe angle to the fake key; the cast to dtype may cost up to one unit roundoff.
        logits = torch.bmm(X.float(), K.float().unsqueeze(-1)).squeeze(-1)
        cosines = -logits / (X.float().norm(dim=-1) * K.float().norm(dim=-1, keepdim=True))
        assert (cosines > MIN_COSINE / 2).all()


def test_search_hyperplane_clamps_fake_keys_to_float16_range():
    # Small queries need a fake key beyond the float16 range to reach TARGET_LOGIT: the key is clamped to the
    # largest representable value, which still makes the attention weights underflow.
    torch.manual_seed(2)
    X = 3e-3 * torch.randn(1, 1, 128, dtype=torch.float16)
    attention_scaling = X.shape[-1] ** -0.5

    K = search_hyperplane(X, attention_scaling=attention_scaling)

    assert K.abs().max() == torch.finfo(torch.float16).max
    assert attention_weights(X, K, attention_scaling).max() == 0


def test_search_hyperplane_raises_when_fake_keys_are_not_representable():
    # A tiny query needs a fake key beyond the float16 range to underflow at all, but not beyond float32's.
    torch.manual_seed(3)
    X = 1e-4 * torch.randn(1, 1, 128)
    attention_scaling = X.shape[-1] ** -0.5

    with pytest.raises(ValueError, match="not representable"):
        search_hyperplane(X.half(), attention_scaling=attention_scaling)

    K = search_hyperplane(X, attention_scaling=attention_scaling)
    assert attention_weights(X, K, attention_scaling).max() == 0


def test_search_hyperplane_raises_for_non_separable_queries():
    X = torch.randn(1, 1, 8)
    X = torch.cat([X, -X], dim=1)  # no hyperplane has both q and -q on its positive side
    with pytest.raises(ValueError, match="Could not find a hyperplane"):
        search_hyperplane(X, max_iter=10)


def test_search_hyperplane_rejects_non_positive_scaling():
    with pytest.raises(ValueError, match="attention_scaling must be positive"):
        search_hyperplane(torch.randn(1, 4, 8), attention_scaling=0.0)
