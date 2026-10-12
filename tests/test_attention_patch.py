# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
from torch import nn

from kvpress.attention_patch import attention_patch, search_hyperplane
from kvpress.presses.base_press import SUPPORTED_MODELS


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


@pytest.mark.parametrize("key_length", [4, 7, 10, 13])
@pytest.mark.parametrize("query_length", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_attention_patch_aligns_causal_mask_to_layer_cache(key_length, query_length, dtype):
    """Unequal layer caches must retain the causal mask for the new query tokens."""
    torch.manual_seed(0)
    query = torch.randn(1, 1, query_length, 4)
    key = torch.randn(1, 1, key_length, 4)
    value = torch.randn_like(key)
    mask_length = 10
    allowed = torch.arange(mask_length) <= torch.arange(query_length).unsqueeze(-1) + mask_length - query_length
    mask = allowed.unsqueeze(0).unsqueeze(0)
    if dtype != torch.bool:
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)
    original_mask = mask.clone()

    def attention(module, query, key, value, attention_mask, dropout):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, attention_mask, dropout)

    actual = attention_patch(attention)(nn.Module(), query, key, value, mask, 0.0)
    expected_allowed = torch.arange(key_length) <= torch.arange(query_length).unsqueeze(-1) + key_length - query_length
    expected_mask = expected_allowed.unsqueeze(0).unsqueeze(0)
    if dtype != torch.bool:
        expected_mask = torch.zeros_like(expected_mask, dtype=dtype).masked_fill(~expected_mask, torch.finfo(dtype).min)
    expected = attention(nn.Module(), query, key, value, expected_mask, 0.0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(mask, original_mask)


@pytest.mark.parametrize("mask", [None, torch.tensor([[1, 1, 0, 0]])])
def test_attention_patch_preserves_flash_attention_padding_mask(mask):
    """Flash attention's two-dimensional padding mask is not a causal score matrix."""
    query = torch.zeros(1, 1, 1, 4)
    key = torch.zeros(1, 1, 2, 4)

    def attention(module, query, key, value, attention_mask, dropout):
        return attention_mask

    assert attention_patch(attention)(nn.Module(), query, key, key, mask, 0.0) is mask


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize("query_length", [1, 4])
def test_attention_patch_preserves_broadcast_mask(dtype, query_length):
    """A singleton key dimension must keep broadcasting, including fully masked queries."""
    query = torch.zeros(1, 1, query_length, 4)
    key = torch.zeros(1, 1, 4, 4)
    value = torch.arange(16, dtype=torch.float32).reshape_as(key)
    mask = torch.ones(1, 1, query_length, 1, dtype=torch.bool)
    mask[..., 0, :] = False
    if dtype != torch.bool:
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, -torch.inf)

    def attention(module, query, key, value, attention_mask, dropout):
        assert attention_mask is mask
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, attention_mask, dropout)

    actual = attention_patch(attention)(nn.Module(), query, key, value, mask)
    expected = attention(nn.Module(), query, key, value, mask, 0.0)
    torch.testing.assert_close(actual, expected)
    assert actual[..., 0, :].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize("restriction", ["padding", "sliding", "future", "query_broadcast"])
def test_attention_patch_rejects_extending_restricted_masks(dtype, restriction):
    """Unknown masking constraints must not be replaced by visible context columns."""
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 8, 4)
    key_positions = torch.arange(5)
    query_positions = torch.arange(2).unsqueeze(-1) + 3
    mask = (key_positions <= query_positions).unsqueeze(0).unsqueeze(0)
    if restriction == "padding":
        mask[..., 0] = False
    elif restriction == "sliding":
        mask &= key_positions > query_positions - 3
    elif restriction == "future":
        mask[..., 0, -1] = True
    else:
        mask = mask[..., -1:, :]
    if dtype != torch.bool:
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)
    original_mask = mask.clone()

    def attention(module, query, key, value, attention_mask, dropout):
        pytest.fail("Restricted masks must be rejected before calling the backend")

    with pytest.raises(ValueError, match="layer-specific"):
        attention_patch(attention)(nn.Module(), query, key, key, mask)
    torch.testing.assert_close(mask, original_mask)


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize("window_source", ["module", "keyword"])
def test_attention_patch_rejects_extending_sliding_window_masks(dtype, window_source):
    """Even a window covering the entire shared mask can exclude keys in a longer layer."""
    query = torch.zeros(1, 1, 3, 4)
    key = torch.zeros(1, 1, 13, 4)
    allowed = torch.arange(10) <= torch.arange(3).unsqueeze(-1) + 7
    mask = allowed.unsqueeze(0).unsqueeze(0)
    if dtype != torch.bool:
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)
    module = nn.Module()
    kwargs = {}
    if window_source == "module":
        module.sliding_window = 10
    else:
        kwargs["sliding_window"] = 10

    def attention(module, query, key, value, attention_mask, dropout, **kwargs):
        pytest.fail("Sliding-window masks must be rejected before calling the backend")

    with pytest.raises(ValueError, match="layer-specific"):
        attention_patch(attention)(module, query, key, key, mask, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("use_infinity", [False, True])
def test_attention_patch_extends_additive_causal_mask(dtype, use_infinity):
    """Both finite minimum values and negative infinity represent blocked causal positions."""
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 8, 4)
    allowed = torch.arange(5) <= torch.arange(2).unsqueeze(-1) + 3
    blocked = -torch.inf if use_infinity else torch.finfo(dtype).min
    mask = torch.zeros(1, 1, 2, 5, dtype=dtype).masked_fill(~allowed, blocked)

    def attention(module, query, key, value, attention_mask, dropout):
        return attention_mask

    actual = attention_patch(attention)(nn.Module(), query, key, key, mask)
    expected_allowed = torch.arange(8) <= torch.arange(2).unsqueeze(-1) + 6
    expected = torch.zeros(1, 1, 2, 8, dtype=dtype).masked_fill(~expected_allowed, blocked)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_attention_patch_preserves_layer_specific_restricted_mask(dtype):
    """Restricted masks with the correct key length must remain usable without modification."""
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 5, 4)
    positions = torch.arange(5)
    query_positions = torch.arange(2).unsqueeze(-1) + 3
    mask = ((positions <= query_positions) & (positions > query_positions - 3))[None, None]
    if dtype != torch.bool:
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)
    module = nn.Module()
    module.sliding_window = 3

    def attention(module, query, key, value, attention_mask, dropout):
        return attention_mask

    assert attention_patch(attention)(module, query, key, key, mask) is mask


def test_attention_patch_rejects_extending_attention_bias():
    """Finite attention biases cannot be inferred for additional context keys."""
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 8, 4)
    allowed = torch.arange(5) <= torch.arange(2).unsqueeze(-1) + 3
    mask = torch.zeros(1, 1, 2, 5).masked_fill(~allowed, -torch.inf)
    mask[..., 0] = 0.5

    def attention(module, query, key, value, attention_mask, dropout):
        pytest.fail("Attention biases must be rejected before calling the backend")

    with pytest.raises(ValueError, match="layer-specific"):
        attention_patch(attention)(nn.Module(), query, key, key, mask)


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
@pytest.mark.parametrize("key_length", [7, 13])
def test_supported_models_eager_attention_preserves_model_specific_behavior(model, key_length):
    """Wrap each original eager backend, including its scaling and soft-capping options."""
    function = importlib.import_module(model.__module__).eager_attention_forward
    original = function.__wrapped__
    module = nn.Module().eval()
    module.num_key_value_groups = 2
    module.head_dim = 4
    torch.manual_seed(0)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, key_length, 4)
    value = torch.randn_like(key)
    allowed = torch.arange(10) <= torch.arange(3).unsqueeze(-1) + 7
    mask = torch.zeros(1, 1, 3, 10).masked_fill(~allowed, torch.finfo(torch.float32).min)
    original_mask = mask.clone()

    actual_output, actual_weights = function(module, query, key, value, mask, scaling=0.25, softcap=1.0)
    expected_allowed = torch.arange(key_length) <= torch.arange(3).unsqueeze(-1) + key_length - 3
    expected_mask = torch.zeros(1, 1, 3, key_length).masked_fill(~expected_allowed, torch.finfo(torch.float32).min)
    expected_output, expected_weights = original(module, query, key, value, expected_mask, scaling=0.25, softcap=1.0)

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(mask, original_mask)
