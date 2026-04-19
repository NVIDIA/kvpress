# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

from kvpress import AdaKVPress, KnormPress, SnapKVPress
from kvpress.presses.merging_press import MergingPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_merge_differs_from_hard_eviction(unit_test_model):  # noqa: F811
    """Merged values should differ from hard-evicted values."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

    base = KnormPress(compression_ratio=0.5)
    with base(unit_test_model):
        cache_hard = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_hard)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.5), similarity_threshold=0.0)
    with wrapper(unit_test_model):
        cache_merge = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_merge)

    assert cache_hard.get_seq_length() == cache_merge.get_seq_length() == 32
    any_diff = any(
        not torch.equal(cache_hard.layers[i].values, cache_merge.layers[i].values)
        for i in range(len(cache_hard.layers))
    )
    assert any_diff, "Merging produced identical values to hard eviction"


def test_default_preserves_keys(unit_test_model):  # noqa: F811
    """Default merge_keys=False should not modify keys (preserves RoPE)."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

    base = KnormPress(compression_ratio=0.5)
    with base(unit_test_model):
        cache_hard = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_hard)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.5))
    with wrapper(unit_test_model):
        cache_merge = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_merge)

    for i in range(len(cache_hard.layers)):
        assert torch.equal(cache_hard.layers[i].keys, cache_merge.layers[i].keys), (
            f"Layer {i}: merge_keys=False should not modify keys"
        )


def test_merge_preserves_more_info(unit_test_model):  # noqa: F811
    """Merge-on-evict stays closer to uncompressed cache than hard eviction."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

    cache_ref = DynamicCache()
    unit_test_model(input_ids.clone(), past_key_values=cache_ref)
    ref_values = [layer.values.float() for layer in cache_ref.layers]

    base = KnormPress(compression_ratio=0.7)
    with base(unit_test_model):
        cache_hard = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_hard)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.7), similarity_threshold=0.0)
    with wrapper(unit_test_model):
        cache_merge = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_merge)

    def recon_error(cache):
        return sum(
            (layer.values.float() - ref_values[i][:, :, : layer.values.shape[2]]).norm().item()
            for i, layer in enumerate(cache.layers)
        )

    assert recon_error(cache_merge) <= recon_error(cache_hard) + 1e-6


def test_half_precision_no_nan(unit_test_model):  # noqa: F811
    """Float32 accumulation must produce finite results in fp16."""
    model = unit_test_model.to(torch.float16)
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 64), device=model.device)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.5))
    with wrapper(model):
        cache = DynamicCache()
        model(input_ids, past_key_values=cache)

    for layer in cache.layers:
        assert torch.isfinite(layer.keys).all()
        assert torch.isfinite(layer.values).all()
    model.float()


def test_batch_size_greater_than_one(unit_test_model):  # noqa: F811
    """Kernel must handle batch_size > 1 correctly."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (2, 64), device=unit_test_model.device)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.5))
    with wrapper(unit_test_model):
        cache = DynamicCache()
        unit_test_model(input_ids, past_key_values=cache)

    assert cache.get_seq_length() == 32
    for layer in cache.layers:
        assert layer.keys.shape[0] == 2


def test_adakv_composition(unit_test_model):  # noqa: F811
    """MergingPress(AdaKV) uses mask-based path and changes values."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)

    plain = AdaKVPress(SnapKVPress(compression_ratio=0.5))
    with plain(unit_test_model):
        cache_plain = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_plain)

    wrapper = MergingPress(press=AdaKVPress(SnapKVPress(compression_ratio=0.5)))
    with wrapper(unit_test_model):
        cache_merge = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_merge)

    any_diff = any(
        not torch.equal(cache_plain.layers[i].values, cache_merge.layers[i].values)
        for i in range(len(cache_plain.layers))
    )
    assert any_diff, "MergingPress(AdaKV) should differ from plain AdaKV"
