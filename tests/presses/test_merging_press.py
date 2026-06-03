# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

from kvpress import KnormPress
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


def test_keys_unchanged(unit_test_model):  # noqa: F811
    """Keys must not be modified (RoPE-safe by design)."""
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
            f"Layer {i}: keys must not be modified"
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


def test_merge_method_signature():
    """Lock in the public merge(keys, values, indices) surface from #219.

    `indices` are the kept positions (output of scores.topk). The method
    folds evicted information into the kept slots; other positions are
    unchanged (they get pruned by compress() after this returns).
    """
    torch.manual_seed(42)
    bsz, num_heads, seq_len, head_dim = 1, 2, 8, 4
    keys = torch.randn(bsz, num_heads, seq_len, head_dim)
    values = torch.randn(bsz, num_heads, seq_len, head_dim)
    # Keep first half, evict second half
    kept = torch.arange(seq_len // 2).expand(bsz, num_heads, seq_len // 2)

    press = MergingPress(press=KnormPress(compression_ratio=0.5))
    new_keys, new_values = press.merge(keys, values, kept)

    assert new_keys.shape == keys.shape
    assert new_values.shape == values.shape
    # Keys are returned unchanged (RoPE-safe by design)
    assert torch.equal(new_keys, keys)
    # Kept positions absorb evicted information: values must change there
    assert not torch.equal(new_values[:, :, : seq_len // 2], values[:, :, : seq_len // 2])
