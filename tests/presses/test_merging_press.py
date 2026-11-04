# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
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


def _merged_share(merge_fraction, rejection_per_head, n=64, dtype=torch.float32):
    """Share of threshold-eligible evicted tokens that ``MergingPress.merge`` actually merges.

    Head h has n kept tokens with keys e_0..e_{n-1} and n evicted tokens whose only similar
    survivor is kept token i, with cosine similarity a_i. A share ``rejection_per_head[h]`` of
    the a_i lie below the similarity threshold 0.5, the rest at or above it, so the number of
    eligible tokens is known exactly. Kept values are zero and evicted values are one, so a
    kept value is non-zero after the merge iff its evicted partner was merged.
    """
    num_heads = len(rejection_per_head)
    head_dim = 2 * n
    keys = torch.zeros(1, num_heads, 2 * n, head_dim)
    values = torch.zeros(1, num_heads, 2 * n, head_dim)
    n_eligible = []
    for h, rejection in enumerate(rejection_per_head):
        n_reject = round(rejection * n)
        a = torch.cat([torch.linspace(0.05, 0.45, n_reject), torch.linspace(0.5, 0.99, n - n_reject)])
        idx = torch.arange(n)
        keys[0, h, idx, idx] = 1.0
        keys[0, h, n + idx, idx] = a
        keys[0, h, n + idx, n + idx] = (1 - a**2).sqrt()
        values[0, h, n:] = 1.0
        n_eligible.append(n - n_reject)
    kept = torch.arange(n).expand(1, num_heads, n)
    press = MergingPress(
        press=KnormPress(compression_ratio=0.5), similarity_threshold=0.5, merge_fraction=merge_fraction
    )
    _, new_values = press.merge(keys.to(dtype), values.to(dtype), kept)
    merged = (new_values[0, :, :n].abs().sum(-1) > 0).sum(-1)
    return [m / e for m, e in zip(merged.tolist(), n_eligible)]


@pytest.mark.parametrize("rejection", [0.0, 0.25, 0.5, 0.75])
def test_merge_fraction_is_a_share_of_eligible_tokens(rejection):
    """merge_fraction=0.75 must merge 75% of the threshold-eligible tokens at any rejection rate."""
    (share,) = _merged_share(0.75, [rejection])
    assert abs(share - 0.75) < 0.02, f"rejection={rejection}: merged share {share:.3f} != 0.75"


def test_merge_fraction_per_row_with_heterogeneous_rejection():
    """The share holds per (batch, head) row when rows reject different fractions."""
    shares = _merged_share(0.75, [0.0, 0.25, 0.5, 0.75])
    assert all(abs(s - 0.75) < 0.02 for s in shares), shares


def test_merge_fraction_one_is_unchanged_by_the_gate():
    """merge_fraction=1.0 merges every eligible token."""
    shares = _merged_share(1.0, [0.0, 0.5])
    assert shares == [1.0, 1.0]
