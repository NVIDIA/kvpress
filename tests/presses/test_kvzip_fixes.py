# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import KVgradPress, KVzipPress
from tests.fixtures import unit_test_model  # noqa: F401

PRESSES = [(KVzipPress, {"chunk_size": 32}), (KVgradPress, {"chunk_size": 32})]


def masked_keys(model, press, run):
    """Run `run(model)` within `press` and return the masked key indices of every layer."""
    with press(model):
        run(model)
    return [tuple(indices.clone() for indices in layer.self_attn.masked_key_indices) for layer in model.model.layers]


def assert_same_masks(masks, reference_masks):
    for layer_masks, reference_layer_masks in zip(masks, reference_masks):
        for indices, reference_indices in zip(layer_masks, reference_layer_masks):
            assert torch.equal(indices, reference_indices)


@pytest.mark.parametrize("press_cls, kwargs", PRESSES)
def test_kvzip_compresses_the_cache_created_by_the_model(unit_test_model, press_cls, kwargs):  # noqa: F811
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
    reference = masked_keys(
        unit_test_model,
        press_cls(compression_ratio=0.5, **kwargs),
        lambda m: m(input_ids, past_key_values=DynamicCache()),
    )

    assert_same_masks(
        masked_keys(unit_test_model, press_cls(compression_ratio=0.5, **kwargs), lambda m: m(input_ids)), reference
    )
    assert_same_masks(
        masked_keys(
            unit_test_model,
            press_cls(compression_ratio=0.5, **kwargs),
            lambda m: m.model(input_ids, past_key_values=DynamicCache()),
        ),
        reference,
    )


@pytest.mark.parametrize("press_cls, kwargs", PRESSES)
def test_kvzip_generate_replays_the_prefilled_context(unit_test_model, press_cls, kwargs):  # noqa: F811
    context_len = 64
    input_ids = torch.randint(0, 1024, (1, context_len), device=unit_test_model.device)
    reference = masked_keys(
        unit_test_model,
        press_cls(compression_ratio=0.5, **kwargs),
        lambda m: m(input_ids, past_key_values=DynamicCache()),
    )

    cache = DynamicCache()
    masks = masked_keys(
        unit_test_model,
        press_cls(compression_ratio=0.5, **kwargs),
        lambda m: m.generate(input_ids, past_key_values=cache, max_new_tokens=5, do_sample=False),
    )

    assert_same_masks(masks, reference)
    assert cache.get_seq_length() == context_len


def test_kvzip_requires_input_ids(unit_test_model):  # noqa: F811
    inputs_embeds = unit_test_model.model.embed_tokens(torch.randint(0, 1024, (1, 16), device=unit_test_model.device))
    with pytest.raises(ValueError, match="input_ids"):
        with KVzipPress(compression_ratio=0.5)(unit_test_model):
            unit_test_model(inputs_embeds=inputs_embeds, past_key_values=DynamicCache())


def test_kvzip_requires_a_cache(unit_test_model):  # noqa: F811
    with pytest.raises(ValueError, match="use_cache=True"):
        with KVzipPress(compression_ratio=0.5)(unit_test_model):
            unit_test_model(torch.randint(0, 1024, (1, 16), device=unit_test_model.device), use_cache=False)
