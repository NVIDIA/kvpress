# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import KVComposePress
from tests.fixtures import unit_test_model  # noqa: F401


@pytest.mark.parametrize("structured", [True, False])
def test_kvcompose_add_v_norm(unit_test_model, structured):  # noqa: F811
    context_len = 64
    for layer in unit_test_model.model.layers:
        layer.self_attn.masked_key_indices = None
    press = KVComposePress(structured=structured, compression_ratio=0.5, add_v_norm=True)
    cache = DynamicCache()
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, context_len), device=unit_test_model.device)
        unit_test_model(input_ids, past_key_values=cache)

    if structured:
        assert 0 < cache.get_seq_length() < context_len
    else:
        assert unit_test_model.model.layers[0].self_attn.masked_key_indices is not None


def test_kvcompose_unstructured_cache_only_holds_the_context(unit_test_model):  # noqa: F811
    context_len = 64
    press = KVComposePress(structured=False, compression_ratio=0.5)
    cache = DynamicCache()
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, context_len), device=unit_test_model.device)
        unit_test_model(input_ids, past_key_values=cache)

    assert cache.get_seq_length() == context_len
    for layer in unit_test_model.model.layers:
        assert layer.self_attn.masked_key_indices[2].max() < context_len


@torch.no_grad()
@pytest.mark.parametrize("structured", [True, False])
def test_kvcompose_generate_decodes_with_the_uncompressed_context(unit_test_model, structured):  # noqa: F811
    """Under model.generate, only the prefill is scored and the compression happens when leaving the
    context manager, so at compression_ratio=0 the decoding steps must match the model without press."""
    context_len, max_new_tokens = 32, 8
    input_ids = torch.randint(0, 1024, (1, context_len), device=unit_test_model.device)
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True
    )
    reference = unit_test_model.generate(input_ids, **generate_kwargs)

    cache = DynamicCache()
    with KVComposePress(structured=structured, compression_ratio=0.0)(unit_test_model):
        outputs = unit_test_model.generate(input_ids, past_key_values=cache, **generate_kwargs)

    assert len(outputs.scores) == len(reference.scores) == max_new_tokens
    for scores, reference_scores in zip(outputs.scores, reference.scores):
        # The press prefills with eager attention, hence the tolerance
        torch.testing.assert_close(scores, reference_scores, rtol=1e-3, atol=1e-3)
    # The structured compression rebuilds the cache from the context only
    expected_length = context_len if structured else context_len + max_new_tokens - 1
    assert cache.get_seq_length() == expected_length
