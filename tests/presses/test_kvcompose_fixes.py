# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

from kvpress import KVComposePress
from tests.fixtures import unit_test_model  # noqa: F401


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
