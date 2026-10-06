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
