# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import CompressionRatioDecodingPress, KnormPress
from tests.fixtures import unit_test_model  # noqa: F401


@pytest.mark.parametrize("prefill_in_context", [True, False])
@pytest.mark.parametrize("logical_position_ids", [True, False])
@torch.no_grad()
def test_target_size_follows_all_tokens_seen(unit_test_model, prefill_in_context, logical_position_ids):  # noqa: F811
    press = CompressionRatioDecodingPress(base_press=KnormPress(), target_compression_ratio=0.5, compression_interval=8)
    cache = DynamicCache()
    input_ids = torch.randint(0, 1000, (1, 64), device=unit_test_model.device)
    if not prefill_in_context:
        logits = unit_test_model(input_ids, past_key_values=cache).logits

    with press(unit_test_model):
        if prefill_in_context:
            logits = unit_test_model(input_ids, past_key_values=cache).logits
        for step in range(64):
            position_ids = torch.tensor([[64 + step]], device=input_ids.device) if logical_position_ids else None
            next_ids = logits[:, -1:].argmax(dim=-1)
            logits = unit_test_model(next_ids, past_key_values=cache, position_ids=position_ids).logits

    # 128 tokens were seen when the last compression happened
    assert cache.get_seq_length() == 64
    assert press.tokens_seen == {}
