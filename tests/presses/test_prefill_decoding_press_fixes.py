# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

from kvpress import DecodingPress, FinchPress, KnormPress, PrefillDecodingPress
from tests.fixtures import unit_test_model  # noqa: F401


@torch.no_grad()
def test_prefilling_press_context_is_entered(unit_test_model):  # noqa: F811
    finch_press = FinchPress(compression_ratio=0.5)
    finch_press.delimiter_token_id = unit_test_model.config.eos_token_id
    press = PrefillDecodingPress(
        prefilling_press=finch_press,
        decoding_press=DecodingPress(base_press=KnormPress(), compression_interval=2, target_size=8),
    )
    input_ids = torch.arange(10, 30, device=unit_test_model.device)
    input_ids[15] = finch_press.delimiter_token_id

    cache = DynamicCache()
    with press(unit_test_model):
        unit_test_model(input_ids.unsqueeze(0), past_key_values=cache)
    # The delimiter is removed and half of the remaining 19 tokens are kept
    assert cache.get_seq_length() == 9
