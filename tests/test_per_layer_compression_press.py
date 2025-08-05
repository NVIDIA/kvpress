# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from transformers import DynamicCache

from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_per_layer_compression_press(unit_test_model):  # noqa: F811
    press = PerLayerCompressionPress(compression_ratios=[0.1, 1], press=KnormPress())
    with press(unit_test_model):
        input_ids = torch.randint(0, 3_000, (5, 256))
        past_key_values = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    assert past_key_values.key_cache[0].shape == torch.Size([5, 2, 230, 6])
    assert past_key_values.key_cache[1].shape == torch.Size([5, 2, 0, 6])
