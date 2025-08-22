# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from kvpress.presses.expected_attention_press import ExpectedAttentionPress, ExpectedAttentionStats
from tests.fixtures import unit_test_model  # noqa: F401


# Test all stats are downloadable
def test_load_expected_attention_stats():
    for model_name in ExpectedAttentionPress.available_stats()[:5]:  # only test first 5 stats
        ExpectedAttentionStats.from_pretrained(model_name)


# Test stats computation for a small model
def test_compute_expected_attention_stats(unit_test_model):  # noqa: F811
    press = ExpectedAttentionPress(use_stats=True, n_future_positions=100, n_samples=2)
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, 256))
        unit_test_model(input_ids)
