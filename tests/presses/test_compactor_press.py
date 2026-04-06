# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress import CompactorPress, LeverageScorePress, NonCausalAttnPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_compactor_press(unit_test_model):  # noqa: F811
    for press in [
        CompactorPress(0.5, sink_size_start=0, sink_size_end=0),
        CompactorPress(0.2, sink_size_start=8, sink_size_end=4),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


def test_leverage_press(unit_test_model):  # noqa: F811
    for press in [
        LeverageScorePress(0.5, sketch_dimension=48),
        LeverageScorePress(0.5, sketch_dimension=64),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


def test_non_causal_attn_press(unit_test_model):  # noqa: F811
    for press in [
        NonCausalAttnPress(0.5, chunk_size=128),
        NonCausalAttnPress(0.5, chunk_size=256),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)
