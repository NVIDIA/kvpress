# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import DynamicCache

from kvpress import ComposedPress, DuoAttentionPress, QFilterPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_composed_press_qfilter_without_post_init(unit_test_model):  # noqa: F811
    press1 = QFilterPress(compression_ratio=0.2)
    press2 = QFilterPress(compression_ratio=0.2)
    composed_press = ComposedPress([press1, press2])
    with pytest.raises(ValueError, match="post_init_from_model"):
        with composed_press(unit_test_model):
            input_ids = unit_test_model.dummy_inputs["input_ids"]
            unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_composed_press_duo_attention_without_post_init(unit_test_model):  # noqa: F811
    press1 = DuoAttentionPress()
    press2 = DuoAttentionPress()
    model = type(
        "model", (), {"config": type("config", (), {"name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct"})}
    )()
    model.device = "cpu"
    composed_press = ComposedPress([press1, press2])
    with pytest.raises(ValueError, match="post_init_from_model"):
        with composed_press(unit_test_model):
            input_ids = unit_test_model.dummy_inputs["input_ids"]
            unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_composed_qfilter_press_with_post_init(unit_test_model):  # noqa: F811
    model = type(
        "model", (), {"config": type("config", (), {"name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct"})}
    )()
    model.device = "cpu"
    model.dtype = torch.float32
    press1 = QFilterPress(compression_ratio=0.2)
    press1.__post_init_from_model__(model)
    press2 = QFilterPress(compression_ratio=0.2)
    press2.__post_init_from_model__(model)

    composed_press = ComposedPress([press1, press2])
    with composed_press(unit_test_model):
        input_ids = unit_test_model.dummy_inputs["input_ids"]
        with pytest.raises(RuntimeError, match="The size of tensor"):
            unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_composed_duo_attention_press_with_post_init(unit_test_model):  # noqa: F811
    model = type(
        "model", (), {"config": type("config", (), {"name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct"})}
    )()
    model.device = "cpu"
    model.dtype = torch.float32
    press1 = DuoAttentionPress()
    press1.__post_init_from_model__(model)
    press2 = DuoAttentionPress()
    press2.__post_init_from_model__(model)

    composed_press = ComposedPress([press1, press2])
    with composed_press(unit_test_model):
        input_ids = unit_test_model.dummy_inputs["input_ids"]
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values
