# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from transformers import AutoModelForCausalLM, pipeline


@pytest.fixture(scope="session")
def unit_test_model():
    return AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test").eval()


@pytest.fixture(scope="session")
def unit_test_model_output_attention():
    return AutoModelForCausalLM.from_pretrained(
        "MaxJeblick/llama2-0b-unit-test", attn_implementation="eager", output_attentions=True
    ).eval()


@pytest.fixture(scope="session")
def danube_500m_model():
    return AutoModelForCausalLM.from_pretrained("h2oai/h2o-danube3-500m-chat").eval()


@pytest.fixture(scope="session")
def kv_press_pipeline():
    return pipeline(
        "kv-press-text-generation",
        model="maxjeblick/llama2-0b-unit-test",
        device=0 if torch.cuda.is_available() else -1,
    )
