# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from transformers import AutoModelForCausalLM, pipeline


def get_device():
    """Helper function that returns the appropriate device (GPU if available, otherwise CPU)"""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def unit_test_model():
    model = AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test").eval()
    return model.to(get_device())


@pytest.fixture(scope="session")
def unit_test_model_output_attention():
    model = AutoModelForCausalLM.from_pretrained(
        "MaxJeblick/llama2-0b-unit-test", attn_implementation="eager", output_attentions=True
    ).eval()
    return model.to(get_device())


@pytest.fixture(scope="session")
def danube_500m_model():
    model = AutoModelForCausalLM.from_pretrained("h2oai/h2o-danube3-500m-chat").eval()
    return model.to(get_device())


@pytest.fixture(scope="session")
def kv_press_unit_test_pipeline():
    return pipeline(
        "kv-press-text-generation",
        model="maxjeblick/llama2-0b-unit-test",
        device=get_device(),
    )


@pytest.fixture(scope="session")
def kv_press_danube_pipeline():
    return pipeline(
        "kv-press-text-generation",
        model="h2oai/h2o-danube3-500m-chat",
        device=get_device(),
    )


@pytest.fixture(scope="session")
def kv_press_adaptive_pipeline():
    """Flexible pipeline that uses GPU+flash attention if available, otherwise CPU"""
    device = get_device()
    ckpt = "meta-llama/Llama-3.2-1B-Instruct"

    # Use flash attention only if GPU is available
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    pipe = pipeline(
        "kv-press-text-generation",
        model=ckpt,
        device=device,
        dtype="auto",
        model_kwargs=model_kwargs,
    )
    return pipe


@pytest.fixture(scope="class")
def kv_press_llama3_1_flash_attn_pipeline():
    device = "cuda:0"
    ckpt = "meta-llama/Llama-3.1-8B-Instruct"
    attn_implementation = "flash_attention_2"
    pipe = pipeline(
        "kv-press-text-generation",
        model=ckpt,
        device=device,
        model_kwargs={"attn_implementation": attn_implementation, "dtype": torch.bfloat16},
    )
    return pipe


@pytest.fixture(scope="class")
def kv_press_llama3_2_flash_attn_pipeline():
    device = "cuda:0"
    ckpt = "meta-llama/Llama-3.2-1B-Instruct"
    attn_implementation = "flash_attention_2"
    pipe = pipeline(
        "kv-press-text-generation",
        model=ckpt,
        device=device,
        model_kwargs={"attn_implementation": attn_implementation, "dtype": torch.bfloat16},
    )
    return pipe


@pytest.fixture(scope="class")
def kv_press_qwen3_flash_attn_pipeline():
    device = "cuda:0"
    ckpt = "Qwen/Qwen3-4B-Instruct-2507"
    attn_implementation = "flash_attention_2"
    pipe = pipeline(
        "kv-press-text-generation",
        model=ckpt,
        device=device,
        model_kwargs={"attn_implementation": attn_implementation, "dtype": torch.bfloat16},
    )
    return pipe
