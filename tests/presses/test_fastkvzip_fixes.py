# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, Gemma3Config, Gemma3ForConditionalGeneration

from kvpress import FastKVzipPress
from kvpress.presses import fastkvzip_press
from kvpress.presses.fastkvzip_press import FastKVzipGate
from tests.fixtures import get_device, unit_test_model  # noqa: F401

MODEL_NAME = "MaxJeblick/llama2-0b-unit-test"


def use_random_gates(monkeypatch, gate_dtype=torch.bfloat16):
    """Replace the gate weights downloaded from the Hub by random ones, stored in `gate_dtype` as published."""
    loaded = []

    def get_gate_weight(model_name):
        loaded.append(model_name)
        config = AutoConfig.from_pretrained(model_name)
        config = getattr(config, "text_config", config)
        ngroup = config.num_attention_heads // config.num_key_value_heads
        gates = [
            FastKVzipGate(idx, config.hidden_size, config.num_key_value_heads, ngroup, gate_dtype, sink=16)
            for idx in range(config.num_hidden_layers)
        ]
        return [gate.state_dict() for gate in gates], f"{model_name.split('/')[-1]}/q{ngroup}_dim16_sink16.pt"

    monkeypatch.setattr(fastkvzip_press, "get_gate_weight", get_gate_weight)
    return loaded


def run_press(model, press, context_len=64):
    with press(model):
        model(torch.randint(0, 1024, (1, context_len), device=model.device), past_key_values=DynamicCache())


def n_masked(model):
    return sum(len(layer.self_attn.masked_key_indices[0]) for layer in model.model.layers)


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float16])
def test_fastkvzip_bfloat16_gates_on_other_model_dtypes(monkeypatch, model_dtype):
    use_random_gates(monkeypatch, gate_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=model_dtype).eval().to(get_device())
    context_len, press = 64, FastKVzipPress(compression_ratio=0.5)
    run_press(model, press, context_len)

    assert press.gates[0].q_proj.weight.dtype == torch.bfloat16
    config = model.config
    assert n_masked(model) == int(config.num_hidden_layers * config.num_key_value_heads * context_len * 0.5)


def tiny_gemma3(path):
    """Random Gemma3 with alternating sliding window and full attention layers."""
    config = Gemma3Config(
        text_config=dict(
            vocab_size=1024,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"] * 2,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
        ),
        mm_tokens_per_image=4,
    )
    torch.manual_seed(0)
    Gemma3ForConditionalGeneration(config).save_pretrained(path)
    return Gemma3ForConditionalGeneration.from_pretrained(path).eval().to(get_device())


def test_fastkvzip_gemma3_compresses_full_attention_layers(monkeypatch, tmp_path):
    use_random_gates(monkeypatch)
    model = tiny_gemma3(tmp_path)
    context_len, press = 64, FastKVzipPress(compression_ratio=0.5)
    with press(model):
        input_ids = torch.randint(0, 1024, (1, context_len), device=model.device)
        model(input_ids=input_ids, past_key_values=DynamicCache(config=model.config))

    layers = model.model.language_model.layers
    full_attention_modules = [layer.self_attn for layer in layers if not layer.self_attn.is_sliding]
    assert press.score_val.shape[0] == len(full_attention_modules) == 2
    n_pruned = sum(len(module.masked_key_indices[0]) for module in full_attention_modules)
    assert n_pruned == int(
        len(full_attention_modules) * model.config.text_config.num_key_value_heads * context_len * 0.5
    )
    assert all(layer.self_attn.masked_key_indices is None for layer in layers if layer.self_attn.is_sliding)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a second device")
def test_fastkvzip_gathers_the_scores_of_layers_on_other_devices(monkeypatch):
    use_random_gates(monkeypatch)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    context_len, press = 64, FastKVzipPress(compression_ratio=0.5)
    with press(model):
        model(torch.randint(0, 1024, (1, context_len)), past_key_values=DynamicCache())
        # Scores are computed on the device of each layer
        press.score_val[-1] = press.score_val[-1].cuda()

    assert press.score_val.device == model.device
    config = model.config
    assert n_masked(model) == int(config.num_hidden_layers * config.num_key_value_heads * context_len * 0.5)


def test_fastkvzip_without_forward_pass_is_a_no_op(unit_test_model, monkeypatch):  # noqa: F811
    use_random_gates(monkeypatch)
    with FastKVzipPress(compression_ratio=0.5)(unit_test_model):
        pass
