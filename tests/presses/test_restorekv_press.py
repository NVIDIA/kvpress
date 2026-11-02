# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

import pytest
import torch
from peft import LoraConfig
from safetensors.torch import save_file
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from kvpress import RestoreKVPress

restorekv_module = import_module("kvpress.presses.restorekv_press")


@pytest.fixture
def local_checkpoints(tmp_path):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = LlamaForCausalLM(config).eval()
    base_path = tmp_path / "base"
    model.save_pretrained(base_path)
    model.add_adapter(LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
    adapter_path = tmp_path / "adapter"
    model.save_pretrained(adapter_path)
    save_file(
        {"restore_embeddings": torch.randn(2, config.hidden_size)}, adapter_path / "restore_embeddings.safetensors"
    )
    return base_path, adapter_path


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("placement", ["to", "device_map"])
def test_adapter_follows_base_model(local_checkpoints, monkeypatch, device, placement):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    base_path, adapter_path = local_checkpoints
    if placement == "device_map":
        model = LlamaForCausalLM.from_pretrained(base_path, device_map=device).eval()
    else:
        model = LlamaForCausalLM.from_pretrained(base_path).to(device).eval()

    monkeypatch.setattr(restorekv_module, "hf_hub_download", lambda repo, filename: str(adapter_path / filename))
    original_load_adapter = model.load_adapter

    def load_local_adapter(repo, **kwargs):
        return original_load_adapter(str(adapter_path), **kwargs)

    monkeypatch.setattr(model, "load_adapter", load_local_adapter)
    press = RestoreKVPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    adapter_params = [param for name, param in model.named_parameters() if "lora_" in name]
    assert adapter_params
    assert all(param.device == model.device for param in adapter_params)
    assert press.restore_embeddings.device == model.device

    model.set_adapter(press.adapter_name)
    model.enable_adapters()
    with torch.no_grad():
        output = model(torch.arange(4, device=model.device).unsqueeze(0), past_key_values=DynamicCache())
    assert torch.isfinite(output.logits).all()
    model.disable_adapters()

    press.post_init_from_model(model)
    assert list(model.peft_config) == [press.adapter_name]
