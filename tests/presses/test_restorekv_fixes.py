# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from safetensors.torch import save_file

from kvpress import RestoreKVPress
from kvpress.presses import restorekv_press
from tests.fixtures import unit_test_model  # noqa: F401


def test_restorekv_loads_the_adapter_on_the_model_device(unit_test_model, monkeypatch, tmp_path):  # noqa: F811
    embeddings_path = tmp_path / "restore_embeddings.safetensors"
    save_file({"restore_embeddings": torch.zeros(4, unit_test_model.config.hidden_size)}, str(embeddings_path))
    monkeypatch.setattr(restorekv_press, "hf_hub_download", lambda *args, **kwargs: str(embeddings_path))
    load_adapter_kwargs = []
    monkeypatch.setattr(unit_test_model, "load_adapter", lambda *args, **kwargs: load_adapter_kwargs.append(kwargs))
    monkeypatch.setattr(unit_test_model, "disable_adapters", lambda: None)

    RestoreKVPress(compression_ratio=0.5).post_init_from_model(unit_test_model)

    assert load_adapter_kwargs[0]["device_map"] == {"": unit_test_model.device}
