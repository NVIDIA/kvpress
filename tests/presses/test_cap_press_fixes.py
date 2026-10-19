# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from kvpress import CapPress
from tests.fixtures import unit_test_model  # noqa: F401


class CPURotaryEmbedding(nn.Module):
    """Rotary embedding returning its outputs on the CPU, like one placed on another device by a device_map."""

    def forward(self, x, position_ids):
        shape = (position_ids.shape[0], position_ids.shape[1], x.shape[-1])
        return torch.ones(shape, dtype=x.dtype), torch.zeros(shape, dtype=x.dtype)


def test_cap_rope_matrix_is_built_on_the_device_of_the_queries(unit_test_model):  # noqa: F811
    module = unit_test_model.model.layers[0].self_attn
    module.rotary_emb = CPURotaryEmbedding()
    try:
        device = torch.device("meta")
        rotation = CapPress()._avg_rope_matrix(module, q_len=8, device=device, dtype=torch.float32)
    finally:
        module.rotary_emb = unit_test_model.model.rotary_emb

    assert rotation.device == device
    assert rotation.shape == (module.head_dim, module.head_dim)


@torch.no_grad()
@pytest.mark.parametrize("n_cached", [0, 100])
def test_cap_future_positions_start_after_the_last_token(unit_test_model, monkeypatch, n_cached):  # noqa: F811
    start_positions = []
    avg_rope_matrix = CapPress._avg_rope_matrix

    def recording_avg_rope_matrix(self, module, q_len, device, dtype):
        start_positions.append(q_len)
        return avg_rope_matrix(self, module, q_len, device, dtype)

    monkeypatch.setattr(CapPress, "_avg_rope_matrix", recording_avg_rope_matrix)
    model, q_len = unit_test_model, 16
    module = model.model.layers[0].self_attn
    module.rotary_emb = model.model.rotary_emb
    hidden_states = torch.randn(1, q_len, model.config.hidden_size, dtype=model.dtype, device=model.device)
    keys = torch.randn(1, model.config.num_key_value_heads, n_cached + q_len, module.head_dim, dtype=model.dtype)
    keys = keys.to(model.device)
    cache_position = torch.arange(n_cached, n_cached + q_len, device=model.device)

    CapPress(compression_ratio=0.5).score(module, hidden_states, keys, keys, None, {"cache_position": cache_position})
    assert start_positions == [n_cached + q_len]
