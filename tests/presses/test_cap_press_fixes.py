# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
