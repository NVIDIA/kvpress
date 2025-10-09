# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DynamicCache

from kvpress.presses.block_press import BlockPress
from kvpress.presses.scorer_press import ScorerPress
from tests.fixtures import unit_test_model  # noqa: F401


@dataclass
class HiddenStatesPress(ScorerPress):  # dummy press using hidden states

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        return hidden_states.mean(-1).unsqueeze(1).expand_as(keys.norm(dim=-1))


def test_block_press_is_streaming_top_k(unit_test_model):  # noqa: F811
    """
    Test that BlockPress correctly applies the compression ratio and keeps the output consistent.
    """
    press = HiddenStatesPress(compression_ratio=0.5)
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, 1024, (1, 256), generator=generator).to(unit_test_model.device)
    keys_hash = []
    values_hash = []

    for block_size in [2, 4, 8, 128, 256]:
        composed_press = BlockPress(press=press, block_size=block_size)
        with composed_press(unit_test_model):
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache).past_key_values
            assert cache.get_seq_length() == 128
            keys = torch.cat([cache.layers[layer_idx].keys for layer_idx in range(len(cache.layers))])
            values = torch.cat([cache.layers[layer_idx].values for layer_idx in range(len(cache.layers))])
            keys_hash.append(keys.sum().item())
            values_hash.append(values.sum().item())

    with press(unit_test_model):
        cache = DynamicCache()
        unit_test_model(input_ids, past_key_values=cache).past_key_values
        assert cache.get_seq_length() == 128
        keys = torch.cat([cache.layers[layer_idx].keys for layer_idx in range(len(cache.layers))])
        values = torch.cat([cache.layers[layer_idx].values for layer_idx in range(len(cache.layers))])
        keys_hash.append(keys.sum().item())
        values_hash.append(values.sum().item())

    keys_tensor = torch.tensor(keys_hash)
    values_tensor = torch.tensor(values_hash)
    assert torch.allclose(keys_tensor, keys_tensor[-1])
    assert torch.allclose(values_tensor, values_tensor[-1])
