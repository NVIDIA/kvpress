# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kvpress import DecodingPress, KnormPress, PrefillDecodingPress
from kvpress.presses.base_press import BasePress, is_prefilling


@dataclass
class CountingPress(BasePress):
    calls: int = 0

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return keys, values


@dataclass
class HookRecorderPress(BasePress):
    hook_calls: int = 0

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return keys, values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        self.hook_calls += 1
        return output

    def reset(self):
        pass


def make_hook_args(cache_position: list[int], q_len: int):
    hidden_states = torch.randn(1, q_len, 4)
    seq_len = max(cache_position) + 1
    cache_layer = SimpleNamespace(
        keys=torch.randn(1, 1, seq_len, 2),
        values=torch.randn(1, 1, seq_len, 2),
    )
    kwargs = {
        "hidden_states": hidden_states,
        "past_key_values": SimpleNamespace(layers=[cache_layer]),
        "cache_position": torch.tensor(cache_position),
    }
    return kwargs, [hidden_states, None]


@pytest.mark.parametrize(
    ("cache_position", "q_len", "expected"),
    [
        ([0], 1, True),
        ([1], 1, False),
        ([0, 1, 2], 3, True),
        ([3], 1, False),
        ([3, 4], 2, False),
    ],
)
def test_is_prefilling_uses_zero_based_cache_length(cache_position, q_len, expected):
    assert is_prefilling(torch.tensor(cache_position), q_len) is expected


@pytest.mark.parametrize(
    ("cache_position", "q_len", "expected_calls"),
    [
        ([0], 1, 1),
        ([1], 1, 0),
        ([0, 1, 2], 3, 1),
        ([3], 1, 0),
    ],
)
def test_base_press_compresses_only_during_prefill(cache_position, q_len, expected_calls):
    press = CountingPress()
    kwargs, output = make_hook_args(cache_position, q_len)

    press.forward_hook(SimpleNamespace(layer_idx=0), [], kwargs, output)

    assert press.calls == expected_calls


@pytest.mark.parametrize(
    ("cache_position", "q_len", "expected_steps"),
    [
        ([0], 1, 0),
        ([1], 1, 1),
        ([0, 1, 2], 3, 0),
        ([3], 1, 1),
    ],
)
def test_decoding_press_buffers_only_during_decoding(cache_position, q_len, expected_steps):
    press = DecodingPress(base_press=KnormPress(), compression_interval=128, target_size=128)
    kwargs, output = make_hook_args(cache_position, q_len)

    press.forward_hook(SimpleNamespace(layer_idx=0), [], kwargs, output)

    assert press.layer_step_counts[0] == expected_steps
    assert len(press.hidden_states_buffer[0]) == expected_steps


@pytest.mark.parametrize(
    ("cache_position", "q_len", "expected_prefill_calls", "expected_decoding_calls"),
    [
        ([0], 1, 1, 0),
        ([1], 1, 0, 1),
    ],
)
def test_prefill_decoding_press_routes_single_token_steps(
    cache_position,
    q_len,
    expected_prefill_calls,
    expected_decoding_calls,
):
    prefilling_press = HookRecorderPress()
    decoding_press = HookRecorderPress()
    press = PrefillDecodingPress(prefilling_press=prefilling_press, decoding_press=decoding_press)
    kwargs, output = make_hook_args(cache_position, q_len)

    press.forward_hook(SimpleNamespace(layer_idx=0), [], kwargs, output)

    assert prefilling_press.hook_calls == expected_prefill_calls
    assert decoding_press.hook_calls == expected_decoding_calls
