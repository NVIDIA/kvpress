# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

from kvpress import DecodingPress, ExpectedAttentionPress
from kvpress.presses.expected_attention_with_stats import ExpectedAttentionStatsPress
from tests.fixtures import unit_test_model  # noqa: F401


def rotate(module, keys, start):
    positions = torch.arange(start, start + keys.shape[2], device=keys.device).unsqueeze(0)
    cos, sin = module.rotary_emb(keys, positions)
    return keys * cos.unsqueeze(1) + rotate_half(keys) * sin.unsqueeze(1)


@pytest.mark.parametrize("with_stats", [False, True])
def test_scores_depend_on_relative_positions_only(unit_test_model, with_stats):  # noqa: F811
    """
    During decoding, DecodingPress passes the buffer of recent hidden states, so hidden_states.shape[1] is not
    the position of the next query. Shifting the cached keys and the next query by the same offset must not
    change the scores.
    """
    # float32 copies so that rotating keys by a large offset does not add half-precision noise
    module = deepcopy(unit_test_model.model.layers[0].self_attn).float()
    module.rotary_emb = deepcopy(unit_test_model.model.rotary_emb).float()
    config, device = unit_test_model.config, unit_test_model.device
    n_kv, head_dim = config.num_key_value_heads, module.head_dim
    k_len, buffer_len, shift = 64, 16, 1000

    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(1, buffer_len, config.hidden_size, generator=generator).to(device)
    keys = torch.randn(1, n_kv, k_len, head_dim, generator=generator).to(device)
    values = torch.randn(1, n_kv, k_len, head_dim, generator=generator).to(device)

    if with_stats:
        press = ExpectedAttentionStatsPress()
        shape = (config.num_hidden_layers, config.num_attention_heads, head_dim)
        press.mu = torch.randn(*shape, generator=generator).to(device)
        a = torch.randn(*shape, head_dim, generator=generator).to(device)
        press.cov = a @ a.transpose(-1, -2) / head_dim
    else:
        press = ExpectedAttentionPress()

    scores = [
        press.score(
            module,
            hidden_states,
            rotate(module, keys, start),
            values,
            None,
            {"position_ids": torch.tensor([[start + k_len - 1]], device=device)},
        )
        for start in (0, shift)
    ]
    torch.testing.assert_close(scores[0], scores[1], rtol=1e-3, atol=1e-4)


def test_decoding_uses_logical_positions(unit_test_model):  # noqa: F811
    """
    Decode with explicit position_ids, as KVPressTextGenerationPipeline does. After the first compression the
    cache is shorter than the sequence, so neither the buffer length nor the cache length is the position of
    the next query.
    """
    device = unit_test_model.device
    n_layers = unit_test_model.config.num_hidden_layers
    press = DecodingPress(base_press=ExpectedAttentionPress(), compression_interval=4, target_size=16)
    input_ids = torch.randint(0, unit_test_model.config.vocab_size, (1, 44), device=device)

    positions = []
    apply_avg_rope = ExpectedAttentionPress.apply_avg_rope

    def record_position(self, module, mu, cov, q_len):
        positions.append(q_len)
        return apply_avg_rope(self, module, mu, cov, q_len)

    with (
        torch.no_grad(),
        patch.object(ExpectedAttentionPress, "apply_avg_rope", record_position),
        press(unit_test_model),
    ):
        cache = DynamicCache()
        unit_test_model(input_ids=input_ids[:, :32], past_key_values=cache)
        for t in range(32, 44):
            position_ids = torch.tensor([[t]], device=device)
            unit_test_model(input_ids=input_ids[:, t : t + 1], past_key_values=cache, position_ids=position_ids)

    # Compressions run on the 4th, 8th and 12th decoding steps, i.e. at positions 35, 39 and 43
    assert cache.get_seq_length() == 16
    assert positions == [36] * n_layers + [40] * n_layers + [44] * n_layers
