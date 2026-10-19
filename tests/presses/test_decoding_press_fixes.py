# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest
import torch
from transformers import DynamicCache

from kvpress import AdaKVPress, CAMPress, DecodingPress, KnormPress, PrefillDecodingPress, ScorerPress
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


@dataclass
class RecordingKnormPress(ScorerPress):
    calls: list = field(default_factory=list)
    fail: bool = False

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.fail:
            raise RuntimeError("scoring failed")
        self.calls.append((hidden_states, kwargs["position_embeddings"], attentions))
        return -keys.norm(dim=-1)


@pytest.mark.parametrize("press_cls", [DecodingPress, CAMPress])
@torch.no_grad()
def test_buffered_hidden_states_come_with_their_rope_embeddings(
    unit_test_model_output_attention, press_cls  # noqa: F811
):
    model = unit_test_model_output_attention
    base_press = RecordingKnormPress()
    with press_cls(base_press=base_press, compression_interval=4, target_size=12)(model):
        model.generate(torch.randint(0, 1000, (1, 16), device=model.device), max_new_tokens=6, do_sample=False)

    # The first compression happens after the decoding steps at positions 16, 17, 18 and 19
    hidden_states, (cos, sin), _ = base_press.calls[0]
    assert hidden_states.shape[1] == 4
    position_ids = torch.arange(16, 20, device=model.device).unsqueeze(0)
    expected_cos, expected_sin = model.model.rotary_emb(hidden_states, position_ids)
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)
    # Attention weights of the eager model only cover the current query
    assert all(attentions is None for _, _, attentions in base_press.calls)


@torch.no_grad()
def test_state_is_reset_on_prefill(unit_test_model):  # noqa: F811
    press = DecodingPress(base_press=KnormPress(), compression_interval=10, target_size=64)
    input_ids = torch.randint(0, 1000, (1, 16), device=unit_test_model.device)
    with press(unit_test_model):
        unit_test_model.generate(input_ids, max_new_tokens=4, do_sample=False)
        assert set(press.layer_step_counts.values()) == {3}

        unit_test_model(input_ids, past_key_values=DynamicCache())
        assert set(press.layer_step_counts.values()) == {0}
        assert all(len(buffer) == 0 for buffer in press.hidden_states_buffer.values())


@pytest.mark.parametrize("press_cls", [DecodingPress, CAMPress])
@torch.no_grad()
def test_decoding_compression_rejects_keys_masked_during_prefilling(unit_test_model, press_cls):  # noqa: F811
    press = PrefillDecodingPress(
        prefilling_press=AdaKVPress(KnormPress(compression_ratio=0.5)),
        decoding_press=press_cls(base_press=KnormPress(), compression_interval=2, target_size=8),
    )
    input_ids = torch.randint(0, 1000, (1, 16), device=unit_test_model.device)
    with pytest.raises(ValueError, match="head-wise press"):
        with press(unit_test_model):
            unit_test_model.generate(input_ids, max_new_tokens=4, do_sample=False)
