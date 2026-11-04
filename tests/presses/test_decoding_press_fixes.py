# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest
import torch

from kvpress import CAMPress, DecodingPress, ScorerPress
from tests.fixtures import unit_test_model_output_attention  # noqa: F401


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
