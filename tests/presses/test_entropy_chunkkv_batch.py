# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from kvpress import EntropyGatedChunkKVPress
from kvpress.presses.scorer_press import ScorerPress


class FixedScorer(ScorerPress):
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return kwargs["scores"]


@pytest.mark.parametrize(
    "scores",
    [
        [[[9.0, 9.0, 1.0, 1.0]], [[1.0, 1.0, 9.0, 9.0]]],
        [[[-2.0, -1.0, 4.0, 1.0, 0.0]], [[5.0, -1.0, -4.0, -2.0, 8.0]]],
        [[[4.0, 3.0, 0.0, 1.0, 2.0, 8.0]], [[0.0, 1.0, 9.0, 8.0, 2.0, 1.0]]],
    ],
)
def test_batched_selection_matches_independent_samples(scores):
    scores = torch.tensor(scores)
    batch, _, length = scores.shape
    keys = torch.arange(batch * length, dtype=torch.float32).reshape(batch, 1, length, 1)
    values = keys + 100
    press = EntropyGatedChunkKVPress(FixedScorer(compression_ratio=0.5), chunk_length=2, low_entropy_chunk_length=1)
    module = SimpleNamespace(head_dim=1)
    hidden = torch.zeros(batch, length, 1)

    batched_keys, batched_values = press.compress(module, hidden, keys, values, None, {"scores": scores})
    independent = [
        press.compress(
            module, hidden[i : i + 1], keys[i : i + 1], values[i : i + 1], None, {"scores": scores[i : i + 1]}
        )
        for i in range(batch)
    ]
    torch.testing.assert_close(batched_keys, torch.cat([item[0] for item in independent]))
    torch.testing.assert_close(batched_values, torch.cat([item[1] for item in independent]))
    assert batched_keys.shape[2] == max(1, int(length * 0.5))
