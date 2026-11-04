# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from kvpress import ChunkKVPress
from kvpress.presses.scorer_press import ScorerPress


class FixedScorer(ScorerPress):
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return kwargs["scores"]


def compress(scores, chunk_length=2):
    batch, _, length = scores.shape
    keys = torch.arange(batch * length, dtype=torch.float32).reshape(batch, 1, length, 1)
    values = keys + 100
    press = ChunkKVPress(FixedScorer(compression_ratio=0.5), chunk_length=chunk_length)
    return press.compress(
        SimpleNamespace(head_dim=1), torch.zeros(batch, length, 1), keys, values, None, {"scores": scores}
    )


def test_chunk_selection_is_independent_per_batch_element():
    scores = torch.tensor([[[9.0, 9.0, 1.0, 1.0]], [[1.0, 1.0, 9.0, 9.0]]])
    keys, values = compress(scores)
    torch.testing.assert_close(keys[:, 0, :, 0], torch.tensor([[0.0, 1.0], [6.0, 7.0]]))
    torch.testing.assert_close(values, keys + 100)


def test_partial_chunks_with_equal_lengths_are_selected_per_sample():
    scores = torch.tensor([[[9.0, 9.0, 1.0, 1.0, 0.0]], [[1.0, 1.0, 9.0, 9.0, 0.0]]])
    keys, _ = compress(scores)
    torch.testing.assert_close(keys[:, 0, :, 0], torch.tensor([[0.0, 1.0], [7.0, 8.0]]))


def test_partial_chunks_with_unequal_lengths_raise():
    scores = torch.tensor([[[9.0, 9.0, 1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0, 0.0, 9.0]]])
    with pytest.raises(ValueError, match="unequal cache lengths"):
        compress(scores)


def test_single_sample_can_select_partial_chunk():
    scores = torch.tensor([[[1.0, 1.0, 0.0, 0.0, 9.0]]])
    keys, _ = compress(scores)
    torch.testing.assert_close(keys[:, 0, :, 0], torch.tensor([[4.0]]))
