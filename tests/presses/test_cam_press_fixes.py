# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from dataclasses import dataclass, field

import pytest
import torch

from kvpress import AdaKVPress, CAMPress, ScorerPress, SnapKVPress


@dataclass
class FixedScorePress(ScorerPress):
    scores: list = field(default_factory=list)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return torch.tensor(self.scores, dtype=keys.dtype).expand(keys.shape[:3])


def test_cam_rejects_adakv_base_press():
    with pytest.raises(ValueError, match="requires a ScorerPress"):
        CAMPress(base_press=AdaKVPress(SnapKVPress()))


def test_cam_merged_tokens_are_not_kept_with_tied_scores():
    scores = [3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 5.0]
    seq_len, target_size = len(scores), 8
    press = CAMPress(base_press=FixedScorePress(scores=scores), target_size=target_size, merge_budget=2)
    module = types.SimpleNamespace(layer_idx=0)
    press.layer_step_counts[0] = seq_len - target_size  # all evicted tokens are merge candidates
    attentions = torch.ones(1, 1, seq_len)  # merge probability of 1
    press._running_attn_sum[0] = attentions.clone()

    keys = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1).expand(-1, -1, -1, seq_len)
    values = torch.eye(seq_len).view(1, 1, seq_len, seq_len)  # one-hot values identify merged tokens
    new_keys, new_values = press.compress(module, torch.randn(1, 4, 8), keys, values, attentions, {})

    kept = new_keys[0, 0, :, 0].long()
    merged = torch.nonzero(new_values[0, 0] * (1 - torch.eye(seq_len)[kept]))[:, 1].unique()
    assert len(kept) == target_size and len(merged) > 0
    assert not set(merged.tolist()) & set(kept.tolist())
